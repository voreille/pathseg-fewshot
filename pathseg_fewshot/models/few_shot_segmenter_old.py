from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg_fewshot.models.histo_encoder import Encoder


def masks_to_soft_patch_labels(
    masks: torch.Tensor,  # [S, H, W], values in {0..K} or ignore_index
    patch_size,  # (ph, pw)
    num_classes: int,  # K (foreground classes, excluding bg)
    ignore_index: int = 255,
    eps: float = 1e-12,
):
    """
    Assumes masks are already remapped:
      0 = background
      1..K = foreground classes
      ignore_index = unlabeled pixels (excluded from valid count)

    Returns:
      soft:       [S, K+1, Hp, Wp] float in [0,1], fraction among VALID pixels per patch
                  channel 0 is background
                  channels 1..K correspond to classes 1..K
      patch_valid:[S, Hp, Wp]      bool  (at least 1 valid pixel in patch)
      valid_frac: [S, Hp, Wp]      float (valid_pixels / (ph*pw))
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be [S,H,W], got {masks.shape}")

    S, H, W = masks.shape
    ph, pw = patch_size

    if (H % ph) != 0 or (W % pw) != 0:
        raise ValueError("H and W must be divisible by patch_size.")

    K = int(num_classes)
    if K < 0:
        raise ValueError(f"num_classes must be >= 0, got {K}")

    valid = masks != ignore_index  # [S,H,W]
    patch_area = float(ph * pw)

    # valid pixel count per patch
    valid_cnt = (
        F.avg_pool2d(valid.float().unsqueeze(1), (ph, pw), stride=(ph, pw)) * patch_area
    )  # [S,1,Hp,Wp]
    valid_cnt2d = valid_cnt.squeeze(1)  # [S,Hp,Wp]
    patch_valid = valid_cnt2d > 0
    valid_frac = valid_cnt2d / patch_area

    Hp, Wp = H // ph, W // pw

    if K == 0:
        # only bg channel exists
        soft = masks.new_zeros((S, 1, Hp, Wp), dtype=torch.float32)
        # bg among valid pixels is 1 when patch has any valid pixels
        soft[:, 0] = patch_valid.float()
        return soft, patch_valid, valid_frac

    # foreground one-hot for classes 1..K (exclude bg=0), masked by valid
    class_idx = torch.arange(1, K + 1, device=masks.device, dtype=masks.dtype)  # [K]
    fg_one_hot = (masks.unsqueeze(1) == class_idx.view(1, K, 1, 1)) & valid.unsqueeze(
        1
    )  # [S,K,H,W]

    fg_sum = (
        F.avg_pool2d(fg_one_hot.float(), (ph, pw), stride=(ph, pw)) * patch_area
    )  # [S,K,Hp,Wp]

    fg_soft = fg_sum / (valid_cnt + eps)  # [S,K,Hp,Wp]

    # background fraction among valid pixels = 1 - sum(fg fractions)
    bg_soft = (1.0 - fg_soft.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)  # [S,1,Hp,Wp]

    soft = torch.cat([bg_soft, fg_soft], dim=1)  # [S,K+1,Hp,Wp]

    # zero patches with no valid pixels
    soft = torch.where(patch_valid.unsqueeze(1), soft, torch.zeros_like(soft))

    return soft, patch_valid, valid_frac


class FewShotSegmenter(nn.Module):
    """
    Few-shot segmentation model (episode-conditioned).

    Responsibilities:
      - encode images into feature maps via `encoder`
      - estimate episode-specific parameters from the support set via `meta_learner.fit(...)`
      - produce query logits via `meta_learner.predict(...)`

    Expected tensor conventions:
      - support_imgs: [S, C, H, W]
      - support_gt:   [S, H, W]  (global ids)
      - query_imgs:   [Q, C, H, W]
      - output logits: [Q, N, H', W'] (N = #episode classes; spatial size depends on encoder/head)
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        meta_learner_id: str = "meta_linear_head_mc_attn",
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder_id)

        if meta_learner_id == "meta_linear_head_mc_attn":
            from pathseg_fewshot.models.meta_linear_head import MetaLinearHeadMC_Attn

            self.meta_learner = MetaLinearHeadMC_Attn(
                embed_dim=self.encoder.embed_dim,
                h_dim=256,
            )
        elif meta_learner_id == "prototype_head":
            from pathseg_fewshot.models.prototype_head import (
                PrototypeHead,
                PrototypeHeadV2,
            )

            self.meta_learner = PrototypeHeadV2(
                embed_dim=self.encoder.embed_dim,
                center=True,
                learnable_temp=False,
                normalize=True,
            )

        self.grid_size = self.encoder.grid_size  # type: ignore[attr-defined]
        self.patch_size = self.encoder.patch_size  # type: ignore[attr-defined]
        self.ignore_index = ignore_index

    def _flatten_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        - [B, F, Hp, Wp]
        - [B, F, Hp*Wp]
        - [B, Hp*Wp, F]   <-- your case

        Returns:
        - [B*Hp*Wp, F]
        """
        B = feats.shape[0]
        Hp, Wp = self.grid_size

        if feats.ndim == 4:
            return feats.permute(0, 2, 3, 1).reshape(B * Hp * Wp, -1)

        if feats.ndim == 3:
            # case A: [B, F, Hp*Wp]
            if feats.shape[1] == self.encoder.embed_dim and feats.shape[2] == Hp * Wp:
                return feats.permute(0, 2, 1).reshape(B * Hp * Wp, -1)

            # case B: [B, Hp*Wp, F]  (your case)
            if feats.shape[1] == Hp * Wp:
                return feats.reshape(B * Hp * Wp, -1)

        raise ValueError(f"Unexpected feats shape: {feats.shape}")

    def fit_support(
        self,
        *,
        support_imgs: torch.Tensor,  # [S,C,H,W]
        support_masks: torch.Tensor,  # [S,H,W]
        episode_class_ids: torch.Tensor,  # [N]
    ):
        feat_s = self.encoder(support_imgs)  # [S,Hp*Wp,F]

        # patch-level soft labels + patch validity
        support_soft, patch_valid, _ = masks_to_soft_patch_labels(
            support_masks,
            num_classes=len(episode_class_ids),
            patch_size=self.patch_size,
            ignore_index=self.ignore_index,
        )  # support_soft: [S,N,Hp,Wp], patch_valid: [S,Hp,Wp]

        # flatten labels to [S*Hp*Wp, N]
        S, N, Hp, Wp = support_soft.shape
        y = support_soft.permute(0, 2, 3, 1).reshape(S * Hp * Wp, N)  # [S*Hp*Wp, N]
        v = patch_valid.reshape(S * Hp * Wp)  # [S*Hp*Wp]

        # flatten feats to [S*Hp*Wp, F]
        feats = self._flatten_feats(feat_s)  # [S*Hp*Wp, F]

        # slice valid patches
        feats = feats[v]  # [N_valid, F]
        y = y[v]  # [N_valid, N]

        support_ctx = self.meta_learner.fit_support(
            support_feats=feats,
            support_labels=y,
        )
        return support_ctx

    def predict_query(
        self,
        *,
        query_imgs: torch.Tensor,  # [Q,C,H,W]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        feat_q = self.encoder(query_imgs)  # [Q, Hp*Wp, F]
        q_flat = self._flatten_feats(feat_q)  # [Q*Hp*Wp, F]

        logits_flat = self.meta_learner.predict_query(
            query_feats=q_flat, support_ctx=support_ctx
        )  # [Q*Hp*Wp, N]

        Q = query_imgs.shape[0]
        Hp, Wp = self.grid_size
        N = logits_flat.shape[1]

        return (
            logits_flat.view(Q, Hp, Wp, N).permute(0, 3, 1, 2).contiguous()
        )  # [Q,N,Hp,Wp]

    def forward(
        self,
        *,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs: torch.Tensor,
        episode_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience forward: fit on support, then predict on query.
        """
        support_ctx = self.fit_support(
            support_imgs=support_imgs,
            support_masks=support_masks,
            episode_class_ids=episode_class_ids,
        )
        return self.predict_query(query_imgs=query_imgs, support_ctx=support_ctx)


class FewShotSegmenterV2(nn.Module):
    """
    Few-shot segmentation model (episode-conditioned).

    Responsibilities:
      - encode images into feature maps via `encoder`
      - estimate episode-specific parameters from the support set via `meta_learner.fit(...)`
      - produce query logits via `meta_learner.predict(...)`

    Expected tensor conventions:
      - support_imgs: [S, C, H, W]
      - support_gt:   [S, H, W]  (global ids)
      - query_imgs:   [Q, C, H, W]
      - output logits: [Q, N, H', W'] (N = #episode classes; spatial size depends on encoder/head)
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        ignore_index: int = 255,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder_id)

        self.grid_size = self.encoder.grid_size  # type: ignore[attr-defined]
        self.patch_size = self.encoder.patch_size  # type: ignore[attr-defined]
        self.ignore_index = ignore_index
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
        from pathseg_fewshot.models.prototype_head import (
            PrototypeHeadV2,
        )

        self.meta_learner = PrototypeHeadV2(
            embed_dim=self.encoder.embed_dim,
            center=True,
            learnable_temp=False,
            normalize=False,
        )

    def fit_support(
        self,
        *,
        support_imgs: torch.Tensor,  # [S,C,H,W]
        support_masks: torch.Tensor,  # [S,H,W]
        episode_class_ids: torch.Tensor,  # [N]
    ):
        S, C, H, W = support_imgs.shape
        feat_s = self.encoder(support_imgs)  # [S,Hp*Wp, F]
        feat_s = self.proj(feat_s)  # [S,Hp*Wp,proj_dim]
        # feat_s = F.interpolate(


        # patch-level soft labels + patch validity

        return support_ctx

    def predict_query(
        self,
        *,
        query_imgs: torch.Tensor,  # [Q,C,H,W]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        feat_q = self.encoder(query_imgs)  # [Q, Hp*Wp, F]
        q_flat = self._flatten_feats(feat_q)  # [Q*Hp*Wp, F]

        logits_flat = self.meta_learner.predict_query(
            query_feats=q_flat, support_ctx=support_ctx
        )  # [Q*Hp*Wp, N]

        Q = query_imgs.shape[0]
        Hp, Wp = self.grid_size
        N = logits_flat.shape[1]

        return (
            logits_flat.view(Q, Hp, Wp, N).permute(0, 3, 1, 2).contiguous()
        )  # [Q,N,Hp,Wp]

    def forward(
        self,
        *,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs: torch.Tensor,
        episode_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience forward: fit on support, then predict on query.
        """
        support_ctx = self.fit_support(
            support_imgs=support_imgs,
            support_masks=support_masks,
            episode_class_ids=episode_class_ids,
        )
        return self.predict_query(query_imgs=query_imgs, support_ctx=support_ctx)
