from __future__ import annotations

from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg_fewshot.models.histo_encoder import Encoder
from pathseg_fewshot.models.metalearner import MetaLearnerBase
from pathseg_fewshot.models.prompt_metalearner import PromptMetaLearnerWithMaxC


LabelMode = Literal["soft_avgpool", "hard_nearest"]


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    # [B,T,F] -> [B*T,F]
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,F], got {x.shape}")
    B, T, Fdim = x.shape
    return x.reshape(B * T, Fdim)


def masks_to_soft_patch_labels(
    masks: torch.Tensor,  # [S,H,W] values {0..K} or ignore
    patch_size: Tuple[int, int],
    num_fg_classes: int,  # K
    ignore_index: int = 255,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      soft:  [S, K+1, Hp, Wp] float
      valid: [S, Hp, Wp] bool
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be [S,H,W], got {masks.shape}")

    S, H, W = masks.shape
    ph, pw = patch_size
    if (H % ph) != 0 or (W % pw) != 0:
        raise ValueError("H and W must be divisible by patch_size.")

    K = int(num_fg_classes)
    Hp, Wp = H // ph, W // pw
    patch_area = float(ph * pw)

    valid = masks != ignore_index  # [S,H,W]
    valid_cnt = (
        F.avg_pool2d(valid.float().unsqueeze(1), (ph, pw), stride=(ph, pw)) * patch_area
    ).squeeze(1)  # [S,Hp,Wp]
    patch_valid = valid_cnt > 0

    # fg one-hot over 1..K (exclude bg=0), masked by valid
    if K > 0:
        class_idx = torch.arange(
            1, K + 1, device=masks.device, dtype=masks.dtype
        )  # [K]
        fg_one_hot = (
            masks.unsqueeze(1) == class_idx.view(1, K, 1, 1)
        ) & valid.unsqueeze(1)  # [S,K,H,W]
        fg_sum = (
            F.avg_pool2d(fg_one_hot.float(), (ph, pw), stride=(ph, pw)) * patch_area
        )  # [S,K,Hp,Wp]
        fg_soft = fg_sum / (valid_cnt.unsqueeze(1) + eps)  # [S,K,Hp,Wp]
        bg_soft = (1.0 - fg_soft.sum(dim=1, keepdim=True)).clamp(
            0.0, 1.0
        )  # [S,1,Hp,Wp]
        soft = torch.cat([bg_soft, fg_soft], dim=1)  # [S,K+1,Hp,Wp]
    else:
        soft = masks.new_zeros((S, 1, Hp, Wp), dtype=torch.float32)
        soft[:, 0] = patch_valid.float()

    soft = torch.where(patch_valid.unsqueeze(1), soft, torch.zeros_like(soft))
    return soft, patch_valid


def masks_to_hard_token_onehot(
    masks: torch.Tensor,  # [S,H,W] values {0..K} or ignore
    grid_size: Tuple[int, int],  # (Hp,Wp)
    num_fg_classes: int,  # K
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      onehot: [S, K+1, Hp, Wp] float (hard one-hot at token grid)
      valid:  [S, Hp, Wp] bool
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be [S,H,W], got {masks.shape}")

    Hp, Wp = grid_size
    K = int(num_fg_classes)
    C = K + 1

    mask_grid = (
        F.interpolate(masks.unsqueeze(1).float(), size=(Hp, Wp), mode="nearest")
        .squeeze(1)
        .long()
    )  # [S,Hp,Wp]

    valid = mask_grid != ignore_index
    safe = mask_grid.clamp(min=0, max=C - 1)
    onehot = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()  # [S,C,Hp,Wp]
    onehot = onehot * valid.unsqueeze(1).float()
    return onehot, valid


class IdentityProj(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearProj(nn.Module):
    """
    Reasonable default for feature adaptation:
      - Linear to proj_dim
      - (optional) LayerNorm
      - (optional) activation

    For foundation/ViT token features, Linear+LayerNorm is very standard.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        *,
        layernorm: bool = True,
        activation: Optional[Literal["gelu", "relu"]] = "gelu",
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, proj_dim)]
        if layernorm:
            layers.append(nn.LayerNorm(proj_dim))
        if activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.net = nn.Sequential(*layers)
        self.out_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_meta_learner(
    meta_learner_id: str, input_dim: int, center=False
) -> MetaLearnerBase:
    if meta_learner_id == "prototype_head":
        from pathseg_fewshot.models.metalearner import PrototypeHead

        return PrototypeHead(embed_dim=input_dim, center=center)
    elif meta_learner_id == "prototype_mixture_bank_head":
        from pathseg_fewshot.models.prototype_mixture_head import (
            PrototypeMixtureBankHead,
        )

        return PrototypeMixtureBankHead(
            embed_dim=input_dim,
            center=center,
            bank_size=256,
            presence_beta=20.0,
            add_diversity_loss=True,
            add_usage_loss=True,
        )
    elif meta_learner_id == "gaussian_mixture_bank_head":
        from pathseg_fewshot.models.prototype_mixture_head import (
            GaussianMixtureBankHead,
        )

        return GaussianMixtureBankHead(
            embed_dim=input_dim,
            center=center,
            bank_size=32,
        )

    else:
        raise ValueError(f"Unknown meta_learner_id: {meta_learner_id}")


class FewShotSegmenter(nn.Module):
    """
    Clean episode-conditioned segmenter:
      - encoder: images -> token feats [B, T, F]
      - proj:    [B, T, F] -> [B, T, Fp]
      - meta:    per-episode head operating on [S,T,Fp] / [Q,T,Fp]

    Label encoding (soft/hard) is handled here.
    """

    def __init__(
        self,
        *,
        encoder_id: str = "h0-mini",
        ignore_index: int = 255,
        proj_dim: Optional[int] = 256,
        proj_layernorm: bool = True,
        proj_activation: Optional[Literal["gelu", "relu"]] = "gelu",
        label_mode: LabelMode = "soft_avgpool",
        meta_learner_id: str = "prototype_head",
        center_features: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(encoder_id)
        self.grid_size: Tuple[int, int] = self.encoder.grid_size  # (Hp,Wp)
        self.patch_size: Tuple[int, int] = self.encoder.patch_size
        self.ignore_index = int(ignore_index)

        in_dim = int(self.encoder.embed_dim)

        if proj_dim is None or proj_dim == in_dim:
            self.proj = IdentityProj(in_dim)
            proj_out = in_dim
        else:
            self.proj = LinearProj(
                in_dim=in_dim,
                proj_dim=int(proj_dim),
                layernorm=proj_layernorm,
                activation=proj_activation,
            )

            proj_out = proj_dim

        # self.proj = IdentityProj(in_dim)
        self.label_mode: LabelMode = label_mode

        # meta-learner is injected so you can swap easily from config

        self.meta = build_meta_learner(
            meta_learner_id,
            input_dim=proj_out,
            center=center_features,
        )
        # self.meta = PromptMetaLearnerWithMaxC(d_model=in_dim, max_classes=5)

        # Optional: sanity check if meta expects certain dim (not enforced here).
        self.proj_out_dim = proj_out

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,C,H,W]
        returns projected token feats: [B,T,Fp]
        """
        feats = self.encoder(images)  # [B,T,F]
        feats = self.proj(feats)  # [B,T,Fp]
        return feats

    def encode_support_labels(
        self,
        support_masks: torch.Tensor,  # [S,H,W] episode ids {0..K,255}
        num_fg_classes: int,  # K
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          labels: [S,T,C] float (C=K+1)
          valid:  [S,T] bool
        """
        S, H, W = support_masks.shape
        Hp, Wp = self.grid_size
        T = Hp * Wp
        K = int(num_fg_classes)

        if self.label_mode == "soft_avgpool":
            soft, valid = masks_to_soft_patch_labels(
                support_masks,
                patch_size=self.patch_size,
                num_fg_classes=K,
                ignore_index=self.ignore_index,
            )  # [S,K+1,Hp,Wp], [S,Hp,Wp]
        elif self.label_mode == "hard_nearest":
            soft, valid = masks_to_hard_token_onehot(
                support_masks,
                grid_size=self.grid_size,
                num_fg_classes=K,
                ignore_index=self.ignore_index,
            )  # [S,K+1,Hp,Wp], [S,Hp,Wp]
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        labels = soft.permute(0, 2, 3, 1).reshape(S, T, K + 1)  # [S,T,C]
        valid = valid.reshape(S, T)  # [S,T]
        return labels, valid

    def fit_support_features(
        self,
        *,
        support_features: torch.Tensor,  # [S,T,Fp]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,  # [S,T] bool
    ) -> Dict[str, torch.Tensor]:
        return self.meta.fit_support(
            support_feats=support_features,
            support_labels=support_labels,
            support_valid=support_valid,
        )

    def fit_support(
        self,
        *,
        support_imgs: torch.Tensor,  # [S,C,H,W]
        support_masks: torch.Tensor,  # [S,H,W] episode ids
        episode_class_ids: torch.Tensor,  # [K] global ids (kept for API, can be unused)
    ) -> Dict[str, torch.Tensor]:
        # NOTE: we assume support_masks already mapped to episode ids {0..K,255}.
        # If not, do the mapping outside (dataset) or add mapping here.
        K = int(episode_class_ids.numel())  # number of foreground ways
        support_feats = self.encode(support_imgs)  # [S,T,Fp]
        support_labels, support_valid = self.encode_support_labels(
            support_masks=support_masks,
            num_fg_classes=K,
        )  # [S,T,K+1], [S,T]

        return self.fit_support_features(
            support_features=support_feats,
            support_labels=support_labels,
            support_valid=support_valid,
        )

    def predict_query_features(
        self,
        *,
        query_features: torch.Tensor,  # [Q,C,H,W]
        support_ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.meta.predict_query(
            query_feats=query_features,
            support_ctx=support_ctx,
        )  # expected [Q*T, C]

    def predict_query(
        self,
        *,
        query_imgs: torch.Tensor,  # [Q,C,H,W]
        support_ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        Q = query_imgs.shape[0]
        query_feats = self.encode(query_imgs)  # [Q,T,Fp]

        logits_flat = self.predict_query_features(
            query_features=query_feats,
            support_ctx=support_ctx,
        )  # [Q*T,C]

        return self.unflatten_logits(
            logits_flat,
            num_query=Q,
        )  # [Q,C,Hp,Wp]

    def forward(
        self,
        *,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor,
        query_imgs: torch.Tensor,
        episode_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.fit_support(
            support_imgs=support_imgs,
            support_masks=support_masks,
            episode_class_ids=episode_class_ids,
        )
        return self.predict_query(query_imgs=query_imgs, support_ctx=ctx)

    def forward_features(
        self,
        *,
        support_features: torch.Tensor,  # [S,T,Fp]
        support_labels: torch.Tensor,  # [S,T,C]
        query_features: torch.Tensor,  # [Q,T,Fp]
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> torch.Tensor:
        ctx = self.fit_support_features(
            support_features=support_features,
            support_labels=support_labels,
            support_valid=support_valid,
        )
        return self.predict_query_features(
            query_features=query_features, support_ctx=ctx
        )  # [Q*T,C]

    def unflatten_logits(
        self,
        logits_flat: torch.Tensor,
        num_query: int,
    ) -> torch.Tensor:
        # [Q*T,C] -> [Q,C,Hp,Wp]
        Hp, Wp = self.grid_size
        T = Hp * Wp
        if logits_flat.shape[0] != num_query * T:
            raise ValueError(
                f"Expected first dim Q*T={num_query * T}, got {logits_flat.shape[0]}"
            )
        C = logits_flat.shape[1]
        return logits_flat.view(num_query, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()


class FewShotSegmenterFMUpsamling(FewShotSegmenter):
    """
    Clean episode-conditioned segmenter:
      - encoder: images -> token feats [B, T, F]
      - proj:    [B, T, F] -> [B, T, Fp]
      - meta:    per-episode head operating on [S,T,Fp] / [Q,T,Fp]

    Label encoding (soft/hard) is handled here.
    """

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,C,H,W]
        returns projected token feats: [B,T,Fp]
        """
        feats = self.encoder(images)  # [B,T,F]
        feats = self.proj(feats)  # [B,T,Fp]
        featues = F.interpolate(
            feats.permute(0, 2, 1).view(
                images.size(0), -1, self.grid_size[0], self.grid_size[1]
            ),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        feats = featues.permute(0, 2, 3, 1).view(images.size(0), -1, feats.size(-1))
        return feats

    def encode_support_labels(
        self,
        support_masks: torch.Tensor,  # [S,H,W] episode ids {0..K,255}
        num_fg_classes: int,  # K
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          labels: [S,T,C] float (C=K+1)
          valid:  [S,T] bool
        """
        S, H, W = support_masks.shape
        Hp, Wp = self.grid_size
        T = Hp * Wp
        K = int(num_fg_classes)

        if self.label_mode == "soft_avgpool":
            soft, valid = masks_to_soft_patch_labels(
                support_masks,
                patch_size=self.patch_size,
                num_fg_classes=K,
                ignore_index=self.ignore_index,
            )  # [S,K+1,Hp,Wp], [S,Hp,Wp]
        elif self.label_mode == "hard_nearest":
            soft, valid = masks_to_hard_token_onehot(
                support_masks,
                grid_size=self.grid_size,
                num_fg_classes=K,
                ignore_index=self.ignore_index,
            )  # [S,K+1,Hp,Wp], [S,Hp,Wp]
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        labels = soft.permute(0, 2, 3, 1).reshape(S, T, K + 1)  # [S,T,C]
        valid = valid.reshape(S, T)  # [S,T]
        return labels, valid
