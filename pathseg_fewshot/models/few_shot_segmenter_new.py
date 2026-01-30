from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg_fewshot.models.histo_encoder import Encoder


LabelMode = Literal["soft_avgpool", "hard_nearest"]


class MetaLearnerBase(nn.Module):
    """
    Episode head interface.
    Works on *token features*.

    Convention:
      - support_feats: [S, T, Fp]
      - support_labels: [S, T, C]   (soft or hard one-hot)
      - support_valid: [S, T] bool  (optional)
      - query_feats: [Q, T, Fp]

    Return:
      - ctx: dict with anything
      - logits: either [Q*T, C] or [Q, C, Hp, Wp] depending on implementation.
        (Here we standardize on flat: [Q*T, C].)
    """

    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,
        support_labels: torch.Tensor,
        support_valid: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,
        support_ctx: Dict[str, torch.Tensor],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    # [B,T,F] -> [B*T,F]
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,F], got {x.shape}")
    B, T, Fdim = x.shape
    return x.reshape(B * T, Fdim)


def _unflatten_logits(
    logits_flat: torch.Tensor, Q: int, grid_size: Tuple[int, int]
) -> torch.Tensor:
    # [Q*T,C] -> [Q,C,Hp,Wp]
    Hp, Wp = grid_size
    T = Hp * Wp
    if logits_flat.shape[0] != Q * T:
        raise ValueError(f"Expected first dim Q*T={Q * T}, got {logits_flat.shape[0]}")
    C = logits_flat.shape[1]
    return logits_flat.view(Q, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()


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


class FewShotSegmenterV2(nn.Module):
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
        meta_learner: Optional[MetaLearnerBase] = None,
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
            proj_out = self.proj.out_dim

        self.label_mode: LabelMode = label_mode

        # meta-learner is injected so you can swap easily from config
        if meta_learner is None:
            raise ValueError("meta_learner must be provided (e.g., Proto/PANet head).")
        self.meta = meta_learner

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

    def _encode_support_labels(
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
        support_labels, support_valid = self._encode_support_labels(
            support_masks=support_masks,
            num_fg_classes=K,
        )  # [S,T,K+1], [S,T]

        ctx = self.meta.fit_support(
            support_feats=support_feats,
            support_labels=support_labels,
            support_valid=support_valid,
            grid_size=self.grid_size,
        )
        return ctx

    def predict_query(
        self,
        *,
        query_imgs: torch.Tensor,  # [Q,C,H,W]
        support_ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        Q = query_imgs.shape[0]
        query_feats = self.encode(query_imgs)  # [Q,T,Fp]

        logits_flat = self.meta.predict_query(
            query_feats=query_feats,
            support_ctx=support_ctx,
            grid_size=self.grid_size,
        )  # expected [Q*T, C]

        return _unflatten_logits(
            logits_flat, Q=Q, grid_size=self.grid_size
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
