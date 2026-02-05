from typing import Any, Dict, Optional, Tuple, Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class PrototypeHead(MetaLearnerBase):
    """
    Prototype head on token features.

    support_labels is assumed to be [N, C] where:
      - channel 0 is background
      - channels 1..K are foreground ways
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        normalize: bool = True,
        center: bool = True,
        bg_mode: str = "global",  # "global" | "panet_relative"
        learnable_temp: bool = True,
        init_temp: float = 20.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.normalize = bool(normalize)
        self.center = bool(center)
        self.bg_mode = str(bg_mode)
        self.eps = float(eps)

        if self.bg_mode not in {"global", "panet_relative"}:
            raise ValueError(
                f"bg_mode must be 'global' or 'panet_relative', got {self.bg_mode!r}"
            )

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

    def _apply_mask(
        self, feats: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None
    ):
        if mask is None:
            return feats, labels
        m = mask.to(dtype=feats.dtype).unsqueeze(-1)  # [N,1]
        return feats * m, labels * m

    def _weighted_mean(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        denom = labels.sum(dim=0).clamp_min(self.eps)  # [C]
        num = labels.transpose(0, 1) @ feats  # [C,H]
        return num / denom.unsqueeze(-1)

    def _compute_prototypes(
        self, feats: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        feats:  [N,H]
        labels: [N,C] with C = K+1 (bg + K ways)
        returns prototypes [C,H]
        """
        if labels.shape[1] < 1:
            raise ValueError("labels must have at least 1 channel (background).")

        if self.bg_mode == "global":
            return self._weighted_mean(feats, labels)

        # PANet-style relative background:
        # - keep fg prototypes from labels[:, 1:]
        # - bg prototype = mean_k prototype(not fg_k)
        C = labels.shape[1]
        if C < 2:
            # no foreground ways -> only bg
            return self._weighted_mean(feats, labels)

        y_fg = labels[:, 1:]  # [N,K]
        # fg prototypes: [K,H]
        proto_fg = self._weighted_mean(feats, y_fg)  # treats columns independently

        # per-way relative bg weights: bg_k = 1 - fg_k
        y_bgk = (1.0 - y_fg).clamp(0.0, 1.0)  # [N,K]

        # bg prototype per way: [K,H]
        proto_bgk = self._weighted_mean(feats, y_bgk)

        # mean over ways -> [1,H]
        proto_bg = proto_bgk.mean(dim=0, keepdim=True)

        # stack back to [K+1,H]
        prototypes = torch.cat([proto_bg, proto_fg], dim=0)
        assert prototypes.shape[0] == C
        return prototypes

    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be 3D: [S,T,E] and support_labels must be 3D: [S,T,C]."
            )
        if support_feats.shape[0] != support_labels.shape[0]:
            raise ValueError("support_feats and support_labels must share N.")

        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected support_feats dim=E={self.embed_dim}, got {support_feats.shape[1]}."
            )
        support_feats = support_feats.reshape(-1, support_feats.shape[-1])  # [N,E]
        support_labels = support_labels.reshape(-1, support_labels.shape[-1])  # [N,C]
        if support_valid is not None:
            support_valid = support_valid.reshape(-1)  # [N]

        # optional centering computed on valid tokens only (mask), BEFORE normalization
        if self.center:
            if support_valid is not None:
                m = support_valid.float().unsqueeze(-1)
                denom = m.sum().clamp_min(self.eps)
                mu = (support_feats * m).sum(dim=0) / denom
            else:
                mu = support_feats.mean(dim=0)
            support_feats = support_feats - mu
        else:
            mu = None

        if self.normalize:
            support_feats = F.normalize(support_feats, dim=-1)

        support_feats, labels = self._apply_mask(support_feats, support_labels, support_valid)

        prototypes = self._compute_prototypes(support_feats, labels)  # [C,H]

        if self.normalize:
            prototypes = F.normalize(prototypes, dim=-1)

        ctx = {"prototypes": prototypes}
        if mu is not None:
            ctx["center"] = mu
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,
        support_ctx: dict[str, Any],
    ) -> torch.Tensor:
        q = query_feats.reshape(-1, query_feats.shape[-1])  # [N,E]

        if self.center and "center" in support_ctx:
            q = q - support_ctx["center"]

        if self.normalize:
            q = F.normalize(q, dim=-1)

        logits = q @ support_ctx["prototypes"].t()
        return logits * torch.exp(self.log_temp).clamp_min(1e-3)


class PrototypeHeadV2(nn.Module):
    """
    One-vs-rest (per-way) prototype head.

    support_labels: [N, C] where C = K+1
      - channel 0: background (not used for relative bg)
      - channels 1..K: ways (soft or hard in [0,1])

    Produces logits: [Nq, K+1] with:
      - logit 0 = background logit (default 0 or learnable bias)
      - logits 1..K = sim(q, fg_k) - sim(q, bg_k)
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        normalize: bool = True,
        center: bool = True,
        learnable_temp: bool = True,
        init_temp: float = 20.0,
        eps: float = 1e-6,
        bg_logit: str = "zero",  # "zero" | "learnable"
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.normalize = bool(normalize)
        self.center = bool(center)
        self.eps = float(eps)

        if bg_logit not in {"zero", "learnable"}:
            raise ValueError(
                f"bg_logit must be 'zero' or 'learnable', got {bg_logit!r}"
            )
        self.bg_logit_mode = bg_logit
        if self.bg_logit_mode == "learnable":
            self.bg_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("bg_bias", torch.tensor(0.0))

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

    def _apply_mask(
        self, feats: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None
    ):
        if mask is None:
            return feats, labels
        m = mask.to(dtype=feats.dtype).unsqueeze(-1)  # [N,1]
        return feats * m, labels * m

    def _weighted_mean(
        self, feats: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        feats:   [N,H]
        weights: [N,K]
        returns: [K,H]
        """
        denom = weights.sum(dim=0).clamp_min(self.eps)  # [K]
        num = weights.transpose(0, 1) @ feats  # [K,H]
        return num / denom.unsqueeze(-1)

    def _compute_fg_bg_prototypes(
        self, feats: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feats:  [N,H]
        labels: [N,C] with C=K+1 (bg + K ways)

        Returns:
          proto_fg: [K,H] for ways 1..K
          proto_bg: [K,H] where bg_k corresponds to "not fg_k"
        """
        C = labels.shape[1]
        if C < 2:
            raise ValueError("Need at least 2 channels (bg + >=1 way).")

        y_fg = labels[:, 1:]  # [N,K]
        proto_fg = self._weighted_mean(feats, y_fg)  # [K,H]

        # Relative bg per way: bg_k = 1 - fg_k (on valid tokens)
        y_bgk = (1.0 - y_fg).clamp(0.0, 1.0)  # [N,K]
        proto_bg = self._weighted_mean(feats, y_bgk)  # [K,H]

        return proto_fg, proto_bg

    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [N,E]
        support_labels: torch.Tensor,  # [N,K+1]
        support_valid: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if support_feats.ndim != 2 or support_labels.ndim != 2:
            raise ValueError(
                "support_feats and support_labels must be 2D: [N,E] and [N,C]."
            )
        if support_feats.shape[0] != support_labels.shape[0]:
            raise ValueError("support_feats and support_labels must share N.")
        if support_feats.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected support_feats dim=E={self.embed_dim}, got {support_feats.shape[1]}."
            )

        # optional centering BEFORE normalization
        if self.center:
            if support_valid is not None:
                m = support_valid.to(dtype=support_feats.dtype).unsqueeze(-1)
                denom = m.sum().clamp_min(self.eps)
                mu = (support_feats * m).sum(dim=0) / denom
            else:
                mu = support_feats.mean(dim=0)
            support_feats = support_feats - mu
        else:
            mu = None

        if self.normalize:
            support_feats = F.normalize(support_feats, dim=-1)

        support_feats, labels = self._apply_mask(
            support_feats, support_labels, support_valid
        )

        proto_fg, proto_bg = self._compute_fg_bg_prototypes(
            support_feats, labels
        )  # [K,H] each

        if self.normalize:
            proto_fg = F.normalize(proto_fg, dim=-1)
            proto_bg = F.normalize(proto_bg, dim=-1)

        ctx: dict[str, torch.Tensor] = {
            "proto_fg": proto_fg,
            "proto_bg": proto_bg,
        }
        if mu is not None:
            ctx["center"] = mu
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,
        support_ctx: dict[str, Any],
    ) -> torch.Tensor:
        """
        query_feats: [Nq,E]
        returns logits: [Nq, K+1]
        """
        if query_feats.ndim != 2:
            raise ValueError(f"query_feats must be [Nq,E], got {query_feats.shape}")

        if self.center and "center" in support_ctx:
            query_feats = query_feats - support_ctx["center"]

        if self.normalize:
            query_feats = F.normalize(query_feats, dim=-1)

        proto_fg = support_ctx["proto_fg"]  # [K,H]
        proto_bg = support_ctx["proto_bg"]  # [K,H]
        if proto_fg.shape != proto_bg.shape:
            raise ValueError(
                f"proto_fg and proto_bg shapes must match, got {proto_fg.shape} vs {proto_bg.shape}"
            )

        # Similarities: [Nq,K]
        sim_fg = query_feats @ proto_fg.t()
        sim_bg = query_feats @ proto_bg.t()

        # Margin logits for foreground ways: [Nq,K]
        fg_logits = sim_fg - sim_bg

        # Background logit: [Nq,1] (0 or learnable bias)
        bg_logit = self.bg_bias.expand(query_feats.shape[0], 1)

        logits = torch.cat([bg_logit, fg_logits], dim=1)  # [Nq, K+1]

        temp = torch.exp(self.log_temp).clamp_min(1e-3)
        return logits * temp
