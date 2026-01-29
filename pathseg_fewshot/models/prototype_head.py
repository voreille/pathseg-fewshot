import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoTypeHead(nn.Module):
    """
    Prototype head on token features.

    fit_support:
      support_feats: [Ns, E]      (Ns = #support tokens/pixels used)
      support_labels: [Ns, C]     soft/hard assignment per token
      support_mask: [Ns] bool optional (valid tokens)

    predict_query:
      query_feats: [Nq, E]
      returns logits: [Nq, C]
    """

    def __init__(
        self,
        embed_dim: int,
        proj_dim: int,
        *,
        normalize: bool = True,
        center: bool = True,
        learnable_temp: bool = True,
        init_temp: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.proj_dim = int(proj_dim)
        self.normalize = bool(normalize)
        self.center = bool(center)
        self.eps = float(eps)

        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.proj_dim),
            nn.ReLU(inplace=True),
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
        """
        feats:  [N,H]
        labels: [N,C]
        returns prototypes [C,H]
        """
        # denom per class: [C]
        denom = labels.sum(dim=0).clamp_min(self.eps)  # [C]
        # sum feats per class: [C,H]
        num = labels.transpose(0, 1) @ feats  # (C,N)@(N,H) -> (C,H)
        return num / denom.unsqueeze(-1)

    def fit_support(
        self,
        support_feats: torch.Tensor,
        support_labels: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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

        feats = self.proj(support_feats)  # [N,H]

        if self.center:
            if support_mask is not None:
                m = support_mask.float().unsqueeze(-1)
                denom = m.sum().clamp_min(self.eps)
                mu = (feats * m).sum(dim=0) / denom
            else:
                mu = feats.mean(dim=0)
            feats = feats - mu
        else:
            mu = None

        if self.normalize:
            feats = F.normalize(feats, dim=-1)

        feats, labels = self._apply_mask(feats, support_labels, support_mask)
        prototypes = self._weighted_mean(feats, labels)  # [C,H]

        if self.normalize:
            prototypes = F.normalize(prototypes, dim=-1)

        ctx = {"prototypes": prototypes}
        if mu is not None:
            ctx["center"] = mu
        return ctx

    def predict_query(
        self, query_feats: torch.Tensor, support_ctx: dict[str, torch.Tensor]
    ) -> torch.Tensor:

        q = self.proj(query_feats)

        if self.center and "center" in support_ctx:
            q = q - support_ctx["center"]

        if self.normalize:
            q = F.normalize(q, dim=-1)

        logits = q @ support_ctx["prototypes"].t()
        return logits * torch.exp(self.log_temp).clamp_min(1e-3)
