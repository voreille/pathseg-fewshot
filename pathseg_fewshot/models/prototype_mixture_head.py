import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathseg_fewshot.models.metalearner import MetaLearnerBase


class PrototypeMixtureBankHead(MetaLearnerBase):
    """
    Option-1 rewrite: compositional prototypes in *embedding space* using a learned bank.

    Bank P: [Kbank, E] is a set of learnable concept vectors / centroids / basis elements.

    For an episode with C channels (bg + fg ways):
      1) Compute token->bank responsibilities r_i over Kbank:
           r_i = softmax(sim(z_i, P)/tau_assign)
      2) For each class c, compute a *presence-based* concept signature alpha_c over Kbank
         using smooth-max (log-sum-exp) pooling over support tokens:
           alpha_c[k] = LSE_i( beta * (y_i^c * r_i[k]) ) / beta
         (presence / signature; robust to concept proportion changes)
      3) Compose class prototype in embedding space:
           mu_c = alpha_c @ P
      4) Predict query tokens by dot-product/cosine to mu_c:
           logit_c(z_q) = <z_q, mu_c> * score_temp

    Background handling:
      - default "open_set": bg logit = -max_fg(logits_fg)
        (bg is the complement of foreground matches; avoids bg distribution mismatch)
      - optional "global": also build a bg prototype from alpha_0, but usually brittle.

    Returns:
      predict_query -> [Q*T, C] logits (C = K_episode+1).
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        bank_size: int = 256,
        normalize_feats: bool = True,  # cosine geometry
        center: bool = True,
        # responsibilities temperature
        learnable_assign_temp: bool = True,
        init_assign_temp: float = 10.0,
        min_assign_temp: float = 1e-3,
        # presence pooling sharpness
        presence_beta: float = 20.0,
        # scoring temperature
        learnable_score_temp: bool = True,
        init_score_temp: float = 20.0,
        # background mode
        bg_mode: str = "open_set",  # "open_set" | "global"
        eps: float = 1e-6,
        # optional sparsity: keep only top-k concepts in alpha (encourages compositionality)
        topk_alpha: Optional[int] = None,
        # optional aux losses toggles
        add_diversity_loss: bool = False,
        add_usage_loss: bool = False,
        usage_on_fg_only: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bank_size = int(bank_size)
        self.normalize_feats = bool(normalize_feats)
        self.center = bool(center)
        self.presence_beta = float(presence_beta)
        self.bg_mode = str(bg_mode)
        self.eps = float(eps)
        self.topk_alpha = None if topk_alpha is None else int(topk_alpha)

        self.min_assign_temp = float(min_assign_temp)

        self.add_diversity_loss = bool(add_diversity_loss)
        self.add_usage_loss = bool(add_usage_loss)
        self.usage_on_fg_only = bool(usage_on_fg_only)

        if self.bg_mode not in {"open_set", "global"}:
            raise ValueError(
                f"bg_mode must be 'open_set' or 'global', got {self.bg_mode!r}"
            )

        # Learnable bank P: [Kbank, E]
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        # token->bank assignment temperature (higher => softer, lower => harder)
        if learnable_assign_temp:
            self.log_assign_temp = nn.Parameter(
                torch.log(torch.tensor(init_assign_temp))
            )
        else:
            self.register_buffer(
                "log_assign_temp", torch.log(torch.tensor(init_assign_temp))
            )

        # query scoring temperature
        if learnable_score_temp:
            self.log_score_temp = nn.Parameter(torch.log(torch.tensor(init_score_temp)))
        else:
            self.register_buffer(
                "log_score_temp", torch.log(torch.tensor(init_score_temp))
            )

    # -------------------------
    # helpers
    # -------------------------
    def _apply_mask(
        self,
        feats: torch.Tensor,  # [N,E]
        labels: torch.Tensor,  # [N,C]
        valid: Optional[torch.Tensor],  # [N] bool
    ):
        if valid is None:
            return feats, labels
        m = valid.to(dtype=feats.dtype).unsqueeze(-1)  # [N,1]
        return feats * m, labels * m

    def _topk_mask(self, v: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or k >= v.numel():
            return v
        vals, idx = torch.topk(v, k=k, dim=-1)
        out = torch.zeros_like(v)
        out.scatter_(0, idx, vals)
        return out

    def _responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [N,E] token embeddings
        returns r: [N,Kbank] responsibilities over bank
        """
        P = self.bank
        if self.normalize_feats:
            z = F.normalize(z, dim=-1)
            P = F.normalize(P, dim=-1)

        tau = torch.exp(self.log_assign_temp).clamp_min(self.min_assign_temp)
        logits = (z @ P.t()) / tau
        r = F.softmax(logits, dim=-1)
        return r

    def _alpha_presence_lse(
        self,
        r: torch.Tensor,  # [N,Kbank]
        y: torch.Tensor,  # [N,1] weights in [0,1] for class c
        beta: float,
    ) -> torch.Tensor:
        """
        Presence / signature pooling over tokens for a class.

          alpha[k] = (1/beta) * logsumexp_i( beta * (y_i * r_i[k]) )

        Returns:
          alpha: [Kbank] nonnegative
        """
        x = r * y  # [N,Kbank]
        alpha = torch.logsumexp(beta * x, dim=0) / beta
        alpha = alpha.clamp_min(0.0)
        if self.topk_alpha is not None:
            alpha = self._topk_mask(alpha, self.topk_alpha)
        # normalize to make dot-products comparable across classes
        alpha = alpha / alpha.sum().clamp_min(self.eps)
        return alpha

    def _compose_prototypes(
        self,
        alphas: torch.Tensor,  # [C,Kbank]
    ) -> torch.Tensor:
        """
        Compose embedding-space prototypes: mu = alpha @ P
        Returns:
          mu: [C,E]
        """
        P = self.bank
        if self.normalize_feats:
            P = F.normalize(P, dim=-1)
        mu = alphas @ P  # [C,E]
        if self.normalize_feats:
            mu = F.normalize(mu, dim=-1)
        return mu

    # -------------------------
    # MetaLearnerBase interface
    # -------------------------
    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[0] != support_labels.shape[0]:
            raise ValueError("support_feats and support_labels must share S.")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected support_feats E={self.embed_dim}, got {support_feats.shape[2]}."
            )

        # Flatten
        z = support_feats.reshape(-1, support_feats.shape[-1])  # [N,E]
        y = support_labels.reshape(-1, support_labels.shape[-1])  # [N,C]
        valid = None if support_valid is None else support_valid.reshape(-1)  # [N]

        # Center (on valid tokens only)
        if self.center:
            if valid is not None:
                m = valid.float().unsqueeze(-1)
                denom = m.sum().clamp_min(self.eps)
                mu_center = (z * m).sum(dim=0) / denom
            else:
                mu_center = z.mean(dim=0)
            z = z - mu_center
        else:
            mu_center = None

        # Normalize token feats if cosine geometry
        if self.normalize_feats:
            z = F.normalize(z, dim=-1)

        # mask invalid tokens out of z and y
        z, y = self._apply_mask(z, y, valid)

        # token responsibilities
        r = self._responsibilities(z)  # [N,Kbank]

        # presence signatures per class
        C = y.shape[1]
        beta = self.presence_beta
        alphas = []
        for c in range(C):
            y_c = y[:, c : c + 1].clamp(0.0, 1.0)
            alpha_c = self._alpha_presence_lse(r, y_c, beta=beta)  # [Kbank]
            alphas.append(alpha_c)
        alphas = torch.stack(alphas, dim=0)  # [C,Kbank]

        # compose prototypes in embedding space
        prototypes = self._compose_prototypes(alphas)  # [C,E]

        ctx: Dict[str, Any] = {"prototypes": prototypes}

        if mu_center is not None:
            ctx["center"] = mu_center

        # Optional aux losses
        if self.add_diversity_loss:
            ctx["aux_bank_div"] = self.bank_diversity_loss()
        if self.add_usage_loss:
            if self.usage_on_fg_only:
                ctx["aux_bank_usage"] = self._usage_loss_from_r_fg(r, y, valid=valid)
            else:
                ctx["aux_bank_usage"] = self._usage_loss_from_r(r, valid=valid)

        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        z = query_feats.reshape(-1, query_feats.shape[-1])  # [N,E]

        if self.center and "center" in support_ctx:
            z = z - support_ctx["center"]

        if self.normalize_feats:
            z = F.normalize(z, dim=-1)

        prototypes = support_ctx["prototypes"]  # [C,E]
        logits = z @ prototypes.t()  # [N,C]

        # Score temperature
        temp = torch.exp(self.log_score_temp).clamp_min(1e-3)
        logits = logits * temp

        # Background handling: open-set bg to avoid mismatched bg distribution
        if self.bg_mode == "open_set":
            if logits.shape[1] < 2:
                return logits
            logits_fg = logits[:, 1:]  # [N,K]
            logit_bg = -logits_fg.max(dim=1, keepdim=True).values
            logits = torch.cat([logit_bg, logits_fg], dim=1)

        return logits

    # -------------------------
    # Optional aux losses (same as you had)
    # -------------------------
    def bank_diversity_loss(self) -> torch.Tensor:
        B = F.normalize(self.bank, dim=-1)
        G = B @ B.t()
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).mean()

    def _usage_loss_from_r(
        self,
        r: torch.Tensor,  # [N,Kbank]
        valid: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:
        eps = self.eps
        K = r.shape[-1]

        if valid is not None:
            m = valid.to(dtype=r.dtype).unsqueeze(-1)
            denom = m.sum().clamp_min(eps)
            p = (r * m).sum(dim=0) / denom
        else:
            p = r.mean(dim=0)

        p = p / p.sum().clamp_min(eps)
        log_u = -math.log(K)
        return (p * (p.clamp_min(eps).log() - log_u)).sum()

    def _usage_loss_from_r_fg(
        self,
        r: torch.Tensor,  # [N,Kbank]
        labels: torch.Tensor,  # [N,C] (0=bg, 1..=fg)
        valid: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:
        eps = self.eps
        K = r.shape[-1]

        fg_w = labels[:, 1:].sum(dim=1, keepdim=True)
        if valid is not None:
            fg_w = fg_w * valid.to(dtype=fg_w.dtype).unsqueeze(-1)

        denom = fg_w.sum().clamp_min(eps)
        p = (r * fg_w).sum(dim=0) / denom
        p = p / p.sum().clamp_min(eps)

        return (p * (p.clamp_min(eps).log() + math.log(K))).sum()


class GaussianMixtureBankHead(MetaLearnerBase):
    """
    Hierarchical mixture model head with a global bank of concept centroids.

    - Global bank: P = [Kbank, E] (learned)
    - Token likelihood (vMF-like): log p(z|k) = kappa * <z_hat, p_hat_k>
    - Token->concept posterior: r_i = softmax(log p(z_i|k))  (latent inference)
    - Episode class mixture weights: pi_c from support via Dirichlet-smoothed soft counts
    - Query class logits: logsumexp_k( log pi_c,k + log p(z|k) )

    Shapes:
      support_feats:  [S,T,E]
      support_labels: [S,T,C] (bg+fg)
      support_valid:  [S,T] optional
      predict_query -> [Q*T, C]
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        bank_size: int = 256,
        center: bool = True,
        normalize_feats: bool = True,  # strongly recommended for cosine likelihood
        # kappa controls likelihood sharpness
        learnable_kappa: bool = True,
        init_kappa: float = 20.0,
        min_kappa: float = 1e-3,
        # Dirichlet prior strength for pi smoothing (fixes underrepresented concepts)
        dirichlet_alpha: float = 1e-2,
        # background handling
        bg_mode: str = "open_set",  # "open_set" | "learned_global"
        learn_bg_global: bool = False,  # if bg_mode=="learned_global"
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bank_size = int(bank_size)
        self.center = bool(center)
        self.normalize_feats = bool(normalize_feats)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.bg_mode = str(bg_mode)
        self.learn_bg_global = bool(learn_bg_global)
        self.eps = float(eps)
        self.min_kappa = float(min_kappa)

        if self.bg_mode not in {"open_set", "learned_global"}:
            raise ValueError(
                f"bg_mode must be 'open_set' or 'learned_global', got {self.bg_mode!r}"
            )

        # Global concept centroids
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        # Optional global background mixture (captures non-tissue etc.)
        if self.bg_mode == "learned_global" and self.learn_bg_global:
            self.bg_logits = nn.Parameter(torch.zeros(self.bank_size))

        # kappa = exp(log_kappa)
        if learnable_kappa:
            self.log_kappa = nn.Parameter(torch.log(torch.tensor(init_kappa)))
        else:
            self.register_buffer("log_kappa", torch.log(torch.tensor(init_kappa)))

    def _apply_mask(
        self,
        feats: torch.Tensor,  # [N,E]
        labels: torch.Tensor,  # [N,C]
        valid: Optional[torch.Tensor],  # [N] bool
    ):
        if valid is None:
            return feats, labels
        m = valid.to(dtype=feats.dtype).unsqueeze(-1)
        return feats * m, labels * m

    def _log_p_z_given_k(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [N,E]
        returns loglik: [N,Kbank]
        """
        P = self.bank
        if self.normalize_feats:
            z = F.normalize(z, dim=-1)
            P = F.normalize(P, dim=-1)

        kappa = torch.exp(self.log_kappa).clamp_min(self.min_kappa)
        return kappa * (z @ P.t())

    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected E={self.embed_dim}, got {support_feats.shape[2]}."
            )

        z = support_feats.reshape(-1, support_feats.shape[-1])  # [N,E]
        y = support_labels.reshape(-1, support_labels.shape[-1])  # [N,C]
        valid = None if support_valid is None else support_valid.reshape(-1)  # [N]

        # Centering (episode-specific), before normalization
        if self.center:
            if valid is not None:
                m = valid.float().unsqueeze(-1)
                denom = m.sum().clamp_min(self.eps)
                mu = (z * m).sum(dim=0) / denom
            else:
                mu = z.mean(dim=0)
            z = z - mu
        else:
            mu = None

        # Mask invalid tokens
        z, y = self._apply_mask(z, y, valid)

        # Token->concept posterior r_i = p(k|z_i) proportional to p(z_i|k)
        loglik = self._log_p_z_given_k(z)  # [N,K]
        r = F.softmax(loglik, dim=-1)  # [N,K]

        # Dirichlet-smoothed mixture weights per class
        # counts: [C,K] = y^T @ r
        counts = y.transpose(0, 1) @ r  # [C,K]
        alpha = self.dirichlet_alpha
        pi = counts + alpha  # [C,K]
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        ctx: Dict[str, Any] = {"pi": pi}  # [C,K]
        if mu is not None:
            ctx["center"] = mu
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        z = query_feats.reshape(-1, query_feats.shape[-1])  # [N,E]

        if self.center and "center" in support_ctx:
            z = z - support_ctx["center"]

        loglik = self._log_p_z_given_k(z)  # [N,K]

        pi = support_ctx["pi"]  # [C,K]
        log_pi = (pi.clamp_min(self.eps)).log()  # [C,K]

        # logits: [N,C] via logsumexp_k(log_pi[c,k] + loglik[n,k])
        logits = torch.logsumexp(loglik.unsqueeze(1) + log_pi.unsqueeze(0), dim=-1)

        # Background handling
        if self.bg_mode == "open_set":
            if logits.shape[1] > 1:
                logits_fg = logits[:, 1:]
                logit_bg = -logits_fg.max(dim=1, keepdim=True).values
                logits = torch.cat([logit_bg, logits_fg], dim=1)
        elif self.bg_mode == "learned_global":
            if self.learn_bg_global:
                bg_pi = F.softmax(self.bg_logits, dim=-1).clamp_min(self.eps)  # [K]
                log_bg_pi = bg_pi.log().unsqueeze(0)  # [1,K]
                logit_bg = torch.logsumexp(loglik + log_bg_pi, dim=-1, keepdim=True)
                if logits.shape[1] > 1:
                    logits_fg = logits[:, 1:]
                    logits = torch.cat([logit_bg, logits_fg], dim=1)
                else:
                    logits = logit_bg

        return logits


class PrototypeHeadMixtureQuery(MetaLearnerBase):
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
        bank_size: int = 256,
        normalize: bool = True,
        center: bool = True,
        bg_mode: str = "global",  # "global" | "panet_relative"
        learnable_temp: bool = True,
        init_temp: float = 20.0,
        eps: float = 1e-6,
        add_diversity_loss: bool = False,
        add_usage_loss: bool = False,
        usage_on_fg_only: bool = True,
    ):
        super().__init__()
        self.bank_size = int(bank_size)
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

        self.add_diversity_loss = bool(add_diversity_loss)
        self.add_usage_loss = bool(add_usage_loss)
        self.usage_on_fg_only = bool(usage_on_fg_only)

        if self.bg_mode not in {"open_set", "global"}:
            raise ValueError(
                f"bg_mode must be 'open_set' or 'global', got {self.bg_mode!r}"
            )

        # Learnable bank P: [Kbank, E]
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

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

        support_feats, labels = self._apply_mask(
            support_feats, support_labels, support_valid
        )

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


class AreaGatedBankMaskHead(MetaLearnerBase):
    """
    Episode head: use a learned bank of prototypes to generate a bank of "concept masks"
    (one per bank entry) on support/query, then compute per-class concept weights from
    *area/activation inside vs outside* the support mask, and finally combine the query
    concept masks with those weights to produce logits.

    Core idea (per episode class c):
      - concept map on tokens: s_k(t) = <x_t, p_k>  (cosine if normalize=True)
      - concept mask (soft):   m_k(t) = softmax_k(s_k(t))   or sigmoid(s_k(t))
      - area inside:  a_in[k]  = E_t[ m_k(t) | y_c(t)=1 ]
      - area outside: a_out[k] = E_t[ m_k(t) | y_c(t)=0 ]
      - discriminative weight: w_c[k] = ReLU(a_in[k] - a_out[k])
      - query class logit map: logit_c(t) = sum_k w_c[k] * m_k^q(t)

    Background:
      - "open_set": bg logit = -max_fg(logits_fg)  (recommended)
      - "explicit": bg computed the same way from channel 0 (often brittle)

    Shapes:
      support_feats:  [S,T,E]
      support_labels: [S,T,C]  (soft/hard one-hot, channel0=bg)
      support_valid:  [S,T] optional bool
      query_feats:    [Q,T,E]

    Output:
      predict_query: [Q*T, C]
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        bank_size: int = 256,
        normalize: bool = True,
        center: bool = True,
        # concept mask type:
        mask_mode: str = "softmax",  # "softmax" (over Kbank) | "sigmoid" (independent)
        # optional sharpening:
        learnable_temp: bool = True,
        init_temp: float = 10.0,
        min_temp: float = 1e-3,
        # weight normalization / gating:
        weight_mode: str = "relu",  # "relu" | "softmax" | "sigmoid"
        weight_temp: float = 1.0,  # used if weight_mode="softmax"
        # background:
        bg_mode: str = "open_set",  # "open_set" | "explicit"
        eps: float = 1e-6,
        add_diversity_loss: bool = False,
        add_usage_loss: bool = False,
        usage_on_fg_only: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bank_size = int(bank_size)
        self.normalize = bool(normalize)
        self.center = bool(center)
        self.mask_mode = str(mask_mode)
        self.weight_mode = str(weight_mode)
        self.weight_temp = float(weight_temp)
        self.bg_mode = str(bg_mode)
        self.eps = float(eps)
        self.min_temp = float(min_temp)

        self.add_diversity_loss = bool(add_diversity_loss)
        self.add_usage_loss = bool(add_usage_loss)
        self.usage_on_fg_only = bool(usage_on_fg_only)

        if self.mask_mode not in {"softmax", "sigmoid"}:
            raise ValueError(
                f"mask_mode must be 'softmax' or 'sigmoid', got {self.mask_mode!r}"
            )
        if self.weight_mode not in {"relu", "softmax", "sigmoid"}:
            raise ValueError(
                f"weight_mode must be 'relu'|'softmax'|'sigmoid', got {self.weight_mode!r}"
            )
        if self.bg_mode not in {"open_set", "explicit"}:
            raise ValueError(
                f"bg_mode must be 'open_set' or 'explicit', got {self.bg_mode!r}"
            )

        # Bank prototypes: [Kbank,E]
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

    # -------------------------
    # internals
    # -------------------------
    def _concept_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,E]
        returns scores s: [N,Kbank]
        """
        P = self.bank
        if self.normalize:
            x = F.normalize(x, dim=-1)
            P = F.normalize(P, dim=-1)

        temp = torch.exp(self.log_temp).clamp_min(self.min_temp)
        return (x @ P.t()) * temp

    def concept_masks(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,E]
        returns m: [N,Kbank] soft masks / responsibilities per token
        """
        s = self._concept_logits(x)  # [N,K]
        if self.mask_mode == "softmax":
            return F.softmax(s, dim=-1)
        else:  # sigmoid
            return torch.sigmoid(s)

    def _masked_mean(self, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        v:    [N,K]
        mask: [N,1] in [0,1]
        returns: [K] mean of v under mask
        """
        denom = mask.sum(dim=0).clamp_min(self.eps)  # [1]
        return (v * mask).sum(dim=0) / denom  # [K]

    def _make_weights(self, a_in: torch.Tensor, a_out: torch.Tensor) -> torch.Tensor:
        """
        a_in, a_out: [K]
        returns w: [K]
        """
        d = a_in - a_out  # [K]
        if self.weight_mode == "relu":
            w = F.relu(d)
            # normalize for stability
            w = w / w.sum().clamp_min(self.eps)
            return w
        if self.weight_mode == "sigmoid":
            return torch.sigmoid(d)
        # softmax
        return F.softmax(d / max(self.weight_temp, 1e-6), dim=-1)

    # -------------------------
    # MetaLearnerBase
    # -------------------------
    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[:2] != support_labels.shape[:2]:
            raise ValueError("support_feats and support_labels must share [S,T].")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {support_feats.shape[2]}."
            )

        S, T, E = support_feats.shape
        C = support_labels.shape[-1]

        x = support_feats.reshape(-1, E)  # [N,E]
        y = support_labels.reshape(-1, C)  # [N,C]
        valid = None if support_valid is None else support_valid.reshape(-1, 1)  # [N,1]

        # Centering (episode-specific), on valid tokens if provided
        if self.center:
            if valid is not None:
                denom = valid.sum().clamp_min(self.eps)
                mu = (x * valid).sum(dim=0) / denom
            else:
                mu = x.mean(dim=0)
            x = x - mu
        else:
            mu = None

        # Concept masks on support tokens: [N,K]
        m = self.concept_masks(x)  # [N,K]

        # If valid mask: zero-out invalid tokens in m and y (so they don't contribute)
        if valid is not None:
            m = m * valid
            y = y * valid  # broadcast to [N,C]

        # Compute per-class weights W: [C,K]
        W = []
        ones = torch.ones((m.shape[0], 1), device=m.device, dtype=m.dtype)

        for c in range(C):
            y_c = y[:, c : c + 1].clamp(0.0, 1.0)  # [N,1] soft membership
            in_mask = y_c
            out_mask = (ones - y_c).clamp(0.0, 1.0)

            # E[m_k | in] and E[m_k | out]
            a_in = self._masked_mean(m, in_mask)  # [K]
            a_out = self._masked_mean(m, out_mask)  # [K]

            w_c = self._make_weights(a_in, a_out)  # [K]
            W.append(w_c)

        W = torch.stack(W, dim=0)  # [C,K]

        ctx: Dict[str, Any] = {"W": W}
        if mu is not None:
            ctx["center"] = mu

        if self.add_diversity_loss:
            ctx["aux_bank_div"] = self.bank_diversity_loss()
        # if self.add_usage_loss:
        #     if self.usage_on_fg_only:
        #         ctx["aux_bank_usage"] = self._usage_loss_from_r_fg(r, y, valid=valid)
        #     else:
        #         ctx["aux_bank_usage"] = self._usage_loss_from_r(r, valid=valid)

        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        if query_feats.ndim != 3:
            raise ValueError("query_feats must be [Q,T,E].")
        Q, T, E = query_feats.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {E}.")

        xq = query_feats.reshape(-1, E)  # [N,E]

        if self.center and "center" in support_ctx:
            xq = xq - support_ctx["center"]

        # Query concept masks: [N,K]
        m_q = self.concept_masks(xq)

        # Combine with per-class weights
        W = support_ctx["W"]  # [C,K]
        # logits: [N,C] = m_q @ W^T
        logits = m_q @ W.t()

        # Background handling
        if self.bg_mode == "open_set":
            if logits.shape[1] > 1:
                logits_fg = logits[:, 1:]  # [N,K_episode]
                logit_bg = -logits_fg.max(dim=1, keepdim=True).values
                logits = torch.cat([logit_bg, logits_fg], dim=1)
        # else "explicit": channel 0 uses its own computed weights already

        return logits

    def bank_diversity_loss(self) -> torch.Tensor:
        B = F.normalize(self.bank, dim=-1)
        G = B @ B.t()
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).mean()

    def _usage_loss_from_r(
        self,
        r: torch.Tensor,  # [N,Kbank]
        valid: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:
        eps = self.eps
        K = r.shape[-1]

        if valid is not None:
            m = valid.to(dtype=r.dtype).unsqueeze(-1)
            denom = m.sum().clamp_min(eps)
            p = (r * m).sum(dim=0) / denom
        else:
            p = r.mean(dim=0)

        p = p / p.sum().clamp_min(eps)
        log_u = -math.log(K)
        return (p * (p.clamp_min(eps).log() - log_u)).sum()

    def _usage_loss_from_r_fg(
        self,
        r: torch.Tensor,  # [N,Kbank]
        labels: torch.Tensor,  # [N,C] (0=bg, 1..=fg)
        valid: Optional[torch.Tensor] = None,  # [N]
    ) -> torch.Tensor:
        eps = self.eps
        K = r.shape[-1]

        fg_w = labels[:, 1:].sum(dim=1, keepdim=True)
        if valid is not None:
            fg_w = fg_w * valid.to(dtype=fg_w.dtype).unsqueeze(-1)

        denom = fg_w.sum().clamp_min(eps)
        p = (r * fg_w).sum(dim=0) / denom
        p = p / p.sum().clamp_min(eps)

        return (p * (p.clamp_min(eps).log() + math.log(K))).sum()


class AreaGatedBankLogitHead(MetaLearnerBase):
    """
    Episode head: use a learned bank of prototypes to compute concept *logits* s_k(t)=<x_t,p_k>.
    On support, estimate per-class concept weights using inside-vs-outside activation statistics,
    then apply those weights on query concept logits to produce per-token class logits.

    Key changes vs your original:
      - NO softmax over Kbank (too diluted)
      - Use raw logits s (or optionally sigmoid(s) as "activation")
      - Presence-based option: use top-q mean (or max) inside mask instead of mean
      - Weighting uses in-vs-out on logits (or activations), not softmax responsibilities
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        bank_size: int = 256,
        normalize: bool = True,
        center: bool = True,
        # temperature:
        learnable_temp: bool = True,
        init_temp: float = 10.0,
        min_temp: float = 1e-3,
        # activation used to compute statistics:
        stat_mode: str = "logits",  # "logits" | "sigmoid"
        # presence aggregator inside FG:
        in_agg: str = "topq_mean",  # "mean" | "max" | "topq_mean"
        topq: float = 0.1,  # used if in_agg="topq_mean" (e.g., 0.05..0.2)
        # outside aggregator:
        out_agg: str = "mean",  # "mean" | "max" | "topq_mean"
        # weight transform:
        weight_mode: str = "relu",  # "relu" | "softmax" | "sigmoid"
        weight_temp: float = 1.0,  # for weight_mode="softmax"
        # optional sparsification of concepts at token level (query & support):
        token_topk: Optional[int] = None,  # e.g., 16; applied on |s| or s
        token_topk_on: str = "abs",  # "abs" | "pos"
        # background:
        bg_mode: str = "open_set",  # "open_set" | "explicit"
        eps: float = 1e-6,
        add_diversity_loss: bool = False,
        add_sparse_loss: bool = False,
        add_usage_loss: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bank_size = int(bank_size)
        self.normalize = bool(normalize)
        self.center = bool(center)

        self.add_diversity_loss = bool(add_diversity_loss)
        self.add_sparse_loss = bool(add_sparse_loss)
        self.add_usage_loss = bool(add_usage_loss)

        self.stat_mode = str(stat_mode)
        if self.stat_mode not in {"logits", "sigmoid"}:
            raise ValueError(
                f"stat_mode must be 'logits'|'sigmoid', got {self.stat_mode!r}"
            )

        self.in_agg = str(in_agg)
        self.out_agg = str(out_agg)
        for m in (self.in_agg, self.out_agg):
            if m not in {"mean", "max", "topq_mean"}:
                raise ValueError(f"agg must be 'mean'|'max'|'topq_mean', got {m!r}")

        self.topq = float(topq)
        if not (0.0 < self.topq <= 1.0):
            raise ValueError(f"topq must be in (0,1], got {self.topq}")

        self.weight_mode = str(weight_mode)
        if self.weight_mode not in {"relu", "softmax", "sigmoid"}:
            raise ValueError(
                f"weight_mode must be 'relu'|'softmax'|'sigmoid', got {self.weight_mode!r}"
            )
        self.weight_temp = float(weight_temp)

        self.token_topk = token_topk
        self.token_topk_on = str(token_topk_on)
        if self.token_topk_on not in {"abs", "pos"}:
            raise ValueError(
                f"token_topk_on must be 'abs'|'pos', got {self.token_topk_on!r}"
            )

        self.bg_mode = str(bg_mode)
        if self.bg_mode not in {"open_set", "explicit"}:
            raise ValueError(
                f"bg_mode must be 'open_set'|'explicit', got {self.bg_mode!r}"
            )

        self.eps = float(eps)
        self.min_temp = float(min_temp)

        # Bank prototypes: [K,E]
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

    # -------------------------
    # concept logits
    # -------------------------
    def _concept_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,E]
        returns s: [N,K]
        """
        P = self.bank
        if self.normalize:
            x = F.normalize(x, dim=-1)
            P = F.normalize(P, dim=-1)

        temp = torch.exp(self.log_temp).clamp_min(self.min_temp)
        s = (x @ P.t()) * temp  # [N,K]
        return s

    def _maybe_token_topk(self, s: torch.Tensor) -> torch.Tensor:
        """
        Optionally keep only top-k concepts per token (set others to 0).
        Applied to logits before computing stats and query scoring.
        """
        if self.token_topk is None:
            return s
        k = int(self.token_topk)
        if k <= 0 or k > s.shape[1]:
            raise ValueError(f"token_topk must be in [1,K], got {k}")

        score = s.abs() if self.token_topk_on == "abs" else s
        vals, idx = torch.topk(score, k=k, dim=-1)
        mask = torch.zeros_like(s)
        mask.scatter_(dim=-1, index=idx, src=torch.ones_like(vals))
        return s * mask

    def _stat(self, s: torch.Tensor) -> torch.Tensor:
        """Map logits to statistic space."""
        if self.stat_mode == "sigmoid":
            return torch.sigmoid(s)
        return s

    # -------------------------
    # masked aggregations
    # -------------------------
    def _masked_agg(
        self, v: torch.Tensor, mask: torch.Tensor, agg: str
    ) -> torch.Tensor:
        """
        v:    [N,K]
        mask: [N,1] in [0,1]
        returns: [K]
        """
        # weights per row
        w = mask.clamp(0.0, 1.0)
        denom = w.sum().clamp_min(self.eps)

        if agg == "mean":
            return (v * w).sum(dim=0) / denom

        if agg == "max":
            # masked max: set outside to very negative (for logits) or 0 (for sigmoid)
            if self.stat_mode == "logits":
                vv = v.masked_fill(w <= 0, -1e9)
                return vv.max(dim=0).values
            else:
                vv = v.masked_fill(w <= 0, 0.0)
                return vv.max(dim=0).values

        # topq_mean
        # We take only rows where mask>0, then compute top-q mean per concept.
        idx = w.squeeze(-1) > 0
        if idx.sum() == 0:
            # no pixels: return zeros
            return torch.zeros(v.shape[1], device=v.device, dtype=v.dtype)

        vv = v[idx]  # [M,K]
        M = vv.shape[0]
        q = max(1, int(math.ceil(self.topq * M)))
        topv, _ = torch.topk(vv, k=q, dim=0)
        return topv.mean(dim=0)

    def _make_weights(self, a_in: torch.Tensor, a_out: torch.Tensor) -> torch.Tensor:
        """
        a_in, a_out: [K]
        returns w: [K]
        """
        d = a_in - a_out  # [K]

        if self.weight_mode == "relu":
            w = F.relu(d)
            w = w / w.sum().clamp_min(self.eps)
            return w

        if self.weight_mode == "sigmoid":
            return torch.sigmoid(d)

        # softmax
        return F.softmax(d / max(self.weight_temp, 1e-6), dim=-1)

    # -------------------------
    # episodic API
    # -------------------------
    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C] (channel0 can be bg)
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[:2] != support_labels.shape[:2]:
            raise ValueError("support_feats and support_labels must share [S,T].")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {support_feats.shape[2]}."
            )

        S, T, E = support_feats.shape
        C = support_labels.shape[-1]

        x = support_feats.reshape(-1, E)  # [N,E]
        y = support_labels.reshape(-1, C)  # [N,C]
        valid = (
            None
            if support_valid is None
            else support_valid.reshape(-1, 1).to(dtype=x.dtype)
        )  # [N,1]

        # Episode centering
        mu = None
        if self.center:
            if valid is not None:
                denom = valid.sum().clamp_min(self.eps)
                mu = (x * valid).sum(dim=0) / denom
            else:
                mu = x.mean(dim=0)
            x = x - mu

        # Concept logits on support: [N,K]
        s = self._concept_logits(x)
        s = self._maybe_token_topk(s)
        v = self._stat(s)  # either logits or sigmoid(logits)

        # Apply valid: make invalid tokens contribute nothing to stats
        if valid is not None:
            v = v * valid
            y = y * valid

        ones = torch.ones((v.shape[0], 1), device=v.device, dtype=v.dtype)

        # Per-class weights W: [C,K]
        W = []
        for c in range(C):
            y_c = y[:, c : c + 1].clamp(0.0, 1.0)  # [N,1]

            in_mask = y_c
            out_mask = (ones - y_c).clamp(0.0, 1.0)

            a_in = self._masked_agg(v, in_mask, agg=self.in_agg)  # [K]
            a_out = self._masked_agg(v, out_mask, agg=self.out_agg)  # [K]

            w_c = self._make_weights(a_in, a_out)  # [K]
            W.append(w_c)

        W = torch.stack(W, dim=0)  # [C,K]

        ctx: Dict[str, Any] = {"W": W}
        if mu is not None:
            ctx["center"] = mu
        if self.add_diversity_loss and self.training:
            ctx["aux_bank_div"] = self.bank_diversity_loss()
        if self.add_sparse_loss and self.training:
            ctx["aux_sparse"] = self._sparse_activation_loss(s, valid=valid)
        if self.add_usage_loss and self.training:
            ctx["aux_bank_usage"] = self._usage_loss(s, valid=valid)
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        if query_feats.ndim != 3:
            raise ValueError("query_feats must be [Q,T,E].")
        Q, T, E = query_feats.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {E}.")

        xq = query_feats.reshape(-1, E)

        if self.center and "center" in support_ctx:
            xq = xq - support_ctx["center"]

        s_q = self._concept_logits(xq)
        s_q = self._maybe_token_topk(s_q)
        v_q = self._stat(s_q)  # [N,K]

        W = support_ctx["W"]  # [C,K]

        # logits: [N,C] = v_q @ W^T
        logits = v_q @ W.t()

        if self.bg_mode == "open_set" and logits.shape[1] > 1:
            logits_fg = logits[:, 1:]  # [N,C-1]
            logit_bg = -logits_fg.max(dim=1, keepdim=True).values
            logits = torch.cat([logit_bg, logits_fg], dim=1)

        return logits

    def bank_diversity_loss(self) -> torch.Tensor:
        B = F.normalize(self.bank, dim=-1)
        G = B @ B.t()
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).mean()

    def _sparse_activation_loss(
        self,
        s: torch.Tensor,  # [N,K] raw logits
        valid: Optional[torch.Tensor] = None,  # [N,1]
    ) -> torch.Tensor:
        """
        Encourage per-token sparsity over concepts.
        Uses sigmoid activations and L1 penalty.
        """
        # Convert logits to [0,1] activations
        a = torch.sigmoid(s)  # [N,K]

        if valid is not None:
            a = a * valid
            denom = valid.sum().clamp_min(self.eps)
        else:
            denom = torch.tensor(a.shape[0], device=a.device, dtype=a.dtype)

        # Mean L1 activation per token
        # (equivalent to average activation mass)
        return a.sum() / (denom * a.shape[1])

    def _usage_loss(
        self,
        s: torch.Tensor,  # [N,K] raw logits
        valid: Optional[torch.Tensor] = None,  # [N,1]
    ) -> torch.Tensor:
        """
        Encourage balanced usage of concepts.
        Computes entropy of mean activation distribution.
        """
        eps = self.eps

        a = torch.sigmoid(s)  # [N,K]

        if valid is not None:
            a = a * valid
            denom = valid.sum().clamp_min(eps)
        else:
            denom = torch.tensor(a.shape[0], device=a.device, dtype=a.dtype)

        # Average activation per concept
        p = a.sum(dim=0) / denom  # [K]

        # Normalize to probability distribution
        p = p / p.sum().clamp_min(eps)

        # Negative entropy (minimize this → maximize entropy)
        return (p * (p.clamp_min(eps).log())).sum()


class AreaGatedBankLogitHeadImproved(MetaLearnerBase):
    def __init__(
        self,
        embed_dim: int,
        *,
        bank_size: int = 256,
        normalize: bool = True,
        center: bool = True,
        learnable_temp: bool = True,
        init_temp: float = 10.0,
        min_temp: float = 1e-3,
        stat_mode: str = "logits",
        in_agg: str = "topq_mean",
        out_agg: str = "mean",
        topq: float = 0.1,
        weight_mode: str = "relu",
        weight_temp: float = 1.0,
        token_topk: Optional[int] = None,
        token_topk_on: str = "abs",
        bg_mode: str = "open_set",
        eps: float = 1e-6,
        add_diversity_loss: bool = False,
        add_sparse_loss: bool = False,
        add_usage_loss: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bank_size = bank_size
        self.normalize = normalize
        self.center = center

        self.stat_mode = stat_mode
        self.in_agg = in_agg
        self.out_agg = out_agg
        self.topq = topq
        self.weight_mode = weight_mode
        self.weight_temp = weight_temp
        self.token_topk = token_topk
        self.token_topk_on = token_topk_on
        self.bg_mode = bg_mode
        self.eps = eps
        self.min_temp = min_temp

        self.add_diversity_loss = add_diversity_loss
        self.add_sparse_loss = add_sparse_loss
        self.add_usage_loss = add_usage_loss

        # ---- BANK ----
        self.bank = nn.Parameter(torch.empty(bank_size, embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

    # ============================================================
    #                    BANK INITIALIZATION
    # ============================================================

    @torch.no_grad()
    def init_bank_from_tokens(
        self,
        tokens: torch.Tensor,  # [N,E]
        *,
        max_tokens: int = 50_000,
        kmeans_iters: int = 25,
        seed: int = 0,
    ):
        """
        Data-driven spherical k-means initialization.
        """

        K, E = self.bank.shape
        assert tokens.ndim == 2 and tokens.shape[1] == E

        if tokens.shape[0] > max_tokens:
            g = torch.Generator(device=tokens.device).manual_seed(seed)
            idx = torch.randperm(tokens.shape[0], generator=g)[:max_tokens]
            tokens = tokens[idx]

        X = tokens

        if self.center:
            X = X - X.mean(dim=0, keepdim=True)

        X = F.normalize(X, dim=-1)

        # ---- spherical kmeans ----
        idx = torch.randperm(X.shape[0], device=X.device)[:K]
        C = X[idx].clone()

        for _ in range(kmeans_iters):
            sim = X @ C.t()
            assign = sim.argmax(dim=1)

            for k in range(K):
                mask = assign == k
                if mask.any():
                    C[k] = X[mask].mean(dim=0)

            C = F.normalize(C, dim=-1)

        self.bank.data.copy_(C)

    # ============================================================
    #                      CONCEPT LOGITS
    # ============================================================

    def _concept_logits(self, x: torch.Tensor) -> torch.Tensor:
        P = self.bank

        if self.normalize:
            x = F.normalize(x, dim=-1)
            P = F.normalize(P, dim=-1)

        temp = torch.exp(self.log_temp).clamp_min(self.min_temp)
        return (x @ P.t()) * temp

    def _maybe_token_topk(self, s: torch.Tensor) -> torch.Tensor:
        if self.token_topk is None:
            return s

        k = self.token_topk
        score = s.abs() if self.token_topk_on == "abs" else s
        vals, idx = torch.topk(score, k=k, dim=-1)

        mask = torch.zeros_like(s)
        mask.scatter_(1, idx, torch.ones_like(vals))
        return s * mask

    def _stat(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(s) if self.stat_mode == "sigmoid" else s

    # ============================================================
    #                     DIVERSITY LOSS (IMPROVED)
    # ============================================================

    def bank_diversity_loss(self) -> torch.Tensor:
        B = F.normalize(self.bank, dim=-1)
        G = B @ B.t()
        I = torch.eye(G.shape[0], device=G.device)
        off_diag = G - I
        return off_diag.abs().mean()

    # ============================================================
    #                    SPARSITY & USAGE
    # ============================================================

    def _sparse_activation_loss(self, s, valid=None):
        a = torch.sigmoid(s)
        if valid is not None:
            a = a * valid
            denom = valid.sum().clamp_min(self.eps)
        else:
            denom = torch.tensor(a.shape[0], device=a.device, dtype=a.dtype)

        return a.sum() / (denom * a.shape[1])

    def _usage_loss(self, s, valid=None):
        eps = self.eps
        a = torch.sigmoid(s)

        if valid is not None:
            a = a * valid
            denom = valid.sum().clamp_min(eps)
        else:
            denom = torch.tensor(a.shape[0], device=a.device, dtype=a.dtype)

        p = a.sum(dim=0) / denom
        p = p / p.sum().clamp_min(eps)

        return (p * (p.clamp_min(eps).log())).sum()

    # -------------------------
    # masked aggregations
    # -------------------------
    def _masked_agg(
        self, v: torch.Tensor, mask: torch.Tensor, agg: str
    ) -> torch.Tensor:
        """
        v:    [N,K]
        mask: [N,1] in [0,1]
        returns: [K]
        """
        # weights per row
        w = mask.clamp(0.0, 1.0)
        denom = w.sum().clamp_min(self.eps)

        if agg == "mean":
            return (v * w).sum(dim=0) / denom

        if agg == "max":
            # masked max: set outside to very negative (for logits) or 0 (for sigmoid)
            if self.stat_mode == "logits":
                vv = v.masked_fill(w <= 0, -1e9)
                return vv.max(dim=0).values
            else:
                vv = v.masked_fill(w <= 0, 0.0)
                return vv.max(dim=0).values

        # topq_mean
        # We take only rows where mask>0, then compute top-q mean per concept.
        idx = w.squeeze(-1) > 0
        if idx.sum() == 0:
            # no pixels: return zeros
            return torch.zeros(v.shape[1], device=v.device, dtype=v.dtype)

        vv = v[idx]  # [M,K]
        M = vv.shape[0]
        q = max(1, int(math.ceil(self.topq * M)))
        topv, _ = torch.topk(vv, k=q, dim=0)
        return topv.mean(dim=0)

    def _make_weights(self, a_in: torch.Tensor, a_out: torch.Tensor) -> torch.Tensor:
        """
        a_in, a_out: [K]
        returns w: [K]
        """
        d = a_in - a_out  # [K]

        if self.weight_mode == "relu":
            w = F.relu(d)
            w = w / w.sum().clamp_min(self.eps)
            return w

        if self.weight_mode == "sigmoid":
            return torch.sigmoid(d)

        # softmax
        return F.softmax(d / max(self.weight_temp, 1e-6), dim=-1)

    # -------------------------
    # episodic API
    # -------------------------
    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C] (channel0 can be bg)
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[:2] != support_labels.shape[:2]:
            raise ValueError("support_feats and support_labels must share [S,T].")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {support_feats.shape[2]}."
            )

        S, T, E = support_feats.shape
        C = support_labels.shape[-1]

        x = support_feats.reshape(-1, E)  # [N,E]
        y = support_labels.reshape(-1, C)  # [N,C]
        valid = (
            None
            if support_valid is None
            else support_valid.reshape(-1, 1).to(dtype=x.dtype)
        )  # [N,1]

        # Episode centering
        mu = None
        if self.center:
            if valid is not None:
                denom = valid.sum().clamp_min(self.eps)
                mu = (x * valid).sum(dim=0) / denom
            else:
                mu = x.mean(dim=0)
            x = x - mu

        # Concept logits on support: [N,K]
        s = self._concept_logits(x)
        s = self._maybe_token_topk(s)
        v = self._stat(s)  # either logits or sigmoid(logits)

        # Apply valid: make invalid tokens contribute nothing to stats
        if valid is not None:
            v = v * valid
            y = y * valid

        ones = torch.ones((v.shape[0], 1), device=v.device, dtype=v.dtype)

        # Per-class weights W: [C,K]
        W = []
        for c in range(C):
            y_c = y[:, c : c + 1].clamp(0.0, 1.0)  # [N,1]

            in_mask = y_c
            out_mask = (ones - y_c).clamp(0.0, 1.0)

            a_in = self._masked_agg(v, in_mask, agg=self.in_agg)  # [K]
            a_out = self._masked_agg(v, out_mask, agg=self.out_agg)  # [K]

            w_c = self._make_weights(a_in, a_out)  # [K]
            W.append(w_c)

        W = torch.stack(W, dim=0)  # [C,K]

        ctx: Dict[str, Any] = {"W": W}
        if mu is not None:
            ctx["center"] = mu
        if self.add_diversity_loss and self.training:
            ctx["aux_bank_div"] = self.bank_diversity_loss()
        if self.add_sparse_loss and self.training:
            ctx["aux_sparse"] = self._sparse_activation_loss(s, valid=valid)
        if self.add_usage_loss and self.training:
            ctx["aux_bank_usage"] = self._usage_loss(s, valid=valid)
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        if query_feats.ndim != 3:
            raise ValueError("query_feats must be [Q,T,E].")
        Q, T, E = query_feats.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {E}.")

        xq = query_feats.reshape(-1, E)

        if self.center and "center" in support_ctx:
            xq = xq - support_ctx["center"]

        s_q = self._concept_logits(xq)
        s_q = self._maybe_token_topk(s_q)
        v_q = self._stat(s_q)  # [N,K]

        W = support_ctx["W"]  # [C,K]

        # logits: [N,C] = v_q @ W^T
        logits = v_q @ W.t()

        if self.bg_mode == "open_set" and logits.shape[1] > 1:
            logits_fg = logits[:, 1:]  # [N,C-1]
            logit_bg = -logits_fg.max(dim=1, keepdim=True).values
            logits = torch.cat([logit_bg, logits_fg], dim=1)

        return logits


class AreaGatedBankLogitHeadFiLM(MetaLearnerBase):
    def __init__(
        self,
        embed_dim: int,
        *,
        bank_size: int = 256,
        normalize: bool = True,
        center: bool = True,
        # temperature:
        learnable_temp: bool = True,
        init_temp: float = 5.0,
        min_temp: float = 1e-3,
        # activation used to compute statistics:
        stat_mode: str = "logits",  # "logits" | "sigmoid"
        # presence aggregator inside FG:
        in_agg: str = "topq_mean",  # "mean" | "max" | "topq_mean"
        topq: float = 0.1,
        # outside aggregator:
        out_agg: str = "mean",  # "mean" | "max" | "topq_mean"
        # weight transform:
        weight_mode: str = "relu",  # "relu" | "softmax" | "sigmoid"
        weight_temp: float = 1.0,
        # optional sparsification of concepts at token level:
        token_topk: Optional[int] = None,
        token_topk_on: str = "abs",
        # background:
        bg_mode: str = "open_set",  # "open_set" | "explicit"
        eps: float = 1e-6,
        add_diversity_loss: bool = True,
        add_sparse_loss: bool = True,
        add_usage_loss: bool = True,
        # -------- Option A additions ----------
        adapt_bank: bool = True,
        adapt_use_fg_only: bool = True,  # summary from foreground tokens only (recommended)
        adapt_hidden_mult: int = 2,  # film MLP width multiplier
        adapt_gamma_scale: float = 1.5,  # gamma = sigmoid(.) * scale  -> (0,scale)
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bank_size = int(bank_size)
        self.normalize = bool(normalize)
        self.center = bool(center)

        self.add_diversity_loss = bool(add_diversity_loss)
        self.add_sparse_loss = bool(add_sparse_loss)
        self.add_usage_loss = bool(add_usage_loss)

        self.stat_mode = str(stat_mode)
        if self.stat_mode not in {"logits", "sigmoid"}:
            raise ValueError(
                f"stat_mode must be 'logits'|'sigmoid', got {self.stat_mode!r}"
            )

        self.in_agg = str(in_agg)
        self.out_agg = str(out_agg)
        for m in (self.in_agg, self.out_agg):
            if m not in {"mean", "max", "topq_mean"}:
                raise ValueError(f"agg must be 'mean'|'max'|'topq_mean', got {m!r}")

        self.topq = float(topq)
        if not (0.0 < self.topq <= 1.0):
            raise ValueError(f"topq must be in (0,1], got {self.topq}")

        self.weight_mode = str(weight_mode)
        if self.weight_mode not in {"relu", "softmax", "sigmoid"}:
            raise ValueError(
                f"weight_mode must be 'relu'|'softmax'|'sigmoid', got {self.weight_mode!r}"
            )
        self.weight_temp = float(weight_temp)

        self.token_topk = token_topk
        self.token_topk_on = str(token_topk_on)
        if self.token_topk_on not in {"abs", "pos"}:
            raise ValueError(
                f"token_topk_on must be 'abs'|'pos', got {self.token_topk_on!r}"
            )

        self.bg_mode = str(bg_mode)
        if self.bg_mode not in {"open_set", "explicit"}:
            raise ValueError(
                f"bg_mode must be 'open_set'|'explicit', got {self.bg_mode!r}"
            )

        self.eps = float(eps)
        self.min_temp = float(min_temp)

        # Bank prototypes: [K,E]
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(init_temp)))

        # -------- Option A: FiLM episode adaptation ----------
        self.adapt_bank = bool(adapt_bank)
        self.adapt_mode = str(
            "film"
        )  # only one mode for now, placeholder for future extensions
        self.adapt_use_fg_only = bool(adapt_use_fg_only)
        self.adapt_gamma_scale = float(adapt_gamma_scale)

        if self.adapt_bank:
            if self.adapt_mode != "film":
                raise ValueError(
                    f"adapt_mode only supports 'film' for now, got {self.adapt_mode!r}"
                )
            h = int(adapt_hidden_mult) * self.embed_dim
            # outputs 2E = [gamma_logits, beta]
            self.film = nn.Sequential(
                nn.Linear(self.embed_dim, h),
                nn.ReLU(),
                nn.Linear(h, 2 * self.embed_dim),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    # -------------------------
    # Option A helpers
    # -------------------------
    def _stat(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(s) if self.stat_mode == "sigmoid" else s

    def _maybe_token_topk(self, s: torch.Tensor) -> torch.Tensor:
        if self.token_topk is None:
            return s
        k = int(self.token_topk)
        if k <= 0 or k > s.shape[1]:
            raise ValueError(f"token_topk must be in [1,K], got {k}")
        score = s.abs() if self.token_topk_on == "abs" else s
        vals, idx = torch.topk(score, k=k, dim=-1)
        mask = torch.zeros_like(s)
        mask.scatter_(dim=-1, index=idx, src=torch.ones_like(vals))
        return s * mask

    def _episode_summary(
        self,
        x: torch.Tensor,  # [N,E] (already centered if center=True)
        y: torch.Tensor,  # [N,C]
        valid: Optional[torch.Tensor],  # [N,1] float
    ) -> torch.Tensor:
        """
        Build a single episode summary vector z: [E].
        If adapt_use_fg_only=True, uses sum of FG channels (excluding channel 0).
        Otherwise uses all valid tokens equally.
        """
        if not self.adapt_bank:
            raise RuntimeError("_episode_summary called but adapt_bank=False")

        if self.adapt_use_fg_only:
            if y.shape[1] <= 1:
                # no explicit FG channels; fall back to uniform
                w = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
            else:
                w = y[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # [N,1]
        else:
            w = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

        if valid is not None:
            w = w * valid  # [N,1]

        denom = w.sum().clamp_min(self.eps)
        z = (x * w).sum(dim=0) / denom  # [E]
        # normalize for stability
        z = F.normalize(z, dim=-1)
        return z

    def _adapt_bank_film(
        self,
        P: torch.Tensor,  # [K,E] (raw bank param)
        z: torch.Tensor,  # [E]
    ) -> torch.Tensor:
        """
        FiLM adaptation: P' = normalize(P * gamma(z) + beta(z)).
        gamma is constrained positive to avoid sign flips early.
        """
        gb = self.film(z)  # [2E]
        gamma_logits, beta = gb.chunk(2, dim=-1)  # [E], [E]
        gamma = torch.sigmoid(gamma_logits) * self.adapt_gamma_scale  # (0,scale)
        P2 = P * gamma.unsqueeze(0) + beta.unsqueeze(0)  # [K,E]
        return F.normalize(P2, dim=-1)

    # -------------------------
    # concept logits
    # -------------------------
    def _concept_logits(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        x: [N,E], P: [K,E] already adapted for the episode if needed
        returns s: [N,K]
        """
        if self.normalize:
            x = F.normalize(x, dim=-1)
            P = F.normalize(P, dim=-1)

        temp = torch.exp(self.log_temp).clamp_min(self.min_temp)
        return (x @ P.t()) * temp

    # -------------------------
    # masked aggregations
    # -------------------------
    def _masked_agg(
        self, v: torch.Tensor, mask: torch.Tensor, agg: str
    ) -> torch.Tensor:
        """
        v: [N,K], mask: [N,1] in [0,1] -> [K]
        """
        w = mask.clamp(0.0, 1.0)
        denom = w.sum().clamp_min(self.eps)

        if agg == "mean":
            return (v * w).sum(dim=0) / denom

        if agg == "max":
            if self.stat_mode == "logits":
                vv = v.masked_fill(w <= 0, -1e9)
                return vv.max(dim=0).values
            vv = v.masked_fill(w <= 0, 0.0)
            return vv.max(dim=0).values

        # topq_mean
        idx = w.squeeze(-1) > 0
        if idx.sum() == 0:
            return torch.zeros(v.shape[1], device=v.device, dtype=v.dtype)
        vv = v[idx]  # [M,K]
        M = vv.shape[0]
        q = max(1, int(math.ceil(self.topq * M)))
        topv, _ = torch.topk(vv, k=q, dim=0)
        return topv.mean(dim=0)

    def _make_weights(self, a_in: torch.Tensor, a_out: torch.Tensor) -> torch.Tensor:
        d = a_in - a_out  # [K]

        if self.weight_mode == "relu":
            w = F.relu(d)
            return w / w.sum().clamp_min(self.eps)

        if self.weight_mode == "sigmoid":
            return torch.sigmoid(d)

        return F.softmax(d / max(self.weight_temp, 1e-6), dim=-1)

    # -------------------------
    # episodic API
    # -------------------------
    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,  # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[:2] != support_labels.shape[:2]:
            raise ValueError("support_feats and support_labels must share [S,T].")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {support_feats.shape[2]}."
            )

        S, T, E = support_feats.shape
        C = support_labels.shape[-1]

        x = support_feats.reshape(-1, E)  # [N,E]
        y = support_labels.reshape(-1, C).to(x.dtype)  # [N,C]
        valid = None
        if support_valid is not None:
            valid = support_valid.reshape(-1, 1).to(dtype=x.dtype)  # [N,1]

        # Episode centering
        mu = None
        if self.center:
            if valid is not None:
                denom = valid.sum().clamp_min(self.eps)
                mu = (x * valid).sum(dim=0) / denom
            else:
                mu = x.mean(dim=0)
            x = x - mu

        # -------- Option A: adapt bank per episode --------
        P = self.bank
        if self.adapt_bank:
            z = self._episode_summary(x=x, y=y, valid=valid)  # [E]
            P = self._adapt_bank_film(P=P, z=z)  # [K,E]

        # Concept logits on support
        s = self._concept_logits(x, P=P)  # [N,K]
        s = self._maybe_token_topk(s)
        v = self._stat(s)

        if valid is not None:
            v = v * valid
            y = y * valid

        ones = torch.ones((v.shape[0], 1), device=v.device, dtype=v.dtype)

        W = []
        for c in range(C):
            y_c = y[:, c : c + 1].clamp(0.0, 1.0)  # [N,1]
            in_mask = y_c
            out_mask = (ones - y_c).clamp(0.0, 1.0)

            a_in = self._masked_agg(v, in_mask, agg=self.in_agg)  # [K]
            a_out = self._masked_agg(v, out_mask, agg=self.out_agg)  # [K]
            w_c = self._make_weights(a_in, a_out)  # [K]
            W.append(w_c)

        W = torch.stack(W, dim=0)  # [C,K]

        ctx: Dict[str, Any] = {"W": W}
        if mu is not None:
            ctx["center"] = mu
        if self.adapt_bank:
            # store episode-adapted bank OR store z and recompute bank later
            # storing P is simplest; it's [K,E] and cheap for K~256
            ctx["P_episode"] = P

        if self.add_diversity_loss and self.training:
            ctx["aux_bank_div"] = self.bank_diversity_loss()
        if self.add_sparse_loss and self.training:
            ctx["aux_sparse"] = self._sparse_activation_loss(s, valid=valid)
        if self.add_usage_loss and self.training:
            ctx["aux_bank_usage"] = self._usage_loss(s, valid=valid)
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        if query_feats.ndim != 3:
            raise ValueError("query_feats must be [Q,T,E].")
        Q, T, E = query_feats.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {E}.")

        xq = query_feats.reshape(-1, E)

        if self.center and "center" in support_ctx:
            xq = xq - support_ctx["center"]

        # Use adapted bank if available; else global bank
        P = support_ctx.get("P_episode", self.bank)

        s_q = self._concept_logits(xq, P=P)
        s_q = self._maybe_token_topk(s_q)
        v_q = self._stat(s_q)  # [N,K]

        W = support_ctx["W"]  # [C,K]
        logits = v_q @ W.t()  # [N,C]

        if self.bg_mode == "open_set" and logits.shape[1] > 1:
            logits_fg = logits[:, 1:]
            logit_bg = -logits_fg.max(dim=1, keepdim=True).values
            logits = torch.cat([logit_bg, logits_fg], dim=1)

        return logits

    # -------------------------
    # aux losses (unchanged)
    # -------------------------
    def bank_diversity_loss(self) -> torch.Tensor:
        B = F.normalize(self.bank, dim=-1)
        G = B @ B.t()
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).mean()

    def _sparse_activation_loss(
        self, s: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        a = torch.sigmoid(s)
        if valid is not None:
            a = a * valid
            denom = valid.sum().clamp_min(self.eps)
        else:
            denom = torch.tensor(a.shape[0], device=a.device, dtype=a.dtype)
        return a.sum() / (denom * a.shape[1])

    def _usage_loss(
        self, s: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        eps = self.eps
        a = torch.sigmoid(s)

        if valid is not None:
            a = a * valid
            denom = valid.sum().clamp_min(eps)
        else:
            denom = torch.tensor(a.shape[0], device=a.device, dtype=a.dtype)

        p = a.sum(dim=0) / denom
        p = p / p.sum().clamp_min(eps)

        # KL(p || U) = sum p log p + log K   (U is uniform over K concepts)
        K = p.numel()
        return (p * p.clamp_min(eps).log()).sum() + math.log(K)

    @torch.no_grad()
    def init_bank_from_tokens(
        self,
        tokens: torch.Tensor,  # [N,E]
        *,
        max_tokens: int = 50_000,
        kmeans_iters: int = 25,
        seed: int = 0,
    ):
        """
        Data-driven spherical k-means initialization.
        """

        K, E = self.bank.shape
        assert tokens.ndim == 2 and tokens.shape[1] == E

        if tokens.shape[0] > max_tokens:
            g = torch.Generator(device=tokens.device).manual_seed(seed)
            idx = torch.randperm(tokens.shape[0], generator=g)[:max_tokens]
            tokens = tokens[idx]

        X = tokens

        if self.center:
            X = X - X.mean(dim=0, keepdim=True)

        X = F.normalize(X, dim=-1)

        # ---- spherical kmeans ----
        idx = torch.randperm(X.shape[0], device=X.device)[:K]
        C = X[idx].clone()

        for _ in range(kmeans_iters):
            sim = X @ C.t()
            assign = sim.argmax(dim=1)

            for k in range(K):
                mask = assign == k
                if mask.any():
                    C[k] = X[mask].mean(dim=0)

            C = F.normalize(C, dim=-1)

        self.bank.data.copy_(C)


class ClassCrossAttnPool(nn.Module):
    """
    Cross-attention from per-class queries to support tokens.

    queries: [C,H]
    support: [N,H]
    mask_s:  [N] bool, True=valid (optional)
    returns: [C,H]
    """

    def __init__(self, h_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # we'll use [B,L,H]
        )
        self.ln_q = nn.LayerNorm(h_dim)
        self.ln_o = nn.LayerNorm(h_dim)
        self.ff = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(
        self,
        *,
        queries: torch.Tensor,  # [C,H]
        support: torch.Tensor,  # [N,H]
        mask_s: Optional[torch.Tensor] = None,  # [N] True=valid
    ) -> torch.Tensor:
        if queries.ndim != 2 or support.ndim != 2:
            raise ValueError("queries must be [C,H] and support must be [N,H].")
        C, H = queries.shape
        N, H2 = support.shape
        if H != H2:
            raise ValueError(f"Dim mismatch: queries H={H}, support H={H2}")

        # batchify
        q = queries.unsqueeze(0)  # [1,C,H]
        k = support.unsqueeze(0)  # [1,N,H]
        v = support.unsqueeze(0)  # [1,N,H]

        # MultiheadAttention uses key_padding_mask where True means "ignore/pad"
        key_padding_mask = None
        if mask_s is not None:
            if mask_s.ndim != 1 or mask_s.shape[0] != N:
                raise ValueError(f"mask_s must be [N], got {mask_s.shape}")
            key_padding_mask = (~mask_s.bool()).unsqueeze(0)  # [1, N]

        qn = self.ln_q(q)
        out, _ = self.mha(
            qn, k, v, key_padding_mask=key_padding_mask, need_weights=False
        )  # [1,C,H]
        out = out + q  # residual
        out = self.ln_o(out)
        out = out + self.ff(out)

        return out.squeeze(0)  # [C,H]


class MetaLinearHeadMC_Attn(MetaLearnerBase):
    """
    Multiclass meta linear head with attention pooling (single episode) in TOKEN space.

    Expected inputs:
      support_feats:  [S,T,E]
      support_labels: [S,T,C]   (soft or one-hot, in [0,1])
      support_valid:  [S,T]     bool or {0,1} (True/1 = valid token)

      query_feats:    [Q,T,E]

    Output:
      predict_query returns logits_flat: [Q*T, C]  (to match your existing pipeline)
    """

    def __init__(
        self,
        embed_dim: int,
        h_dim: int = 256,
        num_heads: int = 4,
        use_gate: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.h_dim = int(h_dim)
        self.use_gate = bool(use_gate)
        self.eps = float(eps)

        self.phi = nn.Sequential(
            nn.Linear(self.embed_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        # Must implement: attn_pool(queries: [C,H], support: [N,H], mask_s: Optional[[N]] ) -> [C,H]
        self.attn_pool = ClassCrossAttnPool(h_dim=self.h_dim, num_heads=num_heads)

        out_dim = self.embed_dim + 1 + (self.embed_dim if self.use_gate else 0)
        self.psi = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, out_dim),
        )

        # global anchor / prior
        self.W0 = nn.Parameter(torch.zeros(self.embed_dim))  # [E]
        self.b0 = nn.Parameter(torch.zeros(1))  # [1]

    @staticmethod
    def _flatten_ST(x: torch.Tensor) -> torch.Tensor:
        # [S,T,...] -> [N,...]
        return x.reshape(-1, *x.shape[2:])

    def _masked_weighted_mean(
        self,
        Hs: torch.Tensor,  # [N,H]
        ys: torch.Tensor,  # [N,C]
        mask_s: Optional[torch.Tensor],  # [N] bool/float, True=valid
    ) -> torch.Tensor:
        """
        Returns initial class queries Q0: [C,H] = masked label-weighted mean of Hs.
        """
        if mask_s is not None:
            m = mask_s.to(dtype=Hs.dtype).unsqueeze(-1)  # [N,1]
            Hs = Hs * m  # [N,H]
            ys = ys * m  # [N,C]

        denom = ys.sum(dim=0).clamp_min(self.eps)  # [C]
        Q0 = (Hs.unsqueeze(1) * ys.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(
            -1
        )  # [C,H]
        return Q0

    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,  # [S,T,E]
        support_labels: torch.Tensor,  # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,  # [S,T] bool or 0/1
    ) -> Dict[str, torch.Tensor]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError(
                "support_feats must be [S,T,E] and support_labels [S,T,C]."
            )
        if support_feats.shape[:2] != support_labels.shape[:2]:
            raise ValueError("support_feats and support_labels must share [S,T].")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {support_feats.shape[2]}."
            )

        S, T, E = support_feats.shape
        C = support_labels.shape[-1]

        # Flatten token dimension
        Xs = support_feats.reshape(-1, E)  # [N,E]
        ys = support_labels.reshape(-1, C).to(dtype=support_feats.dtype)

        mask = None
        if support_valid is not None:
            if support_valid.shape != (S, T):
                raise ValueError(
                    f"support_valid must be [S,T], got {support_valid.shape}."
                )
            mask = support_valid.reshape(-1)  # [N] (bool/0-1)

        # Project tokens
        Hs = self.phi(Xs)  # [N,H]

        # Initial class queries (proto-like), refined by cross-attention over support tokens
        Q0 = self._masked_weighted_mean(Hs, ys, mask)  # [C,H]
        Rc = self.attn_pool(
            queries=Q0,
            support=Hs,
            mask_s=mask if mask is None else mask.bool(),  # [N] True=valid
        )

        # Produce linear classifier params per class
        params = self.psi(Rc)  # [C, out_dim]
        if self.use_gate:
            dW = params[:, :E]  # [C,E]
            db = params[:, E : E + 1].squeeze(-1)  # [C]
            gate = torch.sigmoid(params[:, E + 1 :])  # [C,E]
            W = self.W0.unsqueeze(0) + gate * dW  # [C,E]
        else:
            W = self.W0.unsqueeze(0) + params[:, :E]
            db = params[:, E : E + 1].squeeze(-1)

        b = self.b0.squeeze(0) + db  # [C]
        return {"W": W, "b": b}

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,E]
        support_ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if query_feats.ndim != 3:
            raise ValueError("query_feats must be [Q,T,E].")
        Q, T, E = query_feats.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {E}.")

        W = support_ctx["W"]  # [C,E]
        b = support_ctx["b"]  # [C]

        Xq = query_feats.reshape(-1, E)  # [Q*T, E]
        logits = Xq @ W.t() + b.unsqueeze(0)  # [Q*T, C]
        return logits
