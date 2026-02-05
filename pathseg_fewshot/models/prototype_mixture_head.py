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
        normalize_feats: bool = True,   # cosine geometry
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
        bg_mode: str = "open_set",      # "open_set" | "global"
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
            raise ValueError(f"bg_mode must be 'open_set' or 'global', got {self.bg_mode!r}")

        # Learnable bank P: [Kbank, E]
        self.bank = nn.Parameter(torch.empty(self.bank_size, self.embed_dim))
        nn.init.trunc_normal_(self.bank, std=0.02)

        # token->bank assignment temperature (higher => softer, lower => harder)
        if learnable_assign_temp:
            self.log_assign_temp = nn.Parameter(torch.log(torch.tensor(init_assign_temp)))
        else:
            self.register_buffer("log_assign_temp", torch.log(torch.tensor(init_assign_temp)))

        # query scoring temperature
        if learnable_score_temp:
            self.log_score_temp = nn.Parameter(torch.log(torch.tensor(init_score_temp)))
        else:
            self.register_buffer("log_score_temp", torch.log(torch.tensor(init_score_temp)))

    # -------------------------
    # helpers
    # -------------------------
    def _apply_mask(
        self,
        feats: torch.Tensor,                 # [N,E]
        labels: torch.Tensor,                # [N,C]
        valid: Optional[torch.Tensor],       # [N] bool
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
        r: torch.Tensor,      # [N,Kbank]
        y: torch.Tensor,      # [N,1] weights in [0,1] for class c
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
        support_feats: torch.Tensor,                  # [S,T,E]
        support_labels: torch.Tensor,                 # [S,T,C]
        support_valid: Optional[torch.Tensor] = None, # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError("support_feats must be [S,T,E] and support_labels [S,T,C].")
        if support_feats.shape[0] != support_labels.shape[0]:
            raise ValueError("support_feats and support_labels must share S.")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected support_feats E={self.embed_dim}, got {support_feats.shape[2]}."
            )

        # Flatten
        z = support_feats.reshape(-1, support_feats.shape[-1])              # [N,E]
        y = support_labels.reshape(-1, support_labels.shape[-1])            # [N,C]
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
            y_c = y[:, c:c+1].clamp(0.0, 1.0)
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
        query_feats: torch.Tensor,       # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        z = query_feats.reshape(-1, query_feats.shape[-1])  # [N,E]

        if self.center and "center" in support_ctx:
            z = z - support_ctx["center"]

        if self.normalize_feats:
            z = F.normalize(z, dim=-1)

        prototypes = support_ctx["prototypes"]  # [C,E]
        logits = z @ prototypes.t()             # [N,C]

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
        r: torch.Tensor,                       # [N,Kbank]
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
        r: torch.Tensor,                       # [N,Kbank]
        labels: torch.Tensor,                  # [N,C] (0=bg, 1..=fg)
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
        normalize_feats: bool = True,     # strongly recommended for cosine likelihood
        # kappa controls likelihood sharpness
        learnable_kappa: bool = True,
        init_kappa: float = 20.0,
        min_kappa: float = 1e-3,
        # Dirichlet prior strength for pi smoothing (fixes underrepresented concepts)
        dirichlet_alpha: float = 1e-2,
        # background handling
        bg_mode: str = "open_set",        # "open_set" | "learned_global"
        learn_bg_global: bool = False,    # if bg_mode=="learned_global"
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
            raise ValueError(f"bg_mode must be 'open_set' or 'learned_global', got {self.bg_mode!r}")

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
        feats: torch.Tensor,                 # [N,E]
        labels: torch.Tensor,                # [N,C]
        valid: Optional[torch.Tensor],       # [N] bool
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
        support_feats: torch.Tensor,                  # [S,T,E]
        support_labels: torch.Tensor,                 # [S,T,C]
        support_valid: Optional[torch.Tensor] = None, # [S,T]
    ) -> Dict[str, Any]:
        if support_feats.ndim != 3 or support_labels.ndim != 3:
            raise ValueError("support_feats must be [S,T,E] and support_labels [S,T,C].")
        if support_feats.shape[2] != self.embed_dim:
            raise ValueError(f"Expected E={self.embed_dim}, got {support_feats.shape[2]}.")

        z = support_feats.reshape(-1, support_feats.shape[-1])               # [N,E]
        y = support_labels.reshape(-1, support_labels.shape[-1])             # [N,C]
        valid = None if support_valid is None else support_valid.reshape(-1) # [N]

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
        loglik = self._log_p_z_given_k(z)          # [N,K]
        r = F.softmax(loglik, dim=-1)              # [N,K]

        # Dirichlet-smoothed mixture weights per class
        # counts: [C,K] = y^T @ r
        counts = y.transpose(0, 1) @ r             # [C,K]
        alpha = self.dirichlet_alpha
        pi = counts + alpha                        # [C,K]
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        ctx: Dict[str, Any] = {"pi": pi}           # [C,K]
        if mu is not None:
            ctx["center"] = mu
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,                 # [Q,T,E]
        support_ctx: Dict[str, Any],
    ) -> torch.Tensor:
        z = query_feats.reshape(-1, query_feats.shape[-1])  # [N,E]

        if self.center and "center" in support_ctx:
            z = z - support_ctx["center"]

        loglik = self._log_p_z_given_k(z)                  # [N,K]

        pi = support_ctx["pi"]                             # [C,K]
        log_pi = (pi.clamp_min(self.eps)).log()            # [C,K]

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
                log_bg_pi = bg_pi.log().unsqueeze(0)                           # [1,K]
                logit_bg = torch.logsumexp(loglik + log_bg_pi, dim=-1, keepdim=True)
                if logits.shape[1] > 1:
                    logits_fg = logits[:, 1:]
                    logits = torch.cat([logit_bg, logits_fg], dim=1)
                else:
                    logits = logit_bg

        return logits
