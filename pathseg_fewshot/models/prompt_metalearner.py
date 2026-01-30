from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    """
    Pre-norm cross-attention: queries attend to kv.
    Designed for small #queries (prompt tokens) and large kv (support/query tokens).
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q:  [B, M, D]
        kv: [B, N, D]
        attn_mask: optional mask for attention weights.
                   For batch_first MHA, attn_mask can be [B*M, N] (bool) or [M, N].
        """
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)
        out, _ = self.attn(qn, kvn, kvn, attn_mask=attn_mask, need_weights=False)
        q = q + out
        q = q + self.mlp(self.mlp_norm(q))
        return q


class PromptMetaLearner(nn.Module):
    """
    SAM-spirit meta-learner:
      1) Build per-class prompt tokens from support tokens using learnable queries + cross-attention.
      2) Decode query logits via prompt->query cross-attention.
      3) Produce token logits by dot-product between query tokens and refined class prompts.

    No support self-attention, no prototypes. Label info is used softly (weights), not hard sampling.

    Inputs:
      support_feats:  [S,T,D]
      support_labels: [S,T,C]  (soft/hard in [0,1])
      support_valid:  [S,T] optional bool
      query_feats:    [Q,T,D]

    Output:
      logits_flat: [Q*T, C]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        prompt_layers: int = 1,
        decode_layers: int = 1,
        mlp_ratio: float = 4.0,
        use_prompt_self_attn: bool = True,
        label_bias_strength: float = 0.0,  # set >0 to gently bias attention using labels
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.prompt_layers = int(prompt_layers)
        self.decode_layers = int(decode_layers)
        self.mlp_ratio = float(mlp_ratio)
        self.use_prompt_self_attn = bool(use_prompt_self_attn)
        self.label_bias_strength = float(label_bias_strength)
        self.eps = float(eps)

        # A tiny prompt token "decoder" that extracts class prompts from support tokens.
        self.prompt_blocks = nn.ModuleList(
            [CrossAttnBlock(self.d_model, self.n_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.prompt_layers)]
        )

        # Optionally refine prompts with cheap self-attn (only on C tokens, so tiny cost).
        self.prompt_self = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
        self.prompt_self_norm = nn.LayerNorm(self.d_model)

        # Decode: prompts attend to query tokens (SAM-like prompt->image interaction).
        self.decode_blocks = nn.ModuleList(
            [CrossAttnBlock(self.d_model, self.n_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.decode_layers)]
        )

        # Optional normalization before scoring
        self.score_norm_q = nn.LayerNorm(self.d_model)
        self.score_norm_p = nn.LayerNorm(self.d_model)

    def _build_class_queries(self, C: int, device, dtype) -> torch.Tensor:
        """
        Learnable class queries would typically be parameters, but C varies per episode.
        So we use a small base parameter set for a max C and slice, OR generate queries.

        Simplest: maintain a maximum and slice. Here we do max_C=32 by default.
        """
        raise RuntimeError("Use PromptMetaLearnerWithMaxC instead (see below).")

    @staticmethod
    def _make_valid_mask(support_valid: Optional[torch.Tensor], S: int, T: int, device) -> torch.Tensor:
        if support_valid is None:
            return torch.ones((S, T), device=device, dtype=torch.bool)
        return support_valid

    def _label_attention_bias(
        self,
        *,
        labels: torch.Tensor,   # [S*T, C] or [B, N, C]
        C: int,
    ) -> torch.Tensor:
        """
        Returns additive bias to attention logits, shape [C, N] (or broadcastable).
        We keep it simple: bias_{c,i} = log(w_{i,c}+eps)
        Used to *nudge* attention, not hard select tokens.
        """
        # labels_flat: [N, C]
        w = labels.clamp_min(0.0)
        bias = torch.log(w + self.eps)  # [N,C]
        bias = bias.transpose(0, 1)     # [C,N]
        return bias


class PromptMetaLearnerWithMaxC(PromptMetaLearner):
    """
    Same as PromptMetaLearner, but supports variable C by using a bank of learnable queries.
    """
    def __init__(
        self,
        d_model: int,
        max_classes: int = 16,   # enough for your K+1 (bg + <=2 ways => <=3)
        **kwargs,
    ):
        super().__init__(d_model=d_model, **kwargs)
        self.max_classes = int(max_classes)
        self.class_queries = nn.Parameter(torch.randn(self.max_classes, self.d_model) * 0.02)

    def fit_support(
        self,
        *,
        support_feats: torch.Tensor,    # [S,T,D]
        support_labels: torch.Tensor,   # [S,T,C]
        support_valid: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if support_feats.ndim != 3:
            raise ValueError(f"support_feats must be [S,T,D], got {support_feats.shape}")
        if support_labels.ndim != 3:
            raise ValueError(f"support_labels must be [S,T,C], got {support_labels.shape}")
        if support_feats.shape[:2] != support_labels.shape[:2]:
            raise ValueError("support_feats and support_labels must match on [S,T].")

        S, T, D = support_feats.shape
        C = support_labels.shape[2]
        if C > self.max_classes:
            raise ValueError(f"C={C} exceeds max_classes={self.max_classes}. Increase max_classes.")

        device = support_feats.device
        dtype = support_feats.dtype

        # Flatten support tokens across support images (episode-level)
        kv = support_feats.reshape(S * T, D).unsqueeze(0)          # [1, Ns, D]
        labels_flat = support_labels.reshape(S * T, C)             # [Ns, C]

        valid = self._make_valid_mask(support_valid, S, T, device).reshape(S * T)  # [Ns]
        # Mask invalid tokens by setting their label weights to 0 and optionally masking attention
        labels_flat = labels_flat * valid.unsqueeze(-1).to(labels_flat.dtype)

        # Initialize C prompt tokens from learnable bank
        prompts = self.class_queries[:C].to(device=device, dtype=dtype).unsqueeze(0)  # [1, C, D]

        # Optional: add a gentle label bias to attention logits (nudges prompts towards relevant tokens)
        attn_mask = None
        # MultiheadAttention doesn't support per-key additive bias directly, so we keep it simple:
        # - either (a) rely on label-conditioned kv weighting (below), or
        # - (b) pre-weight kv per class (heavy), not recommended.
        #
        # Instead, we do a cheap trick: incorporate label weights by scaling keys/values globally:
        # For noisy labels, keep it gentle: mix labels into a "relevance" scalar.
        if self.label_bias_strength > 0:
            # relevance per token (max over classes) keeps it label-agnostic-ish.
            rel = labels_flat.max(dim=1).values  # [Ns]
            rel = (rel / (rel.max().clamp_min(self.eps))).clamp(0.0, 1.0)
            rel = (1.0 - self.label_bias_strength) + self.label_bias_strength * rel  # [Ns] in [1-a,1]
            kv = kv * rel.view(1, -1, 1).to(kv.dtype)

        # Extract prompts from support tokens with a few cross-attn blocks
        for blk in self.prompt_blocks:
            prompts = blk(prompts, kv, attn_mask=attn_mask)

        # Cheap refinement among prompts
        if self.use_prompt_self_attn and C > 1:
            p = self.prompt_self_norm(prompts)
            p2, _ = self.prompt_self(p, p, p, need_weights=False)
            prompts = prompts + p2

        # Store prompts and C for decoding
        ctx: Dict[str, torch.Tensor] = {
            "prompts": prompts.squeeze(0),  # [C,D]
        }
        return ctx

    def predict_query(
        self,
        *,
        query_feats: torch.Tensor,  # [Q,T,D]
        support_ctx: Dict[str, torch.Tensor],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if query_feats.ndim != 3:
            raise ValueError(f"query_feats must be [Q,T,D], got {query_feats.shape}")

        Q, T, D = query_feats.shape
        prompts = support_ctx["prompts"]  # [C,D]
        C = prompts.shape[0]

        # Repeat prompts per query image
        prompts_q = prompts.unsqueeze(0).expand(Q, C, D).contiguous()  # [Q,C,D]
        kv = query_feats  # [Q,T,D]

        # Decode: prompts attend to query tokens (SAM-like interaction)
        for blk in self.decode_blocks:
            prompts_q = blk(prompts_q, kv)

        # Score tokens: dot-product between query tokens and refined prompts
        # logits: [Q,T,C]
        qn = self.score_norm_q(query_feats)
        pn = self.score_norm_p(prompts_q)
        logits = torch.einsum("qtd,qcd->qtc", qn, pn)  # [Q,T,C]

        return logits.reshape(Q * T, C)
