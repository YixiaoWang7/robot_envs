"""Minimal Transformer components (from scratch) with optional attention score returns and KV-cache.

This module provides:
- Self-attention
- Cross-attention
- KV cache (for incremental decoding and/or caching encoder memory projections)
- Transformer block
- Multi-layer transformer stack

Design goals:
- Small, explicit, readable (no nn.TransformerEncoder dependency)
- Efficient: uses batched matmuls; optionally leverages PyTorch SDPA when available
- Flexible masking: supports causal masks and explicit additive/bool masks
- Introspection: optionally returns attention scores (works for self- and cross-attention)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ----------------------------
# Utilities
# ----------------------------


def _shape_heads(x: Tensor, n_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh)"""
    b, t, d = x.shape
    if d % n_heads != 0:
        raise ValueError(f"embed_dim={d} must be divisible by num_heads={n_heads}")
    dh = d // n_heads
    return x.view(b, t, n_heads, dh).transpose(1, 2).contiguous()


def _unshape_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D)"""
    b, h, t, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, h * dh)


def _make_causal_mask(t_q: int, t_k: int, device: torch.device) -> Tensor:
    """Returns a boolean mask of shape (t_q, t_k) where True means 'blocked'."""
    # For q position i, keys > i are blocked.
    # If t_k can be larger (cached prefix), we block only the future portion relative to the query.
    # This corresponds to a standard causal mask for the "current" query range.
    i = torch.arange(t_q, device=device)[:, None]
    j = torch.arange(t_k, device=device)[None, :]
    return j > (t_k - t_q + i)


def _merge_attention_masks(
    attn_mask: Optional[Tensor],
    causal: bool,
    t_q: int,
    t_k: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Merge user-provided attn_mask with optional causal mask.

    Supported formats:
    - Bool mask: True means blocked. Shapes: (t_q, t_k) or broadcastable to (B, H, t_q, t_k)
    - Float/additive mask: additive values added to attention logits (e.g. 0 or -inf).
      Shapes: (t_q, t_k) or broadcastable.
    """
    if not causal and attn_mask is None:
        return None

    causal_mask = _make_causal_mask(t_q=t_q, t_k=t_k, device=device) if causal else None

    if attn_mask is None:
        return causal_mask
    if causal_mask is None:
        return attn_mask

    # If both are bool masks -> OR them
    if attn_mask.dtype == torch.bool and causal_mask.dtype == torch.bool:
        return attn_mask | causal_mask

    # Convert bool causal to additive if needed
    if attn_mask.dtype != torch.bool and causal_mask.dtype == torch.bool:
        # additive mask expects shape broadcastable; we create 0/-inf mask
        neg_inf = torch.finfo(attn_mask.dtype).min
        causal_add = torch.zeros((t_q, t_k), device=device, dtype=attn_mask.dtype)
        causal_add = causal_add.masked_fill(causal_mask, neg_inf)
        return attn_mask + causal_add

    # Convert bool attn_mask to additive if needed
    if attn_mask.dtype == torch.bool and causal_mask.dtype != torch.bool:
        neg_inf = torch.finfo(causal_mask.dtype).min
        add = torch.zeros((t_q, t_k), device=device, dtype=causal_mask.dtype)
        add = add.masked_fill(attn_mask, neg_inf)
        return add + causal_mask

    # Both additive -> add
    return attn_mask + causal_mask


def _apply_kv_padding_mask(
    attn_logits: Tensor,
    key_padding_mask: Optional[Tensor],
) -> Tensor:
    """Apply key padding mask to attention logits.

    Args:
        attn_logits: (B, H, Tq, Tk)
        key_padding_mask: (B, Tk) bool where True indicates padding positions to block.
    """
    if key_padding_mask is None:
        return attn_logits
    if key_padding_mask.dtype != torch.bool:
        raise ValueError("key_padding_mask must be a boolean tensor with shape (B, Tk)")
    if key_padding_mask.ndim != 2:
        raise ValueError(f"key_padding_mask must have shape (B, Tk), got {key_padding_mask.shape}")
    # Broadcast to (B, 1, 1, Tk)
    return attn_logits.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(attn_logits.dtype).min)


def _masked_softmax(attn_logits: Tensor) -> Tensor:
    # Safe softmax even with large negative values.
    return F.softmax(attn_logits, dim=-1)


# ----------------------------
# KV Cache
# ----------------------------


@dataclass
class KVCacheEntry:
    """Per-layer KV cache.

    Stores already-projected keys/values in (B, H, T, Dh) format for efficiency.
    """

    k: Optional[Tensor] = None
    v: Optional[Tensor] = None

    def append(self, k_new: Tensor, v_new: Tensor) -> None:
        """Append new tokens along time dimension."""
        if self.k is None:
            self.k = k_new
            self.v = v_new
            return
        self.k = torch.cat([self.k, k_new], dim=2)
        self.v = torch.cat([self.v, v_new], dim=2)

    @property
    def seq_len(self) -> int:
        if self.k is None:
            return 0
        return int(self.k.shape[2])


class KVCache:
    """KV cache container keyed by layer index and cache name.

    Common usage patterns:
    - Self-attention incremental decoding: append projected k/v for each new step.
    - Cross-attention: precompute and store projected memory k/v once (static).
    """

    def __init__(self):
        self._store: Dict[Tuple[str, int], KVCacheEntry] = {}

    def get(self, name: str, layer_idx: int) -> KVCacheEntry:
        key = (name, layer_idx)
        if key not in self._store:
            self._store[key] = KVCacheEntry()
        return self._store[key]

    def clear(self) -> None:
        self._store.clear()


# ----------------------------
# Attention core
# ----------------------------


class AttentionScores:
    """Return container for attention diagnostics.

    - attn_weights: (B, H, Tq, Tk) attention probabilities (after softmax)
    - attn_logits: (B, H, Tq, Tk) pre-softmax scores (optional)
    """

    def __init__(self, attn_weights: Tensor, attn_logits: Optional[Tensor] = None):
        self.attn_weights = attn_weights
        self.attn_logits = attn_logits


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    sparse_topk: Optional[int] = None,
    sparse_topk_ratio: Optional[float] = None,
    # Sparse attention mode:
    # - "hard": forward/backward both use top-k masked softmax (fastest, but k=1 can kill Q/K gradients)
    # - "soft": straight-through top-k (forward uses top-k weights, backward uses dense softmax gradients)
    sparse_mode: str = "hard",
    dropout_p: float = 0.0,
    return_attention: bool = False,
    use_sdpa_if_available: bool = True,
) -> Tuple[Tensor, Optional[AttentionScores]]:
    """Scaled dot-product attention.

    Args:
        q, k, v: (B, H, T, Dh)
        attn_mask: optional mask; bool (blocked=True) or additive float mask.
                   Shape must be broadcastable to (B, H, Tq, Tk) or be (Tq, Tk).
        key_padding_mask: (B, Tk) bool, True blocks key positions.
        return_attention: if True, returns AttentionScores (weights and optionally logits).
        use_sdpa_if_available: uses torch SDPA when available; still supports returning weights via fallback.
    """
    b, h, t_q, dh = q.shape
    t_k = k.shape[2]

    if sparse_topk is not None and sparse_topk_ratio is not None:
        raise ValueError("Provide only one of sparse_topk or sparse_topk_ratio (not both).")
    if sparse_topk_ratio is not None and sparse_topk_ratio <= 0:
        raise ValueError(f"sparse_topk_ratio must be > 0 when provided, got {sparse_topk_ratio}.")
    if sparse_topk is not None and sparse_topk <= 0:
        raise ValueError(f"sparse_topk must be > 0 when provided, got {sparse_topk}.")

    # Derive an effective K for top-k (if any). Clamp to [1, t_k].
    effective_topk: Optional[int] = None
    if sparse_topk_ratio is not None:
        effective_topk = int(max(1, min(t_k, round(float(sparse_topk_ratio) * float(t_k)))))
    elif sparse_topk is not None:
        effective_topk = int(max(1, min(t_k, int(sparse_topk))))
    if effective_topk is not None and effective_topk >= t_k:
        effective_topk = None  # no-op

    if sparse_mode not in ("hard", "soft"):
        raise ValueError(f"Unknown sparse_mode={sparse_mode!r} (expected 'hard' or 'soft').")

    # If user wants attention weights, we use the explicit path (SDPA doesn't expose weights cleanly).
    can_use_sdpa = (
        use_sdpa_if_available
        and hasattr(F, "scaled_dot_product_attention")
        and (not return_attention)
        and (effective_topk is None)  # sparse top-k needs logits/topk -> use manual path
        and key_padding_mask is None  # SDPA supports some masks, but we keep behavior predictable.
        and (attn_mask is None or attn_mask.dtype in (torch.bool, torch.float16, torch.float32, torch.bfloat16))
    )

    if can_use_sdpa:
        # SDPA expects (B, H, T, Dh)
        # IMPORTANT: PyTorch SDPA has had version-specific quirks around boolean masks.
        # To make behavior consistent with our manual path (where True = "blocked"),
        # we convert bool masks to additive masks (0 / -inf) before calling SDPA.
        sdpa_mask = attn_mask
        if sdpa_mask is not None and sdpa_mask.dtype == torch.bool:
            neg_inf = torch.finfo(q.dtype).min
            sdpa_mask = torch.zeros_like(sdpa_mask, dtype=q.dtype)
            sdpa_mask = sdpa_mask.masked_fill(attn_mask, neg_inf)  # type: ignore[arg-type]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=False
        )
        return out, None

    # Manual attention with full control.
    scale = dh**-0.5
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Tq, Tk)

    attn_logits = _apply_kv_padding_mask(attn_logits, key_padding_mask)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_logits = attn_logits.masked_fill(attn_mask, torch.finfo(attn_logits.dtype).min)
        else:
            attn_logits = attn_logits + attn_mask

    # Optional sparse attention.
    # - hard: scatter top-k logits into -inf tensor, then softmax (true sparsity and speed)
    # - soft: straight-through top-k (forward sparse, backward dense) to avoid k=1 gradient collapse
    if effective_topk is not None and sparse_mode == "hard":
        topk_vals, topk_idx = torch.topk(attn_logits, k=effective_topk, dim=-1, largest=True, sorted=False)
        sparse_logits = attn_logits.new_full(attn_logits.shape, torch.finfo(attn_logits.dtype).min)
        sparse_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        attn_logits_for_softmax = sparse_logits
    else:
        attn_logits_for_softmax = attn_logits

    attn_weights = _masked_softmax(attn_logits_for_softmax)

    if effective_topk is not None and sparse_mode == "soft":
        # Straight-through top-k:
        # forward: use sparse weights restricted to top-k keys
        # backward: use dense softmax gradients (prevents k=1 from killing Q/K grads)
        #
        # Build sparse weights by taking the dense probabilities on the top-k keys and renormalizing.
        # Note: top-k selection is non-differentiable; ST makes backward behave like dense attention.
        topk_idx = torch.topk(attn_logits, k=effective_topk, dim=-1, largest=True, sorted=False).indices
        topk_vals = attn_weights.gather(dim=-1, index=topk_idx)
        denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_vals = topk_vals / denom
        sparse_weights = attn_weights.new_zeros(attn_weights.shape)
        sparse_weights.scatter_(dim=-1, index=topk_idx, src=topk_vals)

        # Forward uses sparse_weights; backward uses attn_weights (dense).
        attn_weights = attn_weights + (sparse_weights - attn_weights).detach()
    if dropout_p and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    out = torch.matmul(attn_weights, v)  # (B, H, Tq, Dh)

    # For diagnostics we surface the logits actually used for the softmax in forward (hard mode uses sparse logits).
    scores = (
        AttentionScores(attn_weights=attn_weights, attn_logits=attn_logits_for_softmax)
        if return_attention
        else None
    )
    return out, scores


# ----------------------------
# Self-Attention
# ----------------------------


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional KV cache and attention score returns."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        sparse_topk: Optional[int] = None,
        sparse_topk_ratio: Optional[float] = None,
        sparse_mode: str = "hard",
        dropout: float = 0.0,
        bias: bool = True,
        use_sdpa_if_available: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_sdpa_if_available = use_sdpa_if_available
        self.sparse_topk = sparse_topk
        self.sparse_topk_ratio = sparse_topk_ratio
        self.sparse_mode = sparse_mode

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        causal: bool = False,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        cache_name: str = "self",
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[AttentionScores]]:
        """
        Args:
            x: (B, T, D)
            attn_mask: bool or additive mask (see scaled_dot_product_attention)
            key_padding_mask: (B, T) bool; True indicates padding to block as keys
            causal: if True, apply causal mask (useful for autoregressive decoding)
            cache: KVCache; if provided, will append k/v (for decoding) and use cached prefix
            return_attention: if True, returns per-head attention weights and logits
        """
        b, t, d = x.shape
        if d != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {d}")

        q = _shape_heads(self.q_proj(x), self.num_heads)  # (B, H, T, Dh)
        k_new = _shape_heads(self.k_proj(x), self.num_heads)
        v_new = _shape_heads(self.v_proj(x), self.num_heads)

        if cache is not None:
            entry = cache.get(cache_name, layer_idx)
            entry.append(k_new, v_new)
            k = entry.k
            v = entry.v
            if k is None or v is None:
                raise RuntimeError("KVCache append failed unexpectedly.")
        else:
            k, v = k_new, v_new

        t_k = k.shape[2]
        merged_mask = _merge_attention_masks(attn_mask, causal, t_q=t, t_k=t_k, device=x.device)

        out_h, scores = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=merged_mask,
            key_padding_mask=key_padding_mask,
            sparse_topk=self.sparse_topk,
            sparse_topk_ratio=self.sparse_topk_ratio,
            sparse_mode=self.sparse_mode,
            dropout_p=self.dropout if self.training else 0.0,
            return_attention=return_attention,
            use_sdpa_if_available=self.use_sdpa_if_available,
        )
        out = self.out_proj(_unshape_heads(out_h))
        return out, scores


# ----------------------------
# Cross-Attention
# ----------------------------


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional KV cache (for caching projected memory)."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        *,
        sparse_topk: Optional[int] = None,
        sparse_topk_ratio: Optional[float] = None,
        sparse_mode: str = "hard",
        dropout: float = 0.0,
        bias: bool = True,
        use_sdpa_if_available: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_sdpa_if_available = use_sdpa_if_available
        self.sparse_topk = sparse_topk
        self.sparse_topk_ratio = sparse_topk_ratio
        self.sparse_mode = sparse_mode

        self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=bias)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        context_key_padding_mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        cache_name: str = "cross",
        static_kv: bool = True,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[AttentionScores]]:
        """
        Args:
            x: (B, Tq, Dq)
            context: (B, Tk, Dc)
            attn_mask: optional mask broadcastable to (B, H, Tq, Tk) or (Tq, Tk)
            context_key_padding_mask: (B, Tk) bool; True blocks key positions
            cache: KVCache; if provided and static_kv=True, we cache projected K/V for the context
            static_kv: if True, reuse cached K/V (typical encoder-decoder usage)
            return_attention: if True, return attention weights and logits
        """
        b, t_q, d_q = x.shape
        if d_q != self.query_dim:
            raise ValueError(f"Expected query_dim={self.query_dim}, got {d_q}")

        q = _shape_heads(self.q_proj(x), self.num_heads)  # (B, H, Tq, Dh)

        k: Tensor
        v: Tensor
        if cache is not None and static_kv:
            entry = cache.get(cache_name, layer_idx)
            if entry.k is None or entry.v is None:
                k_ctx = _shape_heads(self.k_proj(context), self.num_heads)
                v_ctx = _shape_heads(self.v_proj(context), self.num_heads)
                entry.k = k_ctx
                entry.v = v_ctx
            k = entry.k
            v = entry.v
        else:
            k = _shape_heads(self.k_proj(context), self.num_heads)
            v = _shape_heads(self.v_proj(context), self.num_heads)

        out_h, scores = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=context_key_padding_mask,
            sparse_topk=self.sparse_topk,
            sparse_topk_ratio=self.sparse_topk_ratio,
            sparse_mode=self.sparse_mode,
            dropout_p=self.dropout if self.training else 0.0,
            return_attention=return_attention,
            use_sdpa_if_available=self.use_sdpa_if_available,
        )
        out = self.out_proj(_unshape_heads(out_h))
        return out, scores


# ----------------------------
# Transformer Blocks
# ----------------------------


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """A transformer block supporting:
    - self-attention (always)
    - optional cross-attention (if context is provided)
    - pre-norm residual structure
    - optional return of attention scores
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        ff_hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        use_cross_attention: bool = False,
        context_dim: Optional[int] = None,
        # Orthogonal Subspace Projection (OSP) for cross-attention outputs when Tq=2.
        # If enabled, we learn two low-rank subspaces U1/U2 (dim x r) and project each query's
        # cross-attention output with P = U U^T (implemented as low-rank projection).
        use_osp: bool = False,
        osp_rank: int = 32,
        # Where to apply OSP:
        # - "cross": project cross-attention output before the residual add (default, matches earlier behavior)
        # - "ff": project the post-FF features (after feedforward residual)
        # - "both": apply at both locations
        osp_position: str = "cross",
        sparse_topk: Optional[int] = None,
        sparse_topk_ratio: Optional[float] = None,
        sparse_mode: str = "hard",
        use_sdpa_if_available: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_cross_attention = use_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(
            embed_dim=dim,
            num_heads=num_heads,
            sparse_topk=sparse_topk,
            sparse_topk_ratio=sparse_topk_ratio,
            sparse_mode=sparse_mode,
            dropout=attention_dropout,
            use_sdpa_if_available=use_sdpa_if_available,
        )
        self.drop1 = nn.Dropout(dropout)

        if self.use_cross_attention:
            if context_dim is None:
                context_dim = dim
            self.norm2 = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                num_heads=num_heads,
                sparse_topk=sparse_topk,
                sparse_topk_ratio=sparse_topk_ratio,
                sparse_mode=sparse_mode,
                dropout=attention_dropout,
                use_sdpa_if_available=use_sdpa_if_available,
            )
            self.drop2 = nn.Dropout(dropout)
        else:
            self.norm2 = None
            self.cross_attn = None
            self.drop2 = None

        # OSP parameters (only meaningful for cross-attention blocks).
        self.use_osp = bool(use_osp) and bool(self.use_cross_attention)
        self.osp_rank = int(osp_rank)
        if self.use_osp:
            if self.osp_rank <= 0:
                raise ValueError(f"osp_rank must be > 0, got {self.osp_rank}")
            self.osp_position = str(osp_position).lower()
            if self.osp_position not in {"cross", "ff", "both"}:
                raise ValueError(
                    f"Unknown osp_position={self.osp_position!r} (expected 'cross', 'ff', or 'both')"
                )
            init_scale = 0.02
            self.osp_U1 = nn.Parameter(torch.randn(dim, self.osp_rank) * init_scale)
            self.osp_U2 = nn.Parameter(torch.randn(dim, self.osp_rank) * init_scale)
        else:
            self.osp_position = "cross"

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidden_dim, dropout=dropout, activation=activation)
        self.drop3 = nn.Dropout(dropout)

    def osp_orthogonality_loss(self) -> Optional[Tensor]:
        """Orthogonality loss between the two learned subspaces (U1, U2).

        We normalize columns and penalize cross-correlation:
            loss = mean((U1^T U2)^2)
        """
        if not getattr(self, "use_osp", False):
            return None
        u1 = F.normalize(self.osp_U1, dim=0)
        u2 = F.normalize(self.osp_U2, dim=0)
        c = u1.T @ u2  # (r, r)
        return (c ** 2).mean()

    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_key_padding_mask: Optional[Tensor] = None,
        causal: bool = False,
        cross_attn_mask: Optional[Tensor] = None,
        context_key_padding_mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, AttentionScores]]]:
        scores: Dict[str, AttentionScores] = {}

        # Self-attention
        x_norm = self.norm1(x)
        sa_out, sa_scores = self.self_attn(
            x_norm,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            causal=causal,
            cache=cache,
            layer_idx=layer_idx,
            cache_name="self",
            return_attention=return_attention,
        )
        x = x + self.drop1(sa_out)
        if return_attention and sa_scores is not None:
            scores["self"] = sa_scores

        # Cross-attention (optional)
        if self.use_cross_attention and context is not None:
            x_norm = self.norm2(x)  # type: ignore[misc]
            ca_out, ca_scores = self.cross_attn(  # type: ignore[misc]
                x_norm,
                context,
                attn_mask=cross_attn_mask,
                context_key_padding_mask=context_key_padding_mask,
                cache=cache,
                layer_idx=layer_idx,
                cache_name="cross",
                static_kv=True,
                return_attention=return_attention,
            )

            # Orthogonal Subspace Projection (OSP): only when we have exactly 2 query tokens.
            if getattr(self, "use_osp", False):
                if ca_out.shape[1] != 2:
                    raise ValueError(
                        f"OSP expects Tq=2 query tokens, got ca_out shape {tuple(ca_out.shape)}"
                    )
                u1 = F.normalize(self.osp_U1, dim=0)  # (D, r)
                u2 = F.normalize(self.osp_U2, dim=0)  # (D, r)
                # Low-rank projection: v_proj = (v @ U) @ U^T
                v0 = ca_out[:, 0, :]  # (B, D)
                v1 = ca_out[:, 1, :]  # (B, D)
                v0p = (v0 @ u1) @ u1.T
                v1p = (v1 @ u2) @ u2.T
                if self.osp_position in {"cross", "both"}:
                    ca_out = torch.stack([v0p, v1p], dim=1)  # (B, 2, D)

            x = x + self.drop2(ca_out)  # type: ignore[misc]
            if return_attention and ca_scores is not None:
                scores["cross"] = ca_scores

        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.ff(x_norm)
        x = x + self.drop3(ff_out)

        # Optional OSP after feedforward (post-FF features).
        if getattr(self, "use_osp", False) and self.osp_position in {"ff", "both"}:
            if x.shape[1] != 2:
                raise ValueError(f"OSP expects Tq=2 query tokens, got x shape {tuple(x.shape)}")
            u1 = F.normalize(self.osp_U1, dim=0)
            u2 = F.normalize(self.osp_U2, dim=0)
            v0 = x[:, 0, :]
            v1 = x[:, 1, :]
            v0p = (v0 @ u1) @ u1.T
            v1p = (v1 @ u2) @ u2.T
            x = torch.stack([v0p, v1p], dim=1)

        return x, (scores if return_attention else None)


class TransformerStack(nn.Module):
    """Multi-layer stack of TransformerBlocks."""

    def __init__(
        self,
        *,
        num_layers: int,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        use_cross_attention: bool = False,
        context_dim: Optional[int] = None,
        use_osp: bool = False,
        osp_rank: int = 32,
        osp_position: str = "cross",
        sparse_topk: Optional[int] = None,
        sparse_topk_ratio: Optional[float] = None,
        sparse_mode: str = "hard",
        use_sdpa_if_available: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                    use_cross_attention=use_cross_attention,
                    context_dim=context_dim,
                    use_osp=use_osp,
                    osp_rank=osp_rank,
                    osp_position=osp_position,
                    sparse_topk=sparse_topk,
                    sparse_topk_ratio=sparse_topk_ratio,
                    sparse_mode=sparse_mode,
                    use_sdpa_if_available=use_sdpa_if_available,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_key_padding_mask: Optional[Tensor] = None,
        causal: bool = False,
        cross_attn_mask: Optional[Tensor] = None,
        context_key_padding_mask: Optional[Tensor] = None,
        cache: Optional[KVCache] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[int, Dict[str, AttentionScores]]]]:
        """Forward through all blocks.

        Returns:
            x_out: final hidden states
            attn_by_layer: dict[layer_idx] -> {"self": AttentionScores, "cross": AttentionScores}
                          returned only if return_attention=True
        """
        attn_by_layer: Dict[int, Dict[str, AttentionScores]] = {}

        for i, layer in enumerate(self.layers):
            x, scores = layer(
                x,
                context=context,
                self_attn_mask=self_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                causal=causal,
                cross_attn_mask=cross_attn_mask,
                context_key_padding_mask=context_key_padding_mask,
                cache=cache,
                layer_idx=i,
                return_attention=return_attention,
            )
            if return_attention and scores is not None:
                attn_by_layer[i] = scores

        return x, (attn_by_layer if return_attention else None)


