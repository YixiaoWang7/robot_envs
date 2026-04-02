#!/usr/bin/env python

"""Attention-based fusion/pooling modules.

These modules fuse information from token sequences or spatial feature maps using
*externally provided* query tokens and a small Transformer stack.

Query token generation/conditioning (learnable queries, language prompts, etc.) lives in
`policies.modules.task`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from policies.modules.transformer.transformer import TransformerStack


def _require_query_tokens(
    *,
    batch_size: int,
    device: torch.device,
    query_tokens: Tensor | None,
) -> Tensor:
    """Return query tokens of shape (B, Tq, D) on `device`.
    """
    if query_tokens is None:
        raise ValueError(
            "query_tokens is required. Provide it explicitly, e.g. from `policies.modules.task` "
            "(learnable queries, language prompts, etc.)."
        )

    q = query_tokens.to(device=device)
    if q.ndim == 2:
        q = q.unsqueeze(0).expand(batch_size, q.shape[0], q.shape[1])
    if q.ndim != 3 or q.shape[0] != batch_size:
        raise ValueError(f"query_tokens must have shape (Tq, D) or (B, Tq, D), got {tuple(q.shape)}")
    return q


class SelfAttentionPooling(nn.Module):
    """Self-attention based pooling for image features.
    
    Similar to TokenizerSelfAttentionGlobalConditioner from flow_state but adapted for image features.
    Applies self-attention over the concatenation of (query tokens, spatial tokens).
    """

    def __init__(
        self,
        input_shape,
        embed_dim=None,
        out_dim=None,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self.spatial_size = self._in_h * self._in_w

        self.embed_dim = embed_dim if embed_dim is not None else self._in_c
        self.out_dim = out_dim if out_dim is not None else self.embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(self._in_c, self.embed_dim)
        self.input_norm = nn.LayerNorm(self.embed_dim)
        
        # Self-attention encoder
        self.encoder = TransformerStack(
            num_layers=num_layers,
            dim=self.embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_dim,
            dropout=dropout,
            attention_dropout=dropout,
            activation="gelu",
            use_cross_attention=False,
        )
        
        # Output normalization and projection
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.out_dim)

    def forward(
        self,
        features: Tensor,
        *,
        query_tokens: Tensor,
        return_intermediate: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            features: (B, C, H, W) spatial feature map, or (B, Tk, C) token sequence
            query_tokens: query tokens, shape (Tq, D) or (B, Tq, D)
        Returns:
            (B, Tq, out_dim) pooled features for each query token
        """
        # Accept either feature maps (B, C, H, W) or token sequences (B, Tk, C).
        if features.ndim == 4:
            B, C, H, W = features.shape
            assert C == self._in_c and H == self._in_h and W == self._in_w
            tokens = features.flatten(2).transpose(1, 2)  # (B, N, C)
        elif features.ndim == 3:
            B, Tk, C = features.shape
            assert C == self._in_c
            tokens = features
        else:
            raise ValueError(f"Expected features to have ndim=3 or 4, got {features.ndim}.")

        tokens = self.input_norm(self.input_proj(tokens))  # (B, Tk, embed_dim)

        q_tokens = _require_query_tokens(
            batch_size=B,
            device=features.device,
            query_tokens=query_tokens,
        )  # (B, Tq, D)
        
        # Concatenate query tokens with spatial tokens
        all_tokens = torch.cat([q_tokens, tokens], dim=1)  # (B, Tq+Tk, embed_dim)
        
        # Apply self-attention
        encoded_tokens, _ = self.encoder(all_tokens, return_attention=False)
        
        # Extract query token outputs
        Tq = q_tokens.shape[1]
        token_embed = self.output_norm(encoded_tokens[:, :Tq, :])  # (B, Tq, embed_dim)
        output_tokens = self.output_proj(token_embed)  # (B, Tq, out_dim)

        if not return_intermediate:
            return output_tokens

        intermediate: dict[str, Tensor] = {"query_token_embed": token_embed}
        return output_tokens, intermediate


class CrossAttentionPooling(nn.Module):
    """Cross-attention based pooling for image features.
    
    Similar to TokenizerCrossAttentionGlobalConditioner from flow_state but adapted for image features.
    Uses externally provided query tokens that cross-attend to spatial feature tokens.
    """

    def __init__(
        self,
        input_shape,
        embed_dim=None,
        out_dim=None,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        *,
        mask_type: str = "none",
    ):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self.spatial_size = self._in_h * self._in_w

        self.embed_dim = embed_dim if embed_dim is not None else self._in_c
        self.out_dim = out_dim if out_dim is not None else self.embed_dim
        
        # Input projection for spatial tokens (keys/values)
        self.input_proj = nn.Linear(self._in_c, self.embed_dim)
        self.input_norm = nn.LayerNorm(self.embed_dim)
        
        # Cross-attention decoder
        self.decoder = TransformerStack(
            num_layers=num_layers,
            dim=self.embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_dim,
            dropout=dropout,
            attention_dropout=dropout,
            activation="gelu",
            use_cross_attention=True,
            context_dim=self.embed_dim,
        )
        
        # Output normalization and projection
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.out_dim)
        self.mask_type = str(mask_type)

    def get_mask(self, device: torch.device, *, num_queries: int) -> Tensor | None:
        """Query-side self-attention mask (no memory mask for image tokens)."""
        if self.mask_type == "none":
            return None
        elif self.mask_type == "query_separate":
            # Block attention between different query positions (each query attends only to itself).
            query_mask = torch.ones(num_queries, num_queries, dtype=torch.bool, device=device)
            query_mask.fill_diagonal_(False)
            return query_mask
        else:
            raise ValueError(
                f"Unknown mask_type={self.mask_type!r} (expected 'none', 'query_separate')."
            )

    def forward(
        self,
        features: Tensor,
        *,
        query_tokens: Tensor,
        return_intermediate: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            features: (B, C, H, W) spatial feature map, or (B, Tk, C) token sequence
            query_tokens: query tokens, shape (Tq, D) or (B, Tq, D)
        Returns:
            (B, Tq, out_dim) pooled features for each query token
        """
        # Accept either feature maps (B, C, H, W) or token sequences (B, Tk, C).
        if features.ndim == 4:
            B, C, H, W = features.shape
            assert C == self._in_c and H == self._in_h and W == self._in_w
            tokens = features.flatten(2).transpose(1, 2)  # (B, N, C)
        elif features.ndim == 3:
            B, Tk, C = features.shape
            assert C == self._in_c
            tokens = features
        else:
            raise ValueError(f"Expected features to have ndim=3 or 4, got {features.ndim}.")

        spatial_tokens = self.input_norm(self.input_proj(tokens))  # (B, Tk, embed_dim)

        q_tokens = _require_query_tokens(
            batch_size=B,
            device=features.device,
            query_tokens=query_tokens,
        )  # (B, Tq, D)

        query_mask = self.get_mask(features.device, num_queries=q_tokens.shape[1])

        # Apply cross-attention: queries attend to spatial tokens (no memory mask available for image tokens).
        decoded_tokens, attn_by_layer = self.decoder(
            q_tokens,
            context=spatial_tokens,
            self_attn_mask=query_mask,
            cross_attn_mask=None,
            return_attention=return_intermediate,
        )  # (B, Tq, embed_dim)
        
        # Normalize and project
        token_embed = self.output_norm(decoded_tokens)  # (B, Tq, embed_dim)
        output_tokens = self.output_proj(token_embed)  # (B, Tq, out_dim)

        if not return_intermediate:
            return output_tokens

        # Extract per-query attention distributions over spatial tokens for overlap loss/metrics.
        query_attn: Tensor | None = None

        if attn_by_layer is not None:
            cross_by_layer = []
            for _, scores_dict in attn_by_layer.items():
                if "cross" not in scores_dict:
                    continue
                cross_by_layer.append(scores_dict["cross"].attn_weights)  # (B, H, Tq, Tk)
            if len(cross_by_layer) > 0:
                attn_all = torch.stack(cross_by_layer, dim=0).mean(dim=0)  # (B, H, Tq, Tk)
                attn_mean = attn_all.mean(dim=1)  # (B, Tq, Tk)
                query_attn = attn_mean

        intermediate: dict[str, Tensor] = {}
        if query_attn is not None:
            intermediate["query_attn"] = query_attn  # (B, Tq, Tk)
        # Also expose the query embeddings (pre-projection).
        intermediate["query_token_embed"] = token_embed  # (B, Tq, D)
        return output_tokens, intermediate