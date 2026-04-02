from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LearnableQueryTokens(nn.Module):
    """Produce learnable query tokens for attention-based fusion.

    This module is intentionally small: it only returns a tensor of query tokens with
    shape (B, Tq, D), suitable for passing as `query_tokens=` into
    `SelfAttentionPooling` / `CrossAttentionPooling`.

    Two modes are supported:
    - Global learnable queries (shared across all samples).
    - Task-indexed learnable queries (one query bank per task id).
    """

    def __init__(
        self,
        *,
        num_query_tokens: int,
        dim: int,
        num_tasks: int | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.dim = int(dim)
        if self.num_query_tokens <= 0:
            raise ValueError("num_query_tokens must be >= 1")
        if self.dim <= 0:
            raise ValueError("dim must be >= 1")

        self.num_tasks = None if num_tasks is None else int(num_tasks)
        if self.num_tasks is not None and self.num_tasks <= 0:
            raise ValueError("num_tasks must be >= 1 when provided")

        shape = (
            (self.num_query_tokens, self.dim)
            if self.num_tasks is None
            else (self.num_tasks, self.num_query_tokens, self.dim)
        )
        self.query_tokens = nn.Parameter(torch.randn(*shape) * float(init_std))

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        task_ids: Tensor | None = None,
    ) -> Tensor:
        """Return query tokens of shape (B, Tq, D).

        Args:
            batch_size: batch size B.
            device: optional device override (defaults to parameter device).
            dtype: optional dtype override (defaults to parameter dtype).
            task_ids: optional task indices of shape (B,) used when `num_tasks` was set.
        """
        B = int(batch_size)
        if B <= 0:
            raise ValueError("batch_size must be >= 1")

        q = self.query_tokens
        if device is None:
            device = q.device
        if dtype is None:
            dtype = q.dtype
        q = q.to(device=device, dtype=dtype)

        if self.num_tasks is None:
            if task_ids is not None:
                raise ValueError("task_ids provided but this module was created without num_tasks")
            return q.unsqueeze(0).expand(B, q.shape[0], q.shape[1])  # (B, Tq, D)

        # Task-indexed query bank: select per-sample tokens.
        if task_ids is None:
            raise ValueError("task_ids is required when this module was created with num_tasks")
        if task_ids.ndim != 1 or task_ids.shape[0] != B:
            raise ValueError(f"task_ids must have shape (B,), got {tuple(task_ids.shape)}")
        task_ids = task_ids.to(device=device, dtype=torch.long)
        # (B, Tq, D)
        return q.index_select(dim=0, index=task_ids)

