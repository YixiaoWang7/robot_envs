"""Reusable training utilities: device transfer and CUDA prefetching."""

from __future__ import annotations

from typing import Any

import torch

from policies.training.robot_processor import RobotProcessor


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested dict/list/tuple to ``device``."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


class CUDAPrefetcher:
    """Prefetch + preprocess one batch ahead (optional CUDA stream)."""

    def __init__(self, dataloader, processor: RobotProcessor, device: torch.device):
        self._it = iter(dataloader)
        self._processor = processor
        self._device = device
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._next = None
        self._prefetch()

    def _prefetch(self):
        try:
            batch = next(self._it)
        except StopIteration:
            self._next = None
            return

        if self._stream is None:
            batch = self._processor(batch)
            batch = move_to_device(batch, self._device)
            self._next = batch
            return

        with torch.cuda.stream(self._stream):
            batch = self._processor(batch)
            batch = move_to_device(batch, self._device)
        self._next = batch

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next
        self._prefetch()
        return batch
