"""Rolling-window metric tracker for training status reports."""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class StatusMonitor:
    """
    Accumulates recent per-step statistics and exposes smoothed (window-mean)
    summaries for stable logging.
    """

    def __init__(self, window: int = 50):
        self.loss_window: deque[float] = deque(maxlen=window)
        self.fetch_ms_window: deque[float] = deque(maxlen=window)
        self.step_ms_window: deque[float] = deque(maxlen=window)
        self.grad_norm_window: deque[float] = deque(maxlen=window)
        self.start_time: float = time.time()

    def update(self, *, loss: float, fetch_ms: float, step_ms: float, grad_norm: float) -> None:
        self.loss_window.append(float(loss))
        self.fetch_ms_window.append(float(fetch_ms))
        self.step_ms_window.append(float(step_ms))
        self.grad_norm_window.append(float(grad_norm))

    def summary(self) -> dict[str, float]:
        return {
            "train/loss_window": float(np.mean(self.loss_window)) if self.loss_window else float("nan"),
            "train/fetch_ms_window": float(np.mean(self.fetch_ms_window)) if self.fetch_ms_window else float("nan"),
            "train/step_ms_window": float(np.mean(self.step_ms_window)) if self.step_ms_window else float("nan"),
            "train/grad_norm_window": float(np.mean(self.grad_norm_window)) if self.grad_norm_window else float("nan"),
            "train/steps_per_sec_window": 1000.0 / max(1e-9, float(np.mean(self.step_ms_window))) if self.step_ms_window else 0.0,
            "train/elapsed_sec": float(time.time() - self.start_time),
        }
