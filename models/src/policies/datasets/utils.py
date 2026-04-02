from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


def now_s() -> float:
    return time.perf_counter()


@dataclass
class TimingStats:
    """
    Lightweight timing + counters container for dataset diagnostics.

    Use `track("key")` as a context manager, or `add("key", dt)`.
    """

    times_s: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, key: str, dt_s: float, *, count: int = 1) -> None:
        k = str(key)
        self.times_s[k] = float(self.times_s.get(k, 0.0)) + float(dt_s)
        self.counts[k] = int(self.counts.get(k, 0)) + int(count)

    def merge(self, other: "TimingStats") -> None:
        for k, v in other.times_s.items():
            self.times_s[str(k)] = float(self.times_s.get(str(k), 0.0)) + float(v)
        for k, v in other.counts.items():
            self.counts[str(k)] = int(self.counts.get(str(k), 0)) + int(v)

    @contextmanager
    def track(self, key: str) -> Iterator[None]:
        t0 = now_s()
        try:
            yield
        finally:
            self.add(key, now_s() - t0)

