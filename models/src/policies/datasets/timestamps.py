from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Alignment:
    train_timestamps: Array  # (T_train,) float64
    indices: Dict[str, Array]  # modality -> (T_train,) int64


class TimestampSampler:
    def build_train_grid(self, modality_timestamps: Mapping[str, Array], *, train_hz: float) -> Optional[Array]:
        if train_hz <= 0:
            raise ValueError("train_hz must be > 0")
        starts = []
        ends = []
        for _name, ts in modality_timestamps.items():
            ts = np.asarray(ts, dtype=np.float64)
            if ts.size <= 0:
                return None
            starts.append(float(ts[0]))
            ends.append(float(ts[-1]))
        t0 = max(starts)
        t1 = min(ends)
        if (not np.isfinite(t0)) or (not np.isfinite(t1)) or (t1 <= t0):
            return None
        dt = 1.0 / float(train_hz)
        grid = np.arange(t0, t1 + 1e-12, dt, dtype=np.float64)
        if grid.size <= 0:
            return None
        return grid

    def align_latest(self, src_timestamps: Array, train_timestamps: Array) -> Array:
        src_ts = np.asarray(src_timestamps, dtype=np.float64)
        tr = np.asarray(train_timestamps, dtype=np.float64)
        idx = np.searchsorted(src_ts, tr, side="right") - 1
        if idx.dtype != np.int64:
            idx = idx.astype(np.int64, copy=False)
        np.maximum(idx, 0, out=idx)
        last = int(src_ts.shape[0] - 1)
        if last >= 0:
            np.minimum(idx, last, out=idx)
        return idx

    def precompute(self, modality_timestamps: Mapping[str, Array], *, train_hz: float) -> Optional[Alignment]:
        grid = self.build_train_grid(modality_timestamps, train_hz=float(train_hz))
        if grid is None:
            return None
        out: Dict[str, Array] = {}
        for name, ts in modality_timestamps.items():
            out[str(name)] = self.align_latest(ts, grid)
        return Alignment(train_timestamps=grid, indices=out)


class LatestTimestampSampler(TimestampSampler):
    pass

