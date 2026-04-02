from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from policies.datasets.timestamps import Alignment


CACHE_VERSION = 1


def _camera_key(cameras: Tuple[str, ...]) -> str:
    """
    Stable short key for the set of cameras used by the cache.

    We canonicalize by sorting, because alignment indices are independent of camera order.
    """
    cams = ",".join(sorted(str(c) for c in cameras))
    h = hashlib.sha1(cams.encode("utf-8")).hexdigest()  # short+stable; not for security
    return h[:10]


@dataclass(frozen=True)
class CacheSpec:
    cache_dir: Path
    task_slug: str
    shard_id: int
    train_hz: float
    sampler_name: str
    cameras: Tuple[str, ...]

    def relpath(self) -> Path:
        hz = f"{self.train_hz:.6f}".rstrip("0").rstrip(".")
        return (
            Path("align_cache")
            / self.task_slug
            / f"{self.sampler_name}"
            / f"train_hz_{hz}"
            / f"cams_{_camera_key(self.cameras)}"
            / f"shard_{self.shard_id:03d}.npz"
        )

    def path(self) -> Path:
        return self.cache_dir / self.relpath()

    def legacy_relpath(self) -> Path:
        """Backward-compatible path used before camera-keyed subdirs were added."""
        hz = f"{self.train_hz:.6f}".rstrip("0").rstrip(".")
        return (
            Path("align_cache")
            / self.task_slug
            / f"{self.sampler_name}"
            / f"train_hz_{hz}"
            / f"shard_{self.shard_id:03d}.npz"
        )

    def legacy_path(self) -> Path:
        return self.cache_dir / self.legacy_relpath()


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if dst.exists():
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return
    tmp.replace(dst)


def load_shard_cache(spec: CacheSpec) -> Optional[Dict[str, Alignment]]:
    # New path (camera-keyed) first, then legacy.
    paths = [spec.path(), spec.legacy_path()]
    npz = None
    for path in paths:
        if not path.exists():
            continue
        try:
            npz = np.load(str(path), allow_pickle=False)
            break
        except Exception:
            npz = None
            continue
    if npz is None:
        return None

    try:
        version = int(npz["__version__"])
        if version != CACHE_VERSION:
            return None
        meta = json.loads(str(npz["__meta_json__"]))
        if float(meta.get("train_hz")) != float(spec.train_hz):
            return None
        if str(meta.get("sampler_name")) != str(spec.sampler_name):
            return None
        if tuple(meta.get("cameras", [])) != tuple(sorted(str(c) for c in spec.cameras)):
            return None
        demo_keys = [str(x) for x in npz["demo_keys"].tolist()]
        t0 = np.asarray(npz["train_t0"], dtype=np.float64)
        dt = np.asarray(npz["train_dt"], dtype=np.float64)
        n = np.asarray(npz["train_len"], dtype=np.int64)
    except Exception:
        return None

    out: Dict[str, Alignment] = {}
    for i, dk in enumerate(demo_keys):
        train_ts = (float(t0[i]) + float(dt[i]) * np.arange(int(n[i]), dtype=np.float64))
        indices: Dict[str, np.ndarray] = {}
        indices["actions"] = np.asarray(npz[f"actions_idx_{i}"], dtype=np.int64)
        indices["robot_state"] = np.asarray(npz[f"robot_idx_{i}"], dtype=np.int64)
        indices["env_state"] = np.asarray(npz[f"env_idx_{i}"], dtype=np.int64)
        for cam in spec.cameras:
            safe = cam.replace("/", "_")
            indices[f"images/{cam}"] = np.asarray(npz[f"img_{safe}_idx_{i}"], dtype=np.int64)
        out[dk] = Alignment(train_timestamps=train_ts, indices=indices)
    return out


def save_shard_cache(spec: CacheSpec, alignments: Mapping[str, Alignment]) -> None:
    demo_keys = sorted(alignments.keys())
    meta = {
        "version": CACHE_VERSION,
        "train_hz": float(spec.train_hz),
        "sampler_name": str(spec.sampler_name),
        # Canonicalize to avoid accidental cache misses due to camera order.
        "cameras": sorted(str(c) for c in spec.cameras),
    }

    payload: Dict[str, np.ndarray] = {}
    payload["__version__"] = np.asarray([CACHE_VERSION], dtype=np.int64)
    payload["__meta_json__"] = np.asarray(json.dumps(meta), dtype=np.str_)
    payload["demo_keys"] = np.asarray(demo_keys, dtype=np.str_)

    t0 = np.zeros((len(demo_keys),), dtype=np.float64)
    dt = np.zeros((len(demo_keys),), dtype=np.float64)
    n = np.zeros((len(demo_keys),), dtype=np.int64)

    for i, dk in enumerate(demo_keys):
        al = alignments[dk]
        ts = np.asarray(al.train_timestamps, dtype=np.float64)
        if ts.size <= 0:
            raise ValueError(f"Empty train_timestamps for demo_key={dk}")
        t0[i] = float(ts[0])
        dt[i] = float(ts[1] - ts[0]) if ts.size >= 2 else 0.0
        n[i] = int(ts.size)

        payload[f"actions_idx_{i}"] = np.asarray(al.indices["actions"], dtype=np.int32)
        payload[f"robot_idx_{i}"] = np.asarray(al.indices["robot_state"], dtype=np.int32)
        payload[f"env_idx_{i}"] = np.asarray(al.indices["env_state"], dtype=np.int32)
        for cam in spec.cameras:
            safe = cam.replace("/", "_")
            payload[f"img_{safe}_idx_{i}"] = np.asarray(al.indices[f"images/{cam}"], dtype=np.int32)

    payload["train_t0"] = t0
    payload["train_dt"] = dt
    payload["train_len"] = n

    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    _atomic_write_bytes(spec.path(), buf.getvalue())

