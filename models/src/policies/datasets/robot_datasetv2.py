#!/usr/bin/env python
"""
Dataset V2: k-pass, m-window, decode-on-demand pipeline.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

from policies.datasets.schema import FeatureConfig, FeatureDef
from policies.datasets.utils import now_s


if not hasattr(np.random, "default_rng"):
    raise ImportError("RobotDataset requires NumPy >= 1.17.")
_np_default_rng = np.random.default_rng


def _stable_int_seed(*parts: object, bits: int = 31) -> int:
    """
    Builds a deterministic positive integer seed from arbitrary parts.

    Notes:
      - Uses MD5 for speed/stability (not cryptographic purposes).
      - Returned seed is in [0, 2**bits).
    """
    s = "|".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    raw = int(h[:8], 16)  # 32-bit chunk
    mask = (1 << int(bits)) - 1
    return int(raw & mask)


def _k_pass_slice_bounds(*, total: int, pass_idx: int, k_passes: int) -> tuple[int, int]:
    """
    Computes contiguous slice bounds for the `pass_idx`-th chunk when splitting `total`
    items into `k_passes` nearly-equal parts (same size pattern as `np.array_split`).

    Returns:
        (start, end): bounds such that each pass covers `order[start:end]`.
    """
    total = int(total)
    pass_idx = int(pass_idx)
    k_passes = int(k_passes)
    if total <= 0 or k_passes <= 1:
        return (0, total)
    base = total // k_passes
    rem = total % k_passes
    start = (pass_idx * base) + min(pass_idx, rem)
    size = base + (1 if pass_idx < rem else 0)
    end = start + size
    return (int(start), int(end))


@dataclass(frozen=True)
class ShardMeta:
    """
    Immutable metadata for one logical dataset shard.

    Attributes:
        task (str): Human-readable task name from task manifest.
        task_slug (str): Stable task identifier used in file paths.
        shard_id (int): Numeric shard identifier within the task.
        hdf5_rel (str): Relative HDF5 path under dataset root.
        demos (list[dict]): Per-demo metadata entries from manifest.
        videos (dict): Camera video metadata (`mp4` + `sidecar`) per camera.
        video_pack (Optional[str]): Video packing mode (expected `per_shard`).
    """
    task: str
    task_slug: str
    shard_id: int
    hdf5_rel: str
    demos: list[dict]
    videos: dict
    video_pack: Optional[str] = None


@dataclass
class DemoData:
    """
    In-memory numeric payload and indexing info for one demo.

    Attributes:
        demo_id (int): Local demo id inside a loaded shard.
        task (str): Demo task label.
        task_slug (str): Demo task slug.
        shard_id (int): Source shard id.
        demo_key (str): Demo key inside HDF5.
        arrays (dict[str, np.ndarray]): Numeric arrays keyed by modality name.
        image_base (dict[str, int]): Base frame offset per camera in shard MP4.
    """
    demo_id: int
    task: str
    task_slug: str
    shard_id: int
    demo_key: str
    arrays: dict[str, np.ndarray]
    image_base: dict[str, int]


@dataclass
class CachedShard:
    """
    Worker-local shard cache unit used by V2 sampling.

    Attributes:
        meta (ShardMeta): Source shard metadata.
        demos_by_id (list[DemoData]): Demo payloads addressable by local demo id.
        all_demo_ids (np.ndarray): Flattened demo-id index for all valid starts.
            Shape: (num_total_starts,)
        all_starts (np.ndarray): Flattened start index per sample window.
            Shape: (num_total_starts,)
    """
    meta: ShardMeta
    demos_by_id: list[DemoData]
    all_demo_ids: np.ndarray
    all_starts: np.ndarray


@dataclass
class DatasetWorkerProfiler:
    """
    Lightweight per-worker timing/counter profiler for data iteration.

    Attributes:
        enabled (bool): Enables or disables all tracking.
        worker_id (int): DataLoader worker id for log context.
        every_samples (int): Report cadence in yielded samples.
        start_s (float): Wall clock start time in seconds.
        yielded (int): Number of yielded samples so far.
        last_report_at (int): Yield count at the previous report.
        times_s (dict[str, float]): Accumulated timing buckets in seconds.
        counts (dict[str, int]): Accumulated decode/planning counters.
    """
    enabled: bool
    worker_id: int
    every_samples: int
    start_s: float
    yielded: int = 0
    last_report_at: int = 0
    times_s: dict[str, float] = None  # type: ignore[assignment]
    counts: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initializes timer and counter buckets after dataclass construction."""
        self.times_s = {"shard_load_s": 0.0, "plan_s": 0.0, "decode_s": 0.0, "assemble_s": 0.0}
        self.counts = {"decode_calls": 0, "requested_frames": 0, "decoded_unique_frames": 0}

    def add_time(self, key: str, dt_s: float) -> None:
        """
        Adds elapsed seconds to a named timing bucket.

        Args:
            key (str): Metric key in `times_s`.
            dt_s (float): Duration in seconds to accumulate.
        """
        if self.enabled:
            self.times_s[key] = float(self.times_s.get(key, 0.0)) + float(dt_s)

    def add_count(self, key: str, inc: int = 1) -> None:
        """
        Increments a named counter bucket.

        Args:
            key (str): Metric key in `counts`.
            inc (int): Increment value.
        """
        if self.enabled:
            self.counts[key] = int(self.counts.get(key, 0)) + int(inc)

    def sample_yielded(self) -> None:
        """Registers one produced sample."""
        if self.enabled:
            self.yielded += 1

    def _avg_ms(self, key: str) -> float:
        """
        Computes average milliseconds per yielded sample for a bucket.

        Args:
            key (str): Timing key from `times_s`.

        Returns:
            float: Average milliseconds per sample.
        """
        return (float(self.times_s.get(key, 0.0)) / max(1, int(self.yielded))) * 1000.0

    def maybe_report(self, *, force: bool = False) -> None:
        """
        Prints a periodic aggregated profile line.

        Args:
            force (bool): If True, prints regardless of cadence threshold.
        """
        if not self.enabled:
            return
        n = int(self.yielded)
        if not force and (n - int(self.last_report_at)) < int(self.every_samples):
            return
        elapsed = max(1e-9, now_s() - float(self.start_s))
        requested = int(self.counts.get("requested_frames", 0))
        unique = int(self.counts.get("decoded_unique_frames", 0))
        reuse = float(requested) / max(1, unique)
        print(
            (
                f"[datasetv2 profile worker={self.worker_id}] "
                f"samples={n} elapsed_s={elapsed:.1f} "
                f"avg_shard_load_ms={self._avg_ms('shard_load_s'):.3f} "
                f"avg_plan_ms={self._avg_ms('plan_s'):.3f} "
                f"avg_decode_ms={self._avg_ms('decode_s'):.3f} "
                f"avg_assemble_ms={self._avg_ms('assemble_s'):.3f} "
                f"decode_calls={int(self.counts.get('decode_calls', 0))} "
                f"requested_frames={requested} unique_frames={unique} frame_reuse={reuse:.2f}x"
            ),
            flush=True,
        )
        self.last_report_at = n


def _load_json(path: Path) -> dict:
    """
    Loads a UTF-8 JSON file.

    Args:
        path (Path): JSON path.

    Returns:
        dict: Parsed JSON object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_split_score(*, split_seed: int, task_slug: str, shard_id: int, demo_key: str) -> float:
    """
    Computes deterministic score in [0, 1) for train/val splitting.

    Args:
        split_seed (int): Split seed.
        task_slug (str): Task identifier.
        shard_id (int): Shard identifier.
        demo_key (str): Demo identifier.

    Returns:
        float: Uniform-like deterministic score in [0, 1).
    """
    s = f"{int(split_seed)}|{task_slug}|{int(shard_id)}|{demo_key}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return float(int(h[:8], 16)) / float(2**32)


def load_shard_metas_from_manifests(data_dir: str) -> list[ShardMeta]:
    """
    Discovers shard metadata from dataset/task manifest files.

    Args:
        data_dir (str): Dataset root directory.

    Returns:
        list[ShardMeta]: Flattened shard metadata across tasks.
    """
    root = Path(data_dir)
    ds_path = root / "dataset_manifest.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"Expected dataset_manifest.json at: {ds_path}")
    ds = _load_json(ds_path)
    tasks = ds.get("tasks", [])
    if not tasks:
        raise ValueError(f"No tasks found in {ds_path}")
    out: list[ShardMeta] = []
    for t in tasks:
        task_slug = str(t["task_slug"])
        tm = _load_json(root / str(t["manifest"]))
        task_name = str(tm.get("task", task_slug))
        video_pack = tm.get("video_pack")
        for sh in tm.get("shards", []):
            out.append(
                ShardMeta(
                    task=task_name,
                    task_slug=task_slug,
                    shard_id=int(sh["shard_id"]),
                    hdf5_rel=f"{task_slug}/{sh['hdf5']}",
                    demos=list(sh.get("demos", [])),
                    videos=dict(sh.get("videos", {})),
                    video_pack=None if video_pack is None else str(video_pack),
                )
            )
    if not out:
        raise ValueError(f"No shards discovered under {data_dir}")
    return out


def split_shard_metas(
    metas: list[ShardMeta],
    *,
    split: Literal["train", "val", "all"] = "train",
    val_fraction: float = 0.05,
    split_seed: int = 0,
) -> list[ShardMeta]:
    """
    Filters demos by deterministic split and returns updated shard metas.

    Args:
        metas (list[ShardMeta]): Input shard metadata list.
        split (Literal["train", "val", "all"]): Target split.
        val_fraction (float): Validation fraction.
        split_seed (int): Seed for deterministic split scoring.

    Returns:
        list[ShardMeta]: Shards containing only demos assigned to the split.
    """
    split = str(split)
    if split not in ("train", "val", "all"):
        raise ValueError(f"split must be one of {{'train','val','all'}}, got {split!r}")
    if not (0.0 <= float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in [0,1)")
    if split == "all" or float(val_fraction) == 0.0:
        return metas
    out: list[ShardMeta] = []
    for m in metas:
        kept: list[dict] = []
        for d in m.demos:
            dk = str(d.get("demo_key", ""))
            if not dk:
                continue
            score = _stable_split_score(
                split_seed=int(split_seed), task_slug=str(m.task_slug), shard_id=int(m.shard_id), demo_key=dk
            )
            is_val = score < float(val_fraction)
            keep = is_val if split == "val" else (not is_val)
            if keep:
                kept.append(d)
        if kept:
            out.append(
                ShardMeta(
                    task=m.task,
                    task_slug=m.task_slug,
                    shard_id=m.shard_id,
                    hdf5_rel=m.hdf5_rel,
                    demos=kept,
                    videos=m.videos,
                    video_pack=m.video_pack,
                )
            )
    if not out:
        raise ValueError(f"No demos left after split={split!r} val_fraction={val_fraction}")
    return out


def _load_window_data(
    data_dir: "Path",
    metas: "list[ShardMeta]",
    cameras: "list[str]",
    feature_config: "FeatureConfig",
    global_seed: int,
    worker_id: int,
    epoch: int,
    k_passes: int,
) -> "tuple[list[CachedShard], list[np.ndarray]]":
    """
    Loads all shards for one window into RAM and pre-computes per-shard sample
    permutations.  Designed to run in a background thread so the next window's
    data is ready before it is needed.

    Returns:
        tuple[list[CachedShard], list[np.ndarray]]:
            (window_shards, window_orders) ready to be consumed by the k-pass loop.
    """
    window_shards: list[CachedShard] = []
    window_orders: list[np.ndarray] = []
    for meta in metas:
        window_shards.append(
            _load_shard_into_ram(data_dir, meta, cameras=cameras, feature_config=feature_config)
        )
        shard_seed = _stable_int_seed(
            "robot_datasetv2",
            global_seed,
            worker_id,
            epoch,
            meta.task_slug,
            meta.shard_id,
            bits=31,
        )
        total = int(window_shards[-1].all_starts.shape[0])
        idx_dtype = np.int32 if total <= np.iinfo(np.int32).max else np.int64
        if total <= 0:
            window_orders.append(np.empty((0,), dtype=idx_dtype))
        elif int(k_passes) <= 1:
            window_orders.append(np.arange(total, dtype=idx_dtype))
        else:
            np_rng = (
                _np_default_rng(int(shard_seed))
                if _np_default_rng is not None
                else np.random.RandomState(int(shard_seed))  # type: ignore[assignment]
            )
            window_orders.append(np_rng.permutation(total).astype(idx_dtype, copy=False))  # type: ignore[attr-defined]
    return window_shards, window_orders


def _h5_has(demo: h5py.Group, path: str) -> bool:
    """
    Checks if an HDF5 path exists in a demo group.

    Args:
        demo (h5py.Group): Demo group under `data/<demo_key>`.
        path (str): Relative dataset path.

    Returns:
        bool: True if the path exists, else False.
    """
    try:
        demo[path]
        return True
    except Exception:
        return False


def _h5_read_array(demo: h5py.Group, path: str, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Reads an HDF5 dataset into a NumPy array.

    Args:
        demo (h5py.Group): Demo group under `data/<demo_key>`.
        path (str): Relative dataset path.
        dtype (Optional[np.dtype]): Optional destination dtype.

    Returns:
        np.ndarray: Loaded array.
    """
    arr = demo[path][()]
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def _decode_mp4_selected_frames(
    path: Path,
    frame_indices: np.ndarray,
    cap_cache: Optional[Dict[str, "cv2.VideoCapture"]] = None,
) -> np.ndarray:
    """
    Decodes specific absolute frame indices from one MP4.

    Args:
        path (Path): MP4 path.
        frame_indices (np.ndarray): Absolute frame indices.
            Shape: (num_requested_frames,)
        cap_cache (Optional[Dict[str, cv2.VideoCapture]]): Per-worker cache of open
            VideoCapture handles keyed by path string.  When provided, VideoCapture
            objects are reused across calls (avoids re-opening and re-probing the MP4
            on every block decode, which is 20–100 ms per open on typical SSDs).
            The caller owns the cache and is responsible for releasing handles when done.

    Returns:
        np.ndarray: Decoded frames ordered as `frame_indices`.
            Shape: (num_requested_frames, height, width, 3)
    """
    idx = np.asarray(frame_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("frame_indices must be 1-D")
    if idx.size == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    if int(idx.min()) < 0:
        raise ValueError("frame_indices must be non-negative")

    path_str = str(path)
    if cap_cache is not None:
        cap = cap_cache.get(path_str)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(path_str)
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture could not open: {path}")
            cap_cache[path_str] = cap
    else:
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture could not open: {path}")
    uniq = np.unique(idx)
    uniq.sort()
    decoded: dict[int, np.ndarray] = {}

    def _release() -> None:
        if cap_cache is None:
            cap.release()

    # Heuristic: when requested indices are relatively dense, do one seek then read sequentially.
    # This avoids N independent seeks in OpenCV, which is often a major bottleneck.
    dense_factor = 4  # allow decoding a span up to ~4x requested frames
    span = int(uniq[-1] - uniq[0] + 1)
    is_dense = (uniq.size > 1) and (span <= (dense_factor * int(uniq.size)))

    if is_dense:
        first = int(uniq[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        cur = first
        for target in uniq.tolist():
            target_i = int(target)
            while cur < target_i:
                ok, _ = cap.read()
                if not ok:
                    _release()
                    raise RuntimeError(f"Failed to decode frame index={int(cur)} from: {path}")
                cur += 1
            ok, bgr = cap.read()
            if not ok:
                _release()
                raise RuntimeError(f"Failed to decode frame index={int(target_i)} from: {path}")
            decoded[target_i] = bgr.astype(np.uint8, copy=False)
            cur += 1
    else:
        for target in uniq.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, bgr = cap.read()
            if not ok:
                _release()
                raise RuntimeError(f"Failed to decode frame index={int(target)} from: {path}")
            decoded[int(target)] = bgr.astype(np.uint8, copy=False)
    _release()
    return np.stack([decoded[int(i)] for i in idx.tolist()], axis=0)


def _try_contiguous_slice(idx: np.ndarray) -> Optional[tuple[int, int]]:
    """
    Detects whether `idx` forms a contiguous range.

    Args:
        idx (np.ndarray): Candidate 1-D indices.
            Shape: (num_indices,)

    Returns:
        Optional[tuple[int, int]]: `(start, end)` if contiguous else None.
    """
    a = np.asarray(idx)
    if a.ndim != 1 or a.size <= 0:
        return None
    if a.size == 1:
        s = int(a[0])
        return (s, s + 1)
    if not np.all((a[1:] - a[:-1]) == 1):
        return None
    s = int(a[0])
    e = int(a[-1]) + 1
    if (e - s) != int(a.size):
        return None
    return (s, e)


def _gather0(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Gathers axis-0 entries with a contiguous-slice fast path.

    Args:
        arr (np.ndarray): Source array.
            Shape: (T, ...)
        idx (np.ndarray): Index vector along axis 0.
            Shape: (K,)

    Returns:
        np.ndarray: Gathered values.
            Shape: (K, ...)
    """
    sl = _try_contiguous_slice(idx)
    if sl is not None:
        s, e = sl
        return arr[s:e]
    return arr[idx]


def _feature_indices(start: int, fd: FeatureDef) -> np.ndarray:
    """Build the [history : current : future] window indices for one feature.

    Output has ``fd.window_size`` elements.  The current timestep sits at
    index ``fd.current_idx``.  Past indices that would fall before the
    sequence boundary are clamped to ``start``.
    """
    lo = int(start) - fd.history
    hi = int(start) + fd.future + 1
    idx = np.arange(lo, hi, dtype=np.int64)
    idx[idx < 0] = int(start)
    return idx


def _load_shard_into_ram(
    data_root: Path,
    meta: ShardMeta,
    *,
    cameras: list[str],
    feature_config: FeatureConfig,
) -> CachedShard:
    """
    Loads one shard's numeric arrays and frame-offset sidecars into RAM.

    Temporal parameters are read from ``feature_config`` directly.

    Args:
        data_root (Path): Dataset root directory.
        meta (ShardMeta): Shard metadata.
        cameras (list[str]): Requested camera names.
        feature_config (FeatureConfig): Declares which features to load and how
            to key them.  Only features listed here are read from HDF5.

    Returns:
        CachedShard: In-memory shard payload and flattened sample index arrays.
    """
    max_future_length = feature_config.max_future
    segments: dict[str, dict[str, int]] = {}
    for cam in cameras:
        v = meta.videos.get(cam)
        if not isinstance(v, dict) or ("sidecar" not in v):
            raise KeyError(f"Missing sidecar for cam={cam} in {meta.task_slug}/shard_{meta.shard_id:03d}")
        sc = _load_json(data_root / meta.task_slug / str(v["sidecar"]))
        segments[cam] = {s["demo_key"]: int(s["frame_start"]) for s in sc.get("segments", [])}

    hdf5_path = data_root / meta.hdf5_rel
    demos_by_id: list[DemoData] = []
    demo_id_chunks: list[np.ndarray] = []
    start_chunks: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as f:
        for d in meta.demos:
            demo_key = str(d["demo_key"])
            demo = f[f"data/{demo_key}"]
            feature_config.validate_hdf5(demo)
            arrays: dict[str, np.ndarray] = {}
            for fd in feature_config.features:
                if not fd.is_hdf5:
                    continue
                parts = [
                    _h5_read_array(demo, p, dtype=np.float32).astype(np.float32, copy=False)
                    for p in fd.paths
                ]
                arrays[fd.key] = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
            n = int(min(arr.shape[0] for arr in arrays.values())) if arrays else 0
            arrays = {k: v[:n] for k, v in arrays.items()}
            n_starts = max(0, n - int(max_future_length))
            starts = np.arange(n_starts, dtype=np.int32)
            image_base: dict[str, int] = {}
            for cam in cameras:
                if demo_key not in segments[cam]:
                    raise KeyError(f"demo_key '{demo_key}' missing in sidecar cam={cam} shard={meta.shard_id}")
                image_base[cam] = int(segments[cam][demo_key])
            demo_id = len(demos_by_id)
            demos_by_id.append(
                DemoData(
                    demo_id=demo_id,
                    task=str(d.get("task", meta.task)),
                    task_slug=str(d.get("task_slug", meta.task_slug)),
                    shard_id=int(meta.shard_id),
                    demo_key=demo_key,
                    arrays=arrays,
                    image_base=image_base,
                )
            )
            if n_starts > 0:
                demo_id_chunks.append(np.full((n_starts,), demo_id, dtype=np.int32))
                start_chunks.append(starts)
    if demo_id_chunks:
        all_demo_ids = np.concatenate(demo_id_chunks, axis=0)
        all_starts = np.concatenate(start_chunks, axis=0)
    else:
        all_demo_ids = np.empty((0,), dtype=np.int32)
        all_starts = np.empty((0,), dtype=np.int32)
    return CachedShard(meta=meta, demos_by_id=demos_by_id, all_demo_ids=all_demo_ids, all_starts=all_starts)


class _LRUFrameCache:
    """
    Thread-safe bounded LRU cache for decoded image frames.

    Attributes:
        max_entries (int): Maximum number of cached frame items.
        data (OrderedDict[tuple[tuple[str, int, str], int], np.ndarray]): Cache store.
        lock (threading.Lock): Mutex for concurrent access.
    """
    def __init__(self, max_entries: int):
        """
        Initializes cache capacity and storage.

        Args:
            max_entries (int): Maximum item count retained.
        """
        self.max_entries = max(1, int(max_entries))
        self.data: OrderedDict[tuple[tuple[str, int, str], int], np.ndarray] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple[tuple[str, int, str], int]) -> Optional[np.ndarray]:
        """
        Fetches a cached frame and refreshes LRU position.

        Args:
            key (tuple[tuple[str, int, str], int]): (source, absolute_frame_index).

        Returns:
            Optional[np.ndarray]: Cached frame if present.
                Shape: (height, width, 3)
        """
        with self.lock:
            val = self.data.get(key)
            if val is None:
                return None
            self.data.move_to_end(key)
            return val

    def put(self, key: tuple[tuple[str, int, str], int], value: np.ndarray) -> None:
        """
        Inserts a frame and evicts oldest entries beyond capacity.

        Args:
            key (tuple[tuple[str, int, str], int]): (source, absolute_frame_index).
            value (np.ndarray): Decoded frame.
                Shape: (height, width, 3)
        """
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.max_entries:
                self.data.popitem(last=False)


class RobotDataset(IterableDataset):
    """
    Iterable dataset implementing k-pass m-window decode-on-demand sampling.

    The dataset assigns shards to workers, loads `active_window_shards` shards at a time,
    samples a (complementary) partition for each pass (`k_passes`), then decodes only the
    frames needed for each FIFO block (no sample reordering).

    Temporal parameters (per-feature history/future windows) are
    read directly from ``feature_config`` and do not have separate constructor arguments.

    Attributes:
        data_dir (Path): Dataset root directory.
        cameras (list[str]): Requested camera names.
        repeat (bool): If True, repeats indefinitely across epochs.
        global_seed (int): Global sampling seed.
        k_passes (int): Pass count for strict disjoint coverage.
        active_window_shards (int): Number of shards loaded in one window.
        locality_block_size (int): Number of samples per decode block.
        active_sources_per_worker (int): Deprecated; kept for CLI/backwards compatibility.
        predecode_next_block (bool): Enables overlap by predecoding next block.
        frame_cache_max_entries (int): Bounded frame LRU cache size.
        split (Literal["train", "val", "all"]): Dataset split mode.
        val_fraction (float): Validation fraction used by deterministic split.
        split_seed (int): Seed for deterministic split scoring.
        profile_timing (bool): Enables worker profiling logs.
        profile_every_samples (int): Profiling report cadence.
        feature_config (FeatureConfig): Declares which scalar/vector features to
            load and how to assemble them. Also carries all temporal config
            (per-feature ``history`` and ``future``).
        shards (list[ShardMeta]): Filtered shard metadata list.
    """
    def __init__(
        self,
        data_dir: str,
        *,
        cameras: list[str] = ["agentview", "robot0_eye_in_hand"],
        repeat: bool = True,
        global_seed: int = 42,
        k_passes: int = 1,
        active_window_shards: int = 2,
        locality_block_size: int = 16,
        active_sources_per_worker: int = 2,
        predecode_next_block: bool = True,
        frame_cache_max_entries: int = 12000,
        split: Literal["train", "val", "all"] = "train",
        val_fraction: float = 0.05,
        split_seed: int = 0,
        profile_timing: bool = False,
        profile_every_samples: int = 500,
        log_stall_ms: float = 200.0,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initializes dataset configuration and validates requested modalities.

        Args:
            data_dir (str): Dataset root.
            cameras (list[str]): Requested camera names.
            repeat (bool): Repeat indefinitely when True.
            global_seed (int): Global random seed.
            k_passes (int): Number of disjoint coverage passes.
            active_window_shards (int): Shards loaded per window.
            locality_block_size (int): Samples processed per decode block.
            active_sources_per_worker (int): Deprecated; kept for backwards compatibility.
            predecode_next_block (bool): Enable async next-block decode.
            frame_cache_max_entries (int): Max decoded-frame cache entries.
            split (Literal["train", "val", "all"]): Split name.
            val_fraction (float): Validation split fraction.
            split_seed (int): Split seed.
            profile_timing (bool): Enable profile logs.
            profile_every_samples (int): Profile log cadence.
            log_stall_ms (float): Print a STALL line whenever any single blocking
                phase (shard load, window preload wait, block decode wait) exceeds
                this many milliseconds. Set to 0 to disable.
            feature_config (Optional[FeatureConfig]): Declares which features to
                load and how to map raw sources to batch keys. Carries all
                temporal config (per-feature ``history`` / ``future``).
                Defaults to ``FeatureConfig.default()``.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cameras = list(cameras)
        self.repeat = bool(repeat)
        self.global_seed = int(global_seed)
        self.k_passes = int(k_passes)
        self.active_window_shards = int(active_window_shards)
        self.locality_block_size = int(locality_block_size)
        self.active_sources_per_worker = int(active_sources_per_worker)
        self.predecode_next_block = bool(predecode_next_block)
        self.frame_cache_max_entries = int(frame_cache_max_entries)
        self.split = str(split)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.profile_timing = bool(profile_timing)
        self.profile_every_samples = int(profile_every_samples)
        self.log_stall_ms = float(log_stall_ms)
        if feature_config is None:
            raise ValueError("RobotDataset requires a feature_config.")
        self.feature_config = feature_config
        if self.k_passes <= 0:
            raise ValueError("k_passes must be > 0")
        if self.active_window_shards <= 0:
            raise ValueError("active_window_shards must be > 0")
        if self.locality_block_size <= 0:
            raise ValueError("locality_block_size must be > 0")
        if self.active_sources_per_worker <= 0:
            raise ValueError("active_sources_per_worker must be > 0")
        if self.profile_every_samples <= 0:
            raise ValueError("profile_every_samples must be > 0")

        shards_all = load_shard_metas_from_manifests(str(self.data_dir))
        self.shards = split_shard_metas(
            shards_all, split=self.split, val_fraction=self.val_fraction, split_seed=self.split_seed  # type: ignore[arg-type]
        )
        self._validate_decode_request()

    def _active_video_cameras(self, fd: FeatureDef) -> tuple[str, ...]:
        """
        Resolves the active camera list for one video feature.

        The feature declares the full available camera set; ``self.cameras`` is
        treated as the runtime selection and order.
        """
        if not fd.is_video:
            return tuple()
        source_cameras = set(fd.camera_names)
        return tuple(cam for cam in self.cameras if cam in source_cameras)

    def _validate_decode_request(self) -> None:
        """
        Validates dataset/video availability for requested modalities.

        Raises:
            ValueError: If shards/camera layout is incompatible.
            KeyError: If required numeric/video fields are missing.
        """
        if not self.shards:
            raise ValueError("No shards discovered")
        video_features = [fd for fd in self.feature_config.features if fd.is_video]
        for sh in self.shards:
            if video_features:
                if sh.video_pack != "per_shard":
                    raise ValueError(
                        f"Requested video features, but task={sh.task_slug} has video_pack={sh.video_pack!r}"
                    )
                for fd in video_features:
                    for cam in self._active_video_cameras(fd):
                        v = sh.videos.get(cam)
                        if not isinstance(v, dict) or ("mp4" not in v) or ("sidecar" not in v):
                            raise KeyError(
                                f"Missing video for feature='{fd.key}' in "
                                f"{sh.task_slug}/shard_{sh.shard_id:03d} cam={cam}"
                            )
        probe = self.shards[0]
        h5_path = self.data_dir / probe.hdf5_rel
        demo_key = str(probe.demos[0]["demo_key"])
        with h5py.File(h5_path, "r") as f:
            demo = f[f"data/{demo_key}"]
            for fd in self.feature_config.features:
                if not fd.is_hdf5:
                    continue
                missing = [p for p in fd.paths if not _h5_has(demo, p)]
                if missing:
                    raise KeyError(
                        f"Feature '{fd.key}': HDF5 path(s) not found: {missing}"
                    )

        # Validate video frame resolution against declared feature shapes.
        # probe is a ShardMeta, so task_slug and videos are direct attributes.
        for fd in self.feature_config.features:
            if not fd.is_video:
                continue
            for cam in self._active_video_cameras(fd):
                mp4_path = self.data_dir / probe.task_slug / str(probe.videos[cam]["mp4"])
                frames = _decode_mp4_selected_frames(mp4_path, np.array([0], dtype=np.int64))
                self.feature_config.validate_video(cam, frames[0])

    @staticmethod
    def _source_key(shard: CachedShard, cam: str) -> tuple[str, int, str]:
        """
        Builds unique video source key for decode grouping.

        Args:
            shard (CachedShard): Source shard.
            cam (str): Camera name.

        Returns:
            tuple[str, int, str]: (task_slug, shard_id, cam).
        """
        return (str(shard.meta.task_slug), int(shard.meta.shard_id), str(cam))

    # NOTE: we intentionally do not reorder samples for "locality".
    # Decoding is already grouped by MP4 source and frames are decoded in sorted frame-index order.

    def _collect_frame_requests_for_block(
        self,
        chosen: list[tuple[CachedShard, DemoData, int]],
        *,
        include_cache: bool,
        frame_cache: _LRUFrameCache,
    ) -> tuple[
        Dict[tuple[str, int, str], set[int]],
        Dict[tuple[str, int, str], Path],
        Dict[tuple[tuple[str, int, str], int], np.ndarray],
        int,
    ]:
        """
        Collects frame requests for one block and optionally resolves cache hits.

        Args:
            chosen (list[tuple[CachedShard, DemoData, int]]): Block samples.
            include_cache (bool): If True, attempt frame-cache lookups.
            frame_cache (_LRUFrameCache): Shared worker-local frame cache.

        Returns:
            tuple[Dict[...], Dict[...], Dict[...], int]:
                - request map by source -> absolute indices
                - source -> MP4 path
                - pre-resolved decoded frames map
                - total requested frame count across all video features
        """
        req_map: Dict[tuple[str, int, str], set[int]] = {}
        src_to_path: Dict[tuple[str, int, str], Path] = {}
        resolved: Dict[tuple[tuple[str, int, str], int], np.ndarray] = {}
        requested = 0
        for sh, dm, st in chosen:
            for fd in self.feature_config.features:
                if not fd.is_video:
                    continue
                window_idx = _feature_indices(st, fd)
                for cam in self._active_video_cameras(fd):
                    source = self._source_key(sh, cam)
                    req_map.setdefault(source, set())
                    src_to_path[source] = self.data_dir / sh.meta.task_slug / str(sh.meta.videos[cam]["mp4"])
                    base = int(dm.image_base[cam])
                    for i in window_idx.tolist():
                        abs_i = base + int(i)
                        key_i = (source, int(abs_i))
                        requested += 1
                        if include_cache:
                            c = frame_cache.get(key_i)
                            if c is not None:
                                resolved[key_i] = c
                            else:
                                req_map[source].add(int(abs_i))
                        else:
                            req_map[source].add(int(abs_i))
        return req_map, src_to_path, resolved, requested

    def _decode_frames_for_block(
        self,
        chosen: list[tuple[CachedShard, DemoData, int]],
        *,
        include_cache: bool,
        frame_cache: _LRUFrameCache,
        cap_cache: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[tuple[tuple[str, int, str], int], np.ndarray], int, int, int]:
        """
        Decodes all unresolved frames for one selected block.

        Args:
            chosen (list[tuple[CachedShard, DemoData, int]]): Block samples.
            include_cache (bool): Enables cache read/write path.
            frame_cache (_LRUFrameCache): Shared worker-local frame cache.
            cap_cache (Optional[Dict[str, Any]]): Per-worker VideoCapture handle cache.
                When provided, VideoCapture objects are reused across blocks to avoid
                repeated file open/probe overhead.

        Returns:
            tuple[Dict[...], int, int, int]:
                - decoded frame map keyed by (source, absolute_index)
                - requested frame count
                - decode call count
                - unique decoded frame count
        """
        req_map, src_to_path, resolved, requested = self._collect_frame_requests_for_block(
            chosen, include_cache=include_cache, frame_cache=frame_cache
        )
        decode_calls = 0
        unique_frames = 0
        # Decode in a stable shard/cam order for clarity and locality.
        for source in sorted(req_map.keys()):
            idx_set = req_map[source]
            if not idx_set:
                continue
            idx_sorted = np.asarray(sorted(idx_set), dtype=np.int64)
            frames = _decode_mp4_selected_frames(src_to_path[source], idx_sorted, cap_cache=cap_cache)
            decode_calls += 1
            unique_frames += int(len(idx_sorted))
            for i, fr in zip(idx_sorted.tolist(), frames, strict=True):
                key = (source, int(i))
                resolved[key] = fr
                if include_cache:
                    frame_cache.put(key, fr)
        return resolved, requested, decode_calls, unique_frames

    def _assemble_sample(
        self,
        *,
        shard: CachedShard,
        demo: DemoData,
        start: int,
        decoded_frames: Dict[tuple[tuple[str, int, str], int], np.ndarray],
    ) -> dict[str, Any]:
        """
        Assembles one training sample dictionary from decoded assets.

        Every feature produces a payload with an explicit time axis:

        * HDF5-backed features: ``(T, *shape)``
        * Video-backed features: ``dict[cam] -> (T, C, H, W)``

        The output batch dict is flat — no ``"observations"`` nesting.
        A feature may represent either a past/current history window or a
        current/future rollout window.

        Args:
            shard (CachedShard): Source shard.
            demo (DemoData): Source demo.
            start (int): Current timestep index.
            decoded_frames (Dict[tuple[tuple[str, int, str], int], np.ndarray]): Frame map.

        Returns:
            dict[str, Any]: Sample payload as configured by self.feature_config.
        """
        out: dict[str, Any] = {}

        for fd in self.feature_config.features:
            window_idx = _feature_indices(int(start), fd)
            if fd.is_hdf5:
                arr = demo.arrays[fd.key]
                out[fd.key] = torch.from_numpy(_gather0(arr, window_idx))
                continue
            if fd.is_video:
                images: dict[str, torch.Tensor] = {}
                for cam in self._active_video_cameras(fd):
                    source = self._source_key(shard, cam)
                    base = int(demo.image_base[cam])
                    abs_window = base + window_idx
                    imgs = np.stack(
                        [decoded_frames[(source, int(i))] for i in abs_window.tolist()],
                        axis=0,
                    )  # (T, H, W, C)
                    images[cam] = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(torch.uint8)
                out[fd.key] = images
                continue
            raise TypeError(f"Unsupported feature source: {fd.source_type!r}")

        out.update({
            "task": demo.task,
            "task_slug": demo.task_slug,
            "shard_id": int(demo.shard_id),
            "demo_key": demo.demo_key,
            "demo_idx": int(demo.demo_id),
            "frame_idx": int(start),
        })
        return out

    def __iter__(self) -> Iterable[dict[str, Any]]:
        """
        Iterates samples for one worker using k-pass m-window decode-on-demand flow.

        Pipeline overview (per epoch):
          1. Shuffle shard order once → stable pairing across all k passes.
          2. Slide an m-shard window over the shuffled shard list.
             a. Load all m shards from HDF5/sidecar into RAM (no video decode yet).
             b. For each pass_idx in [0, k_passes), take exactly 1/k of each shard's samples
                using a shard+epoch deterministic seed (complementary/disjoint across passes).
             c. Concatenate all chosen samples across the m shards and shuffle them (training stream).
             d. Process the stream in blocks: decode only frames needed for the block, then yield.
          3. Optionally overlap block N+1 decode with block N yield (predecode_next_block).

        Yields:
            Iterable[dict[str, Any]]: Sample dictionaries consumable by trainer.
        """
        # ── Worker identity ──────────────────────────────────────────────────────────
        worker_info = torch.utils.data.get_worker_info()
        # worker_id: 0-based index of this DataLoader worker process.
        worker_id = worker_info.id if worker_info is not None else 0
        # num_workers: total DataLoader worker count; 1 when running in main process.
        num_workers = worker_info.num_workers if worker_info is not None else 1
        # assigned: this worker's slice of the global shard list (stride = num_workers).
        # e.g. worker 2 of 4 gets shards [2, 6, 10, ...] from the full list.
        assigned = self.shards[worker_id::num_workers]
        if not assigned:
            # Worker has no shards to process (e.g. more workers than shards); exit cleanly.
            return

        # ── Per-worker utilities ─────────────────────────────────────────────────────
        profiler = DatasetWorkerProfiler(
            enabled=bool(self.profile_timing),
            worker_id=worker_id,
            every_samples=int(self.profile_every_samples),
            start_s=now_s(),
        )
        # Shared LRU frame cache across all epochs/windows for this worker process.
        # Keyed by (source_key, abs_frame_idx); evicts least-recently-used entries
        # when size exceeds frame_cache_max_entries.
        frame_cache = _LRUFrameCache(self.frame_cache_max_entries)

        # Per-worker VideoCapture handle cache: keeps MP4 files open within one shard window.
        # Cleared and released at each window transition, so at most
        # active_window_shards × num_cameras handles are open at any time.
        # Avoids 20–100 ms open+probe cost per block that caused periodic stalls.
        cap_cache: Dict[str, Any] = {}

        # Single background-decode thread reused across all epochs, windows, and passes.
        # Creating a new ThreadPoolExecutor per pass (inside the k-pass loop) caused hangs:
        # shutdown(wait=True) would block indefinitely if cv2.VideoCapture stalled on a seek.
        prefetch_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1) if self.predecode_next_block else None
        )

        # Separate single-thread executor for pre-loading the next window's shard data
        # (HDF5 arrays + sidecar JSON) in the background while processing the current window.
        # Uses a dedicated executor so shard I/O never blocks the video decode thread.
        shard_load_executor = ThreadPoolExecutor(max_workers=1)
        next_window_future: Optional[Future] = None

        # ── Epoch loop ───────────────────────────────────────────────────────────────
        epoch = 0
        while True:  # repeat=True → loop forever; repeat=False → break after one epoch.

            # Fresh RNG per epoch, unique per worker.
            # XOR of three independent offsets ensures no two workers share the same
            # sequence even if global_seed or worker_id alone would collide.
            rng = random.Random((self.global_seed * 10007) ^ (worker_id * 1009) ^ (epoch * 9173))

            # Build an index list [0, 1, ..., N-1] and shuffle it in place.
            # This defines the visit order of shards for the ENTIRE epoch (all k passes).
            # Same order is reused across passes so window pairings are stable.
            shard_order = list(range(len(assigned)))
            rng.shuffle(shard_order)

            # ── m-shard window loop ───────────────────────────────────────────────
            # Load each shard window once, then iterate all k passes over it.
            for window_start in range(0, len(shard_order), self.active_window_shards):
                window_ids = shard_order[window_start : window_start + self.active_window_shards]
                window_metas = [assigned[i] for i in window_ids]

                # Release VideoCapture handles from the previous window before loading
                # the new one.  This bounds cap_cache to active_window_shards × num_cameras
                # open file descriptors regardless of total dataset size.
                for cap in cap_cache.values():
                    cap.release()
                cap_cache.clear()

                # ── Load m shards into RAM ────────────────────────────────────────
                # Use the pre-loaded result from the previous iteration if available;
                # otherwise load synchronously (first window of each epoch).
                t0 = now_s()
                if next_window_future is not None:
                    window_shards, window_orders = next_window_future.result()
                    next_window_future = None
                    dt_ms = (now_s() - t0) * 1000.0
                    if self.log_stall_ms > 0 and dt_ms > self.log_stall_ms:
                        print(
                            f"[datasetv2 STALL preload_wait worker={worker_id} "
                            f"epoch={epoch} window={window_start}] "
                            f"waited={dt_ms:.0f}ms for background shard preload "
                            f"(preload did not finish before window was needed)",
                            flush=True,
                        )
                    elif self.log_stall_ms > 0:
                        print(
                            f"[datasetv2 shard_preload_ready worker={worker_id} "
                            f"epoch={epoch} window={window_start}] "
                            f"preloaded shards ready in {dt_ms:.0f}ms (no stall)",
                            flush=True,
                        )
                else:
                    window_shards, window_orders = _load_window_data(
                        self.data_dir, window_metas, self.cameras, self.feature_config,
                        self.global_seed, worker_id, epoch, self.k_passes,
                    )
                    dt_ms = (now_s() - t0) * 1000.0
                    if self.log_stall_ms > 0:
                        print(
                            f"[datasetv2 shard_load_sync worker={worker_id} "
                            f"epoch={epoch} window={window_start}] "
                            f"loaded {len(window_metas)} shards in {dt_ms:.0f}ms (synchronous — first window of epoch)",
                            flush=True,
                        )
                profiler.add_time("shard_load_s", float(now_s() - t0))

                # Pre-load the NEXT window in the background while we process this one.
                # By the time k_passes finishes, the next window's HDF5 + JSON data
                # will already be in RAM, eliminating the shard-loading stall at window
                # boundaries.  Uses a separate executor so shard I/O never queues behind
                # a video-decode job in prefetch_executor.
                next_window_start = window_start + self.active_window_shards
                if next_window_start < len(shard_order):
                    _nw_ids = shard_order[next_window_start : next_window_start + self.active_window_shards]
                    _nw_metas = [assigned[i] for i in _nw_ids]
                    next_window_future = shard_load_executor.submit(
                        _load_window_data,
                        self.data_dir, _nw_metas, self.cameras, self.feature_config,
                        self.global_seed, worker_id, epoch, self.k_passes,
                    )

                # ── k-pass loop (no shard reload) ─────────────────────────────────
                for pass_idx in range(self.k_passes):
                    t_plan0 = now_s()
                    window_samples: list[tuple[CachedShard, DemoData, int]] = []
                    for sh, order in zip(window_shards, window_orders, strict=True):
                        s, e = _k_pass_slice_bounds(total=int(order.shape[0]), pass_idx=int(pass_idx), k_passes=int(self.k_passes))
                        idx = order[s:e]
                        if idx.size <= 0:
                            continue
                        demo_ids = sh.all_demo_ids[idx]
                        starts = sh.all_starts[idx]
                        for did, st in zip(demo_ids.tolist(), starts.tolist(), strict=True):
                            window_samples.append((sh, sh.demos_by_id[int(did)], int(st)))

                    # Randomize the training stream for this window+pass.
                    rng.shuffle(window_samples)
                    profiler.add_time("plan_s", float(now_s() - t_plan0))
                    if not window_samples:
                        continue

                    # ── FIFO block pipeline with optional predecode overlap ───────
                    stream_cursor = 0
                    block_idx = 0

                    staged_block: Optional[list[tuple[CachedShard, DemoData, int]]] = None
                    staged_future: Optional[Future] = None

                    def _next_block() -> Optional[list[tuple[CachedShard, DemoData, int]]]:
                        nonlocal stream_cursor
                        if stream_cursor >= len(window_samples):
                            return None
                        blk = window_samples[stream_cursor : stream_cursor + self.locality_block_size]
                        stream_cursor += len(blk)
                        return blk

                    def _stage_next() -> None:
                        nonlocal staged_block, staged_future
                        if staged_block is not None:
                            return
                        blk = _next_block()
                        if blk is None:
                            return
                        staged_block = blk
                        if prefetch_executor is not None:
                            staged_future = prefetch_executor.submit(
                                self._decode_frames_for_block, blk, include_cache=False, frame_cache=frame_cache, cap_cache=cap_cache
                            )
                        else:
                            staged_future = None

                    _stage_next()

                    while staged_block is not None:
                        current_block = staged_block
                        current_future = staged_future
                        staged_block = None
                        staged_future = None

                        # Stage next before decoding/yielding to maximize overlap.
                        _stage_next()

                        t_dec0 = now_s()
                        if current_future is not None:
                            # wait util get the results.
                            decoded, requested, calls, unique = current_future.result()
                            dt_wait_ms = (now_s() - t_dec0) * 1000.0
                            if self.log_stall_ms > 0 and dt_wait_ms > self.log_stall_ms:
                                print(
                                    f"[datasetv2 STALL decode_wait worker={worker_id} "
                                    f"epoch={epoch} window={window_start} pass={pass_idx} "
                                    f"block={block_idx}] "
                                    f"waited={dt_wait_ms:.0f}ms for background video decode "
                                    f"(block_size={len(current_block)})",
                                    flush=True,
                                )
                            for k, v in decoded.items():
                                frame_cache.put(k, v)
                        else:
                            decoded, requested, calls, unique = self._decode_frames_for_block(
                                current_block, include_cache=True, frame_cache=frame_cache, cap_cache=cap_cache
                            )
                        profiler.add_time("decode_s", float(now_s() - t_dec0))
                        profiler.add_count("requested_frames", int(requested))
                        profiler.add_count("decode_calls", int(calls))
                        profiler.add_count("decoded_unique_frames", int(unique))

                        t_asm0 = now_s()
                        for sh, dm, st in current_block:
                            yield self._assemble_sample(shard=sh, demo=dm, start=st, decoded_frames=decoded)
                            profiler.sample_yielded()
                            profiler.maybe_report()
                        profiler.add_time("assemble_s", float(now_s() - t_asm0))
                        block_idx += 1

            # ── End of epoch ─────────────────────────────────────────────────────────
            epoch += 1
            if not self.repeat:
                # Single-pass mode (e.g. evaluation): emit final profiler report and stop.
                profiler.maybe_report(force=True)
                # Clean up executors and VideoCapture handles on generator exit.
                if prefetch_executor is not None:
                    prefetch_executor.shutdown(wait=False)
                shard_load_executor.shutdown(wait=False)
                for cap in cap_cache.values():
                    cap.release()
                cap_cache.clear()
                return
            # reset next_window_future for the new epoch (the pre-loaded data used
            # the old epoch's seed — discard it and let the first window reload).
            if next_window_future is not None:
                try:
                    next_window_future.result()
                except Exception:
                    pass
                next_window_future = None


def create_robot_dataloader(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    **dataset_kwargs: Any,
):
    """
    Creates a DataLoader backed by `RobotDataset`.

    Args:
        data_dir (str): Dataset root directory.
        batch_size (int): Batch size.
        num_workers (int): DataLoader worker count.
        prefetch_factor (int): DataLoader prefetch factor.
        **dataset_kwargs (Any): Forwarded dataset constructor args.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader for dataset iteration.
    """
    from torch.utils.data import DataLoader

    dataset = RobotDataset(data_dir=data_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
