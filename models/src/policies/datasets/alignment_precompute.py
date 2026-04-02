from __future__ import annotations

import json
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import h5py
import numpy as np

from policies.datasets.alignment_cache import CacheSpec, load_shard_cache, save_shard_cache
from policies.datasets.schema import TimestampSchema
from policies.datasets.timestamps import Alignment, LatestTimestampSampler, TimestampSampler


class ShardLike(Protocol):
    task_slug: str
    shard_id: int
    hdf5_rel: str
    demos: list[dict]
    timestamps_hz: Optional[float]


@dataclass(frozen=True)
class CacheGroupSpec:
    cache_dir: Path
    task_slug: str
    train_hz: float
    sampler_name: str
    cameras: Tuple[str, ...]

    def dir_relpath(self) -> Path:
        hz = f"{self.train_hz:.6f}".rstrip("0").rstrip(".")
        # Must match CacheSpec.relpath() folder layout.
        # CacheSpec includes cams_<key>/shard_XXX.npz, so marker lives next to shard files.
        dummy = CacheSpec(
            cache_dir=self.cache_dir,
            task_slug=self.task_slug,
            shard_id=0,
            train_hz=self.train_hz,
            sampler_name=self.sampler_name,
            cameras=self.cameras,
        )
        return dummy.relpath().parent

    def dir_path(self) -> Path:
        return self.cache_dir / self.dir_relpath()

    def complete_marker_path(self) -> Path:
        return self.dir_path() / "_COMPLETE.json"


def _atomic_write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if dst.exists():
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return
    tmp.replace(dst)


def _hash_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
        try:
            data = p.read_bytes()
        except Exception:
            data = b""
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()


def _discover_manifest_paths(dataset_root: Path, task_slugs: set[str]) -> list[Path]:
    ds_path = dataset_root / "dataset_manifest.json"
    paths: list[Path] = [ds_path]
    try:
        ds = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception:
        return paths
    for t in ds.get("tasks", []):
        slug = str(t.get("task_slug", ""))
        if slug and slug in task_slugs:
            m = t.get("manifest")
            if m:
                paths.append(dataset_root / str(m))
    return paths


def _h5_has(demo: h5py.Group, path: str) -> bool:
    try:
        demo[path]
        return True
    except Exception:
        return False


def _h5_read_array(demo: h5py.Group, path: str, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = demo[path][()]
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def _safe_get_float_attr(attrs: object, key: str) -> Optional[float]:
    """
    Best-effort scalar float attr reader.

    Avoid `dict(f.attrs)` because converting all attrs can trigger object-converter
    failures on unsupported attr types (e.g. PYTHON:OBJECT).
    """
    try:
        if key not in attrs:  # type: ignore[operator]
            return None
        v = attrs[key]  # type: ignore[index]
        if isinstance(v, np.ndarray):
            if v.size != 1:
                return None
            v = v.reshape(-1)[0]
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8", errors="ignore")
        return float(v)
    except Exception:
        return None


def _load_demo_timestamps(
    demo: h5py.Group,
    *,
    schema: TimestampSchema,
    cameras: Sequence[str],
    meta_timestamps_hz: Optional[float],
    actions_len: int,
    video_fps_attr: Optional[float],
) -> Dict[str, np.ndarray]:
    if _h5_has(demo, schema.actions_timestamps_path):
        action_ts = _h5_read_array(demo, schema.actions_timestamps_path, dtype=np.float64)
    elif _h5_has(demo, schema.legacy_actions_timestamps_path):
        action_ts = _h5_read_array(demo, schema.legacy_actions_timestamps_path, dtype=np.float64)
    else:
        hz = float(meta_timestamps_hz or 50.0)
        action_ts = (np.arange(int(actions_len), dtype=np.float64) / hz)

    robot_ts = _h5_read_array(demo, schema.robot_state_timestamps_path, dtype=np.float64) if _h5_has(demo, schema.robot_state_timestamps_path) else action_ts
    env_ts = _h5_read_array(demo, schema.env_state_timestamps_path, dtype=np.float64) if _h5_has(demo, schema.env_state_timestamps_path) else action_ts

    out: Dict[str, np.ndarray] = {"actions": action_ts, "robot_state": robot_ts, "env_state": env_ts}
    if cameras:
        fps = float(meta_timestamps_hz or 50.0)
        if video_fps_attr is not None:
            fps = float(video_fps_attr)
        for cam in cameras:
            p = schema.image_timestamps_template.format(cam=cam)
            out[f"images/{cam}"] = _h5_read_array(demo, p, dtype=np.float64) if _h5_has(demo, p) else (np.arange(int(actions_len), dtype=np.float64) / fps)
    return out


def _ensure_one_shard_cache(
    *,
    dataset_root: Path,
    meta: ShardLike,
    train_hz: float,
    sampler: TimestampSampler,
    cameras: Tuple[str, ...],
    schema: TimestampSchema,
    actions_hdf5_path: str = "actions",
    cache_dir: Path,
    debug: bool,
) -> tuple[str, int, bool]:
    """
    Returns (task_slug, shard_id, ok).

    ok means: cache file exists and covers all demo_keys in meta.demos.
    """
    cache_spec = CacheSpec(
        cache_dir=Path(cache_dir),
        task_slug=str(meta.task_slug),
        shard_id=int(meta.shard_id),
        train_hz=float(train_hz),
        sampler_name=str(type(sampler).__name__),
        cameras=tuple(cameras),
    )

    cached = load_shard_cache(cache_spec)
    if cached is not None:
        demo_keys_needed = [str(d.get("demo_key", "")) for d in meta.demos if str(d.get("demo_key", ""))]
        if all((dk in cached) for dk in demo_keys_needed):
            return (str(meta.task_slug), int(meta.shard_id), True)

    # Precompute from HDF5 timestamps only.
    alignments: Dict[str, Alignment] = {}
    hdf5_path = dataset_root / str(meta.hdf5_rel)
    t0 = time.time()
    with h5py.File(str(hdf5_path), "r") as f:
        video_fps_attr = _safe_get_float_attr(f.attrs, "video_fps")
        for d in meta.demos:
            demo_key = str(d.get("demo_key", ""))
            if not demo_key:
                continue
            demo = f[f"data/{demo_key}"]
            # Shape-only read for actions length (no heavy array load).
            try:
                actions_len = int(demo[actions_hdf5_path].shape[0])
            except Exception:
                continue
            mod_ts = _load_demo_timestamps(
                demo,
                schema=schema,
                cameras=list(cameras),
                meta_timestamps_hz=meta.timestamps_hz,
                actions_len=actions_len,
                video_fps_attr=video_fps_attr,
            )
            al = sampler.precompute(mod_ts, train_hz=float(train_hz))
            if al is not None:
                alignments[demo_key] = al

    save_shard_cache(cache_spec, alignments)
    if debug:
        dt = time.time() - t0
        print(
            f"[align_precompute] wrote {cache_spec.path()} "
            f"(task={meta.task_slug} shard={int(meta.shard_id)} demos={len(meta.demos)} cached={len(alignments)}) "
            f"in {dt:.2f}s",
            flush=True,
        )
    # Consider this shard ok only if we covered all demo keys.
    demo_keys_needed = [str(d.get("demo_key", "")) for d in meta.demos if str(d.get("demo_key", ""))]
    ok = all((dk in alignments) for dk in demo_keys_needed)
    return (str(meta.task_slug), int(meta.shard_id), bool(ok))


def ensure_alignment_cache_complete(
    *,
    dataset_root: Path,
    metas: Sequence[ShardLike],
    train_hz: float,
    cameras: Sequence[str],
    schema: TimestampSchema,
    actions_hdf5_path: str = "actions",
    cache_dir: Path,
    timestamp_sampler: Optional[TimestampSampler] = None,
    debug: bool = False,
    max_workers: Optional[int] = None,
) -> None:
    """
    Eagerly precompute alignment caches for all provided shard metas.

    - Uses only HDF5 timestamps (no MP4 decode).
    - Skips shards whose cache file already exists and covers required demo keys.
    - Writes a per-(task_slug, train_hz, sampler, cameras) `_COMPLETE.json` marker once all shards are ok.
    """
    if float(train_hz) <= 0:
        raise ValueError("train_hz must be > 0")
    if not metas:
        return

    sampler = timestamp_sampler or LatestTimestampSampler()
    cameras_t = tuple(str(c) for c in cameras)

    # Group by task for completeness markers.
    by_task: Dict[str, list[ShardLike]] = {}
    for m in metas:
        by_task.setdefault(str(m.task_slug), []).append(m)

    # Determine which tasks need work by checking markers.
    tasks_to_process: Dict[str, list[ShardLike]] = {}
    for task, ms in by_task.items():
        shard_ids_expected = sorted({int(m.shard_id) for m in ms})
        manifest_hash = _hash_files(_discover_manifest_paths(Path(dataset_root), {str(task)}))
        group = CacheGroupSpec(
            cache_dir=Path(cache_dir),
            task_slug=str(task),
            train_hz=float(train_hz),
            sampler_name=str(type(sampler).__name__),
            cameras=cameras_t,
        )
        marker_path = group.complete_marker_path()
        ok_marker = False
        if marker_path.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                ok_marker = (
                    str(marker.get("task_slug")) == str(task)
                    and float(marker.get("train_hz")) == float(train_hz)
                    and str(marker.get("sampler_name")) == str(type(sampler).__name__)
                    and tuple(marker.get("cameras", [])) == tuple(sorted(cameras_t))
                    and str(marker.get("dataset_manifest_hash")) == str(manifest_hash)
                    and sorted(int(x) for x in marker.get("shard_ids_expected", [])) == shard_ids_expected
                    and sorted(int(x) for x in marker.get("shard_ids_cached", [])) == shard_ids_expected
                )
            except Exception:
                ok_marker = False
        # Also ensure the underlying shard cache files still exist (marker can go stale).
        if ok_marker:
            for m in ms:
                spec = CacheSpec(
                    cache_dir=Path(cache_dir),
                    task_slug=str(m.task_slug),
                    shard_id=int(m.shard_id),
                    train_hz=float(train_hz),
                    sampler_name=str(type(sampler).__name__),
                    cameras=cameras_t,
                )
                if (not spec.path().exists()) and (not spec.legacy_path().exists()):
                    ok_marker = False
                    break
        if not ok_marker:
            tasks_to_process[task] = ms

    if not tasks_to_process:
        if debug:
            print("[align_precompute] complete marker hit; skip recompute", flush=True)
        return

    # Precompute shards (parallel across shards).
    cpu = os.cpu_count() or 1
    if max_workers is None:
        max_workers = min(8, max(1, cpu // 2))

    results: Dict[tuple[str, int], bool] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        for _task, ms in tasks_to_process.items():
            for meta in ms:
                futures.append(
                    ex.submit(
                        _ensure_one_shard_cache,
                        dataset_root=Path(dataset_root),
                        meta=meta,
                        train_hz=float(train_hz),
                        sampler=sampler,
                        cameras=cameras_t,
                        schema=schema,
                        actions_hdf5_path=str(actions_hdf5_path),
                        cache_dir=Path(cache_dir),
                        debug=bool(debug),
                    )
                )
        for fut in as_completed(futures):
            task, shard_id, ok = fut.result()
            results[(str(task), int(shard_id))] = bool(ok)

    # Write/refresh markers per task.
    for task, ms in tasks_to_process.items():
        shard_ids_expected = sorted({int(m.shard_id) for m in ms})
        shard_ids_cached = sorted({int(sid) for (t, sid), ok in results.items() if t == str(task) and ok})
        group = CacheGroupSpec(
            cache_dir=Path(cache_dir),
            task_slug=str(task),
            train_hz=float(train_hz),
            sampler_name=str(type(sampler).__name__),
            cameras=cameras_t,
        )
        manifest_hash = _hash_files(_discover_manifest_paths(Path(dataset_root), {str(task)}))
        marker = {
            "task_slug": str(task),
            "train_hz": float(train_hz),
            "sampler_name": str(type(sampler).__name__),
            "cameras": list(sorted(cameras_t)),
            "dataset_manifest_hash": str(manifest_hash),
            "shard_ids_expected": shard_ids_expected,
            "shard_ids_cached": shard_ids_cached,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_write_text(group.complete_marker_path(), json.dumps(marker, indent=2, sort_keys=True) + "\n")

