"""
Split large robomimic-style HDF5 demonstration files into smaller shards, and
store images separately as MP4 to reduce per-file size and improve I/O.

This is the **importable** implementation. The CLI entrypoint lives at:
  `scripts/policies/convert_hdf5_to_small_files.py`

This script is designed for robomimic datasets that look like:

  /data/<demo_key>/
    actions                         (T, 7) float
    obs/robot0_eef_pos               (T, 3)
    obs/robot0_eef_quat              (T, 4)
    obs/robot0_gripper_qpos          (T, 2)
    obs/robot0_joint_pos             (T, 7)
    obs/object                       (T, 42)
    obs/agentview_image              (T, H, W, 3) uint8  [optional]
    obs/robot0_eye_in_hand_image     (T, H, W, 3) uint8  [optional]

Output layout (per task / per input file):

  <output_root>/<task_slug>/
    demo_manifest.json
    shards/
      shard_000.hdf5
      shard_001.hdf5
      ...
    videos_shards/
      agentview/shard_000.mp4
      robot0_eye_in_hand/shard_000.mp4

Each shard HDF5 keeps the same numeric dataset names (actions, obs/*) plus:
  - timestamps (T,) float64, assuming fixed rate (default: 50Hz).

The manifest records per-demo length, shard filename, and per-shard video files/segments.
"""

from __future__ import annotations

import argparse
import json
import random
import math
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


# Match the task instructions used elsewhere in this repo (see robomimic_to_lerobot_unified.py).
DEMO_FILE_TO_INSTRUCTION: dict[str, str] = {
    "demo_00_highres.hdf5": "place the cross into the bin",
    "demo_01_highres.hdf5": "place the cross into the cup",
    "demo_02_highres.hdf5": "place the cross into the plate",
    "demo_10_highres.hdf5": "place the cube into the bin",
    "demo_11_highres.hdf5": "place the cube into the cup",
    "demo_12_highres.hdf5": "place the cube into the plate",
    "demo_20_highres.hdf5": "place the cylinder into the bin",
    "demo_21_highres.hdf5": "place the cylinder into the cup",
    "demo_22_highres.hdf5": "place the cylinder into the plate",
}


CAMERA_TO_DATASET = {
    "agentview": "obs/agentview_image",
    "robot0_eye_in_hand": "obs/robot0_eye_in_hand_image",
}


def _task_slug_from_filename(filename: str) -> str:
    """
    Produce a stable folder name from the instruction mapping if available,
    otherwise fall back to the file stem.
    """
    instruction = DEMO_FILE_TO_INSTRUCTION.get(filename)
    if instruction is None:
        return Path(filename).stem
    # Example: "place the cross into the bin" -> "cross_bin"
    words = instruction.lower().split()
    obj = "unknown"
    container = "unknown"
    for cand in ["cross", "cube", "cylinder"]:
        if cand in words:
            obj = cand
            break
    for cand in ["bin", "cup", "plate"]:
        if cand in words:
            container = cand
            break
    return f"{obj}_{container}"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clear_directory_contents(p: Path) -> None:
    """Delete all files and subdirectories under `p`, but keep `p` itself."""
    if not p.exists():
        return
    if not p.is_dir():
        raise NotADirectoryError(f"Expected directory path, got: {p}")
    for child in p.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _json_dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_demo_keys(f: h5py.File) -> List[str]:
    if "data" not in f:
        raise ValueError("Input HDF5 does not contain top-level group 'data'")
    return sorted(list(f["data"].keys()))


def _read_numeric_demo(demo: h5py.Group) -> dict:
    """
    Read all numeric datasets for a single demo into memory.
    We intentionally do NOT load images here.
    """
    out = {}
    out["actions"] = demo["actions"][()]
    out["obs/robot0_eef_pos"] = demo["obs/robot0_eef_pos"][()]
    out["obs/robot0_eef_quat"] = demo["obs/robot0_eef_quat"][()]
    out["obs/robot0_gripper_qpos"] = demo["obs/robot0_gripper_qpos"][()]
    out["obs/robot0_joint_pos"] = demo["obs/robot0_joint_pos"][()]
    out["obs/object"] = demo["obs/object"][()]
    # Also compute object poses relative to robot EEF (world -> eef frame).
    #
    # This mirrors `transform_to_relative_coordinates` from
    # `repos/robomimic-lerobot-pipeline/utils/robomimic_to_lerobot_unified.py`.
    eef_pos = out["obs/robot0_eef_pos"].astype(np.float32, copy=False)
    eef_quat = out["obs/robot0_eef_quat"].astype(np.float32, copy=False)
    obj = out["obs/object"].astype(np.float32, copy=False)
    out["obs/object_rel_eef"] = _object_world_to_eef_relative(eef_pos=eef_pos, eef_quat=eef_quat, objects=obj)
    # Optional timestamps (if the source provides them). We do not assume a particular name,
    # but we check a couple common conventions.
    #
    # If missing, the caller will synthesize timestamps from a fixed Hz.
    if "timestamps" in demo:
        out["timestamps"] = demo["timestamps"][()]
        out["timestamps_source"] = "source:timestamps"
    elif "obs" in demo and "timestamps" in demo["obs"]:
        out["timestamps"] = demo["obs/timestamps"][()]
        out["timestamps_source"] = "source:obs/timestamps"
    else:
        out["timestamps"] = None
        out["timestamps_source"] = "assumed"
    return out


def _make_timestamps(length: int, hz: float) -> np.ndarray:
    return np.arange(length, dtype=np.float64) / float(hz)


def _quat_inverse_xyzw_np(q: np.ndarray) -> np.ndarray:
    q_inv = q.copy()
    q_inv[..., :3] *= -1.0
    return q_inv


def _quat_multiply_xyzw_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=-1)


def _quat_canonicalize_xyzw_np(q: np.ndarray) -> np.ndarray:
    q2 = q.copy()
    mask = q2[..., 3] < 0
    q2[mask] *= -1.0
    return q2


def _object_world_to_eef_relative(*, eef_pos: np.ndarray, eef_quat: np.ndarray, objects: np.ndarray) -> np.ndarray:
    """Convert packed object poses from world frame to EEF-relative.

    Args:
        eef_pos: (T,3)
        eef_quat: (T,4) in [x,y,z,w]
        objects: (T, 7*N) where each entity is [pos(3), quat(4)] in [x,y,z,w]
    Returns:
        (T, 7*N) in the same packing but relative to EEF.
    """
    if eef_pos.ndim != 2 or eef_quat.ndim != 2 or objects.ndim != 2:
        raise ValueError("Expected eef_pos/eef_quat/objects to have shape (T,D).")
    if eef_pos.shape[0] != eef_quat.shape[0] or eef_pos.shape[0] != objects.shape[0]:
        raise ValueError("eef_pos/eef_quat/objects must share the same T.")
    if eef_pos.shape[1] != 3 or eef_quat.shape[1] != 4:
        raise ValueError("Expected eef_pos (T,3) and eef_quat (T,4).")
    T, D = objects.shape
    n = D // 7
    base = n * 7
    if n <= 0:
        raise ValueError("objects must have at least one entity (D>=7).")

    world = objects[:, :base].reshape(T, n, 7)
    obj_pos = world[..., :3]
    obj_quat = world[..., 3:7]

    pos_rel = obj_pos - eef_pos[:, None, :]
    eef_inv = _quat_inverse_xyzw_np(eef_quat)[:, None, :]
    quat_rel = _quat_multiply_xyzw_np(eef_inv, obj_quat)
    quat_rel = _quat_canonicalize_xyzw_np(quat_rel)
    rel = np.concatenate([pos_rel, quat_rel], axis=-1).reshape(T, base)
    if base < D:
        rel = np.concatenate([rel, objects[:, base:]], axis=-1)
    return rel.astype(np.float32, copy=False)


class VideoWriterError(RuntimeError):
    pass


def _get_video_writer_backend(*, overwrite: bool):
    """
    Returns (backend_name, open_writer_fn).
    open_writer_fn(path, fps, height, width) -> writer with .append(frame) and .close()
    """
    # For debugging: do not silently fall back between backends.
    # We require `ffmpeg` and use it exclusively to produce widely playable H.264 MP4s.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoWriterError("ffmpeg not found on PATH; required for MP4 writing in this script.")

    class _FFmpegWriter:
        def __init__(self, path: str, fps: float, height: int, width: int):
            self._path = path
            self._fps = float(fps)
            self._height = int(height)
            self._width = int(width)
            # Encode RGB frames piped via stdin.
            # -pix_fmt yuv420p: maximum compatibility
            # -movflags +faststart: makes files seek/play before full download (and helps some players)
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if overwrite else "-n",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self._width}x{self._height}",
                "-r",
                str(self._fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-g",
                "10",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                path,
            ]
            self._p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if self._p.stdin is None:
                raise VideoWriterError("Failed to open ffmpeg stdin pipe")

        def append(self, frame: np.ndarray) -> None:
            if frame.shape != (self._height, self._width, 3):
                raise VideoWriterError(f"Expected frame shape {(self._height, self._width, 3)}, got {frame.shape}")
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)
            frame = np.ascontiguousarray(frame)
            assert self._p.stdin is not None
            self._p.stdin.write(frame.tobytes())

        def append_many(self, frames: np.ndarray) -> None:
            # frames: (T,H,W,3) uint8
            if frames.ndim != 4 or frames.shape[1:] != (self._height, self._width, 3):
                raise VideoWriterError(f"Expected frames shape (T,{self._height},{self._width},3), got {frames.shape}")
            if frames.dtype != np.uint8:
                frames = frames.astype(np.uint8, copy=False)
            frames = np.ascontiguousarray(frames)
            assert self._p.stdin is not None
            self._p.stdin.write(frames.tobytes())

        def close(self) -> None:
            """
            Finalize the ffmpeg process.

            Important: do NOT use Popen.communicate() after manually closing stdin; in some Python
            builds this can raise `ValueError: flush of closed file`. Instead, close stdin, drain
            stderr, then wait.
            """
            if self._p.stdin is not None and (not self._p.stdin.closed):
                self._p.stdin.close()
            # Drain stderr (ffmpeg is run with -loglevel error, so this should be small).
            err = b""
            if self._p.stderr is not None:
                err = self._p.stderr.read()
                self._p.stderr.close()
            rc = self._p.wait()
            if rc != 0:
                msg = (err or b"").decode("utf-8", errors="replace")
                raise VideoWriterError(f"ffmpeg failed for {self._path} (rc={rc}): {msg}")

    def _open(path: str, fps: float, height: int, width: int):
        return _FFmpegWriter(path, fps, height, width)

    return "ffmpeg", _open


def _open_video_writer(out_path: Path, *, fps: float, height: int, width: int, overwrite: bool):
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing video file: {out_path}. "
            "Pass --overwrite or choose a fresh --output_root."
        )
    backend, open_writer = _get_video_writer_backend(overwrite=overwrite)
    _ensure_dir(out_path.parent)
    writer = open_writer(str(out_path), float(fps), int(height), int(width))
    return backend, writer


def _write_mp4_from_hdf5_dataset(
    dset: h5py.Dataset,
    out_path: Path,
    *,
    fps: float,
    timestamps: Optional[np.ndarray] = None,
    timestamps_source: str = "assumed",
    demo_key: Optional[str] = None,
    chunk: int = 64,
    overwrite: bool = False,
) -> None:
    """
    Stream frames from an HDF5 dataset (T,H,W,3) uint8 to an MP4 file.
    """
    if dset.ndim != 4 or dset.shape[-1] != 3:
        raise ValueError(f"Expected image dataset of shape (T,H,W,3), got {dset.shape}")

    T, H, W, _C = dset.shape
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing video file: {out_path}. "
            "Pass --overwrite or choose a fresh --output_root."
        )
    backend, open_writer = _get_video_writer_backend(overwrite=overwrite)
    _ensure_dir(out_path.parent)

    writer = open_writer(str(out_path), float(fps), int(H), int(W))
    try:
        for s in range(0, int(T), int(chunk)):
            e = min(int(T), s + int(chunk))
            frames = dset[s:e]
            if frames.dtype != np.uint8:
                frames = frames.astype(np.uint8, copy=False)
            frames = np.ascontiguousarray(frames)
            for i in range(frames.shape[0]):
                writer.append(frames[i])
    finally:
        writer.close()

    sidecar: dict = {
        "video_backend": backend,
        "fps": float(fps),
        "frames": int(T),
        "height": int(H),
        "width": int(W),
        "timestamps_source": str(timestamps_source),
        "time_axis": {
            "type": "hdf5",
            "dataset_path": "/data/<demo_key>/timestamps",
        },
    }
    if demo_key is not None:
        sidecar["demo_key"] = str(demo_key)
        sidecar["timestamps_dataset"] = f"/data/{demo_key}/timestamps"
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape != (int(T),):
            raise ValueError(f"timestamps must have shape (T,), got {ts.shape} for T={T}")
        # We intentionally do not store per-frame timestamps here; the authoritative time axis
        # is written into the shard HDF5 as /data/<demo_key>/timestamps.
    _json_dump(out_path.with_suffix(".json"), sidecar)


def _append_hdf5_frames_to_writer(dset: h5py.Dataset, writer, *, chunk: int = 64) -> Tuple[int, int, int]:
    """
    Append all frames from a dataset (T,H,W,3) into an already-open writer.
    Returns (T, H, W).
    """
    if dset.ndim != 4 or dset.shape[-1] != 3:
        raise ValueError(f"Expected image dataset of shape (T,H,W,3), got {dset.shape}")
    T, H, W, _C = dset.shape
    for s in range(0, int(T), int(chunk)):
        e = min(int(T), s + int(chunk))
        frames = dset[s:e]
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)
        frames = np.ascontiguousarray(frames)
        # Fast-path for backends that can take a whole chunk at once (e.g., ffmpeg rawvideo pipe).
        append_many = getattr(writer, "append_many", None)
        if callable(append_many):
            append_many(frames)
        else:
            for i in range(frames.shape[0]):
                writer.append(frames[i])
    return int(T), int(H), int(W)


@dataclass
class PackedVideoState:
    cam: str
    shard_name: str
    out_path: Path
    backend: str
    writer: object
    fps: float
    height: int
    width: int
    frame_cursor: int
    segments: list[dict]

    def close_and_write_sidecar(self) -> None:
        self.writer.close()
        sidecar = {
            "video_backend": self.backend,
            "fps": float(self.fps),
            "height": int(self.height),
            "width": int(self.width),
            "frames": int(self.frame_cursor),
            "cam": self.cam,
            "shard": self.shard_name,
            # Timestamps are stored in the shard HDF5 under each demo group:
            #   /data/<demo_key>/timestamps
            # This keeps a single unified time axis for all modalities without bloating JSON.
            "time_axis": {
                "type": "hdf5",
                "dataset_path": "/data/<demo_key>/timestamps",
            },
            # Each segment carries boundaries; per-frame timestamps are only included when sourced.
            "segments": self.segments,
        }
        _json_dump(self.out_path.with_suffix(".json"), sidecar)


@dataclass
class DemoInfo:
    demo_key: str
    length: int
    timestamps_source: str


@dataclass(frozen=True)
class DemoRef:
    """
    Reference to a single demo inside a source HDF5 file.
    Used for mixed-task sharding across many source files.
    """

    source_path: Path
    source_file: str
    source_task_slug: str
    source_task: str
    source_demo_key: str
    out_demo_key: str


def split_hdf5_to_shards(
    input_path: Path,
    output_root: Path,
    *,
    demos_per_shard: int = 25,
    hz: float = 50.0,
    save_videos: bool = True,
    video_fps: Optional[float] = None,
    video_chunk: int = 64,
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
    overwrite: bool = False,
) -> dict:
    """
    Split one large HDF5 file into multiple smaller shard HDF5 files.
    Images (if present) are written as per-shard MP4 under videos_shards/.
    """
    if demos_per_shard <= 0:
        raise ValueError("demos_per_shard must be > 0")
    if hz <= 0:
        raise ValueError("hz must be > 0")

    if video_fps is None:
        video_fps = float(hz)

    filename = input_path.name
    task_slug = _task_slug_from_filename(filename)
    instruction = DEMO_FILE_TO_INSTRUCTION.get(filename, task_slug)

    task_dir = output_root / task_slug
    shards_dir = task_dir / "shards"
    videos_shards_dir = task_dir / "videos_shards"

    if overwrite and task_dir.exists():
        shutil.rmtree(task_dir)
    _ensure_dir(shards_dir)
    if save_videos:
        for cam in CAMERA_TO_DATASET.keys():
            _ensure_dir(videos_shards_dir / cam)

    if not overwrite:
        # Safety: if output already exists, stop early to prevent mixing old/new shards.
        existing = []
        if (task_dir / "demo_manifest.json").exists():
            existing.append(str(task_dir / "demo_manifest.json"))
        if list(shards_dir.glob("shard_*.hdf5")):
            existing.append(str(shards_dir / "shard_*.hdf5"))
        if save_videos:
            for cam in CAMERA_TO_DATASET.keys():
                if list((videos_shards_dir / cam).glob("shard_*.mp4")) or list((videos_shards_dir / cam).glob("shard_*.json")):
                    existing.append(str(videos_shards_dir / cam))
        if existing:
            raise FileExistsError(
                "Output directory already contains converted files; refusing to proceed without --overwrite.\n"
                f"  task_dir: {task_dir}\n"
                f"  found: {existing[:5]}{' ...' if len(existing) > 5 else ''}\n"
                "Pass --overwrite to delete existing shard/videos first, or choose a new --output_root."
            )

    # Shard-centric manifest builder:
    # shard_id -> {"hdf5": "...", "demos": [ {demo_key,length,...}, ... ], "videos": {cam:{mp4,sidecar}} }
    shard_records: dict[int, dict] = {}
    packed_states: Dict[str, Optional[PackedVideoState]] = {cam: None for cam in CAMERA_TO_DATASET.keys()}
    current_shard_idx: Optional[int] = None

    with h5py.File(str(input_path), "r") as fin:
        demo_keys = _iter_demo_keys(fin)
        n_demos = len(demo_keys)

        iterator: Iterable[Tuple[int, str]] = list(enumerate(demo_keys))
        if tqdm is not None:
            iterator = tqdm(iterator, total=n_demos, desc=f"Splitting {filename}", unit="demo")

        for idx, demo_key in iterator:
            shard_idx = idx // demos_per_shard
            shard_name = f"shard_{shard_idx:03d}.hdf5"
            shard_path = shards_dir / shard_name
            shard_rel = f"shards/{shard_name}"

            shard_rec = shard_records.get(shard_idx)
            if shard_rec is None:
                shard_rec = {
                    "shard_id": int(shard_idx),
                    "hdf5": shard_rel,
                    "demos": [],
                    "demo_keys": [],
                    "videos": {},
                }
                shard_records[shard_idx] = shard_rec

            demo = fin[f"data/{demo_key}"]
            payload = _read_numeric_demo(demo)
            act_len = int(payload["actions"].shape[0])
            # Use source timestamps if present; otherwise assume fixed-rate.
            if payload.get("timestamps", None) is not None:
                action_timestamps = np.asarray(payload["timestamps"]).astype(np.float64, copy=False)
                if action_timestamps.shape != (act_len,):
                    raise ValueError(
                        f"{filename} {demo_key}: timestamps shape {action_timestamps.shape} does not match actions length {act_len}"
                    )
                timestamps_source = str(payload.get("timestamps_source", "source"))
            else:
                action_timestamps = _make_timestamps(act_len, hz)
                timestamps_source = "assumed"

            # Videos (optional)
            if save_videos:
                # Rotate shard writers if shard changed
                if current_shard_idx is None:
                    current_shard_idx = shard_idx
                if shard_idx != current_shard_idx:
                    # Close previous shard writers and write sidecars
                    for cam in CAMERA_TO_DATASET.keys():
                        st = packed_states.get(cam)
                        if st is not None:
                            st.close_and_write_sidecar()
                            packed_states[cam] = None
                    current_shard_idx = shard_idx

                # Append this demo into shard video(s)
                for cam, dset_name in CAMERA_TO_DATASET.items():
                    if dset_name not in demo:
                        raise ValueError(
                            f"{filename} {demo_key}: missing image dataset '{dset_name}' required for per_shard videos"
                        )
                    dset = demo[dset_name]
                    img_len = int(dset.shape[0])
                    # In the unified interface, each modality may have a different time axis / length.
                    # We therefore do NOT require image length to match actions length.

                    st = packed_states.get(cam)
                    if st is None:
                        # Create new shard video for this cam
                        rel = f"videos_shards/{cam}/shard_{shard_idx:03d}.mp4"
                        out_path = task_dir / rel
                        backend, writer = _open_video_writer(
                            out_path,
                            fps=float(video_fps),
                            height=int(dset.shape[1]),
                            width=int(dset.shape[2]),
                            overwrite=bool(overwrite),
                        )
                        st = PackedVideoState(
                            cam=cam,
                            shard_name=f"shard_{shard_idx:03d}",
                            out_path=out_path,
                            backend=backend,
                            writer=writer,
                            fps=float(video_fps),
                            height=int(dset.shape[1]),
                            width=int(dset.shape[2]),
                            frame_cursor=0,
                            segments=[],
                        )
                        packed_states[cam] = st

                    # Ensure consistent resolution within shard
                    if int(dset.shape[1]) != st.height or int(dset.shape[2]) != st.width:
                        raise ValueError(
                            f"{filename} {demo_key}: resolution {dset.shape[1]}x{dset.shape[2]} "
                            f"differs from shard video {st.height}x{st.width} for cam={cam}"
                        )

                    frame_start = int(st.frame_cursor)
                    _T_written, _H, _W = _append_hdf5_frames_to_writer(dset, st.writer, chunk=int(video_chunk))
                    st.frame_cursor += int(img_len)
                    seg = {
                        "demo_key": demo_key,
                        "frame_start": int(frame_start),
                        "frame_count": int(img_len),
                        "timestamps_source": "assumed_video_fps",
                        # Modality-specific time axis lives in HDF5.
                        "timestamps_dataset": f"/data/{demo_key}/modalities/images/{cam}/timestamps",
                    }
                    st.segments.append(seg)
                    # Record per-shard video paths in the shard manifest (one MP4 + sidecar per camera).
                    mp4_rel = f"videos_shards/{cam}/shard_{shard_idx:03d}.mp4"
                    shard_rec["videos"][cam] = {"mp4": mp4_rel, "sidecar": mp4_rel.replace(".mp4", ".json")}

            # Append into shard HDF5 (open/close per demo for robustness)
            with h5py.File(str(shard_path), "a") as fout:
                fout.attrs["source_file"] = filename
                fout.attrs["task_instruction"] = instruction
                fout.attrs["timestamps_hz"] = float(hz)
                fout.attrs["video_fps"] = float(video_fps)
                fout.attrs["format_version"] = 1

                g = fout.require_group("data").require_group(demo_key)
                g.attrs["instruction"] = instruction
                g.attrs["length"] = int(act_len)

                def _write_dataset(parent: h5py.Group, name: str, arr: np.ndarray) -> None:
                    if name in parent:
                        del parent[name]
                    kwargs = {}
                    if compression is not None:
                        kwargs["compression"] = compression
                        kwargs["compression_opts"] = int(compression_level)
                    parent.create_dataset(name, data=arr, **kwargs)

                _write_dataset(g, "actions", payload["actions"].astype(np.float32, copy=False))
                obs = g.require_group("obs")
                _write_dataset(obs, "robot0_eef_pos", payload["obs/robot0_eef_pos"].astype(np.float32, copy=False))
                _write_dataset(obs, "robot0_eef_quat", payload["obs/robot0_eef_quat"].astype(np.float32, copy=False))
                _write_dataset(obs, "robot0_gripper_qpos", payload["obs/robot0_gripper_qpos"].astype(np.float32, copy=False))
                _write_dataset(obs, "robot0_joint_pos", payload["obs/robot0_joint_pos"].astype(np.float32, copy=False))
                _write_dataset(obs, "object", payload["obs/object"].astype(np.float32, copy=False))
                _write_dataset(obs, "object_rel_eef", payload["obs/object_rel_eef"].astype(np.float32, copy=False))
                # Backward-compat unified timestamp (actions time axis)
                _write_dataset(g, "timestamps", action_timestamps)
                g.attrs["timestamps_source"] = timestamps_source

                # ---- Unified interface: per-modality time axes ----
                mods = g.require_group("modalities")
                # actions
                m_act = mods.require_group("actions")
                _write_dataset(m_act, "timestamps", action_timestamps)
                m_act.attrs["timestamps_source"] = timestamps_source
                # Link to numeric data dataset without copying
                if "data" in m_act:
                    del m_act["data"]
                m_act["data"] = g["actions"]

                # robot_state and env_state get separate timestamps (may differ)
                m_robot = mods.require_group("robot_state")
                _write_dataset(m_robot, "timestamps", action_timestamps)
                m_robot.attrs["timestamps_source"] = timestamps_source
                for name in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]:
                    if name in m_robot:
                        del m_robot[name]
                m_robot["robot0_eef_pos"] = obs["robot0_eef_pos"]
                m_robot["robot0_eef_quat"] = obs["robot0_eef_quat"]
                m_robot["robot0_gripper_qpos"] = obs["robot0_gripper_qpos"]
                m_robot["robot0_joint_pos"] = obs["robot0_joint_pos"]

                m_env = mods.require_group("env_state")
                _write_dataset(m_env, "timestamps", action_timestamps)
                m_env.attrs["timestamps_source"] = timestamps_source
                if "object" in m_env:
                    del m_env["object"]
                m_env["object"] = obs["object"]
                if "object_rel_eef" in m_env:
                    del m_env["object_rel_eef"]
                m_env["object_rel_eef"] = obs["object_rel_eef"]

                # images (per camera): timestamps can differ from actions
                if save_videos:
                    m_img_root = mods.require_group("images")
                    for cam in CAMERA_TO_DATASET.keys():
                        m_cam = m_img_root.require_group(cam)
                        # For now we assume fixed fps for images; other converters can write source timestamps here.
                        # Length is inferred from the image dataset (if present).
                        if CAMERA_TO_DATASET[cam] in demo:
                            img_len = int(demo[CAMERA_TO_DATASET[cam]].shape[0])
                            img_ts = _make_timestamps(img_len, float(video_fps))
                            _write_dataset(m_cam, "timestamps", img_ts)
                            m_cam.attrs["timestamps_source"] = "assumed_video_fps"

            shard_rec["demo_keys"].append(demo_key)
            shard_rec["demos"].append(
                {
                    "demo_key": demo_key,
                    "length": int(act_len),
                    "timestamps_source": str(timestamps_source),
                    "modalities": {
                        "actions": {
                            "length": int(act_len),
                            "timestamps_dataset": f"/data/{demo_key}/modalities/actions/timestamps",
                            "data_dataset": f"/data/{demo_key}/actions",
                        },
                        "robot_state": {
                            "length": int(act_len),
                            "timestamps_dataset": f"/data/{demo_key}/modalities/robot_state/timestamps",
                            "data_group": "/data/<demo_key>/obs",
                        },
                        "env_state": {
                            "length": int(act_len),
                            "timestamps_dataset": f"/data/{demo_key}/modalities/env_state/timestamps",
                            "data_dataset": "/data/<demo_key>/obs/object",
                            "data_dataset_rel_eef": "/data/<demo_key>/obs/object_rel_eef",
                        },
                        **(
                            {
                                "images": {
                                    cam: {
                                        "frame_count": int(demo[CAMERA_TO_DATASET[cam]].shape[0]) if (CAMERA_TO_DATASET[cam] in demo) else None,
                                        "timestamps_dataset": f"/data/{demo_key}/modalities/images/{cam}/timestamps",
                                    }
                                    for cam in CAMERA_TO_DATASET.keys()
                                }
                            }
                            if save_videos
                            else {}
                        ),
                    },
                }
            )

    # Close any remaining packed shard writers
    if save_videos:
        for cam in CAMERA_TO_DATASET.keys():
            st = packed_states.get(cam)
            if st is not None:
                st.close_and_write_sidecar()
                packed_states[cam] = None

    shards_out = [shard_records[k] for k in sorted(shard_records.keys())]
    modalities_schema = {
        "actions": {
            "storage": "hdf5",
            "data_dataset": "/data/<demo_key>/actions",
            "timestamps_dataset": "/data/<demo_key>/modalities/actions/timestamps",
            "dtype": "float32",
            "timestamps_dtype": "float64",
            "timestamps_hz_if_assumed": float(hz),
        },
        "robot_state": {
            "storage": "hdf5",
            "timestamps_dataset": "/data/<demo_key>/modalities/robot_state/timestamps",
            "timestamps_dtype": "float64",
            "datasets": {
                "robot0_eef_pos": "/data/<demo_key>/obs/robot0_eef_pos",
                "robot0_eef_quat": "/data/<demo_key>/obs/robot0_eef_quat",
                "robot0_gripper_qpos": "/data/<demo_key>/obs/robot0_gripper_qpos",
                "robot0_joint_pos": "/data/<demo_key>/obs/robot0_joint_pos",
            },
            "note": "RobotDataset concatenates robot0_* into 16D robot_state.",
        },
        "env_state": {
            "storage": "hdf5",
            "timestamps_dataset": "/data/<demo_key>/modalities/env_state/timestamps",
            "timestamps_dtype": "float64",
            "datasets": {
                "object": "/data/<demo_key>/obs/object",
                "object_rel_eef": "/data/<demo_key>/obs/object_rel_eef",
            },
        },
    }
    if save_videos:
        modalities_schema["images"] = {
            "storage": "mp4_shard",
            "pack": "per_shard",
            "video_files_field": "videos",
            "segments_field": "segments",
            "segment_key_field": "demo_key",
            "segment_start_field": "frame_start",
            "segment_count_field": "frame_count",
            "timestamps_dataset": "/data/<demo_key>/modalities/images/<cam>/timestamps",
            "timestamps_dtype": "float64",
            "timestamps_hz_if_assumed": float(video_fps),
        }
    task_manifest = {
        "version": 4,
        "task": instruction,
        "task_slug": task_slug,
        "source_file": filename,
        "timestamps_hz": float(hz),
        "video_fps": float(video_fps),
        "demos_per_shard": int(demos_per_shard),
        "video_pack": "per_shard" if save_videos else None,
        "cameras": list(CAMERA_TO_DATASET.keys()),
        "camera_datasets": dict(CAMERA_TO_DATASET),
        "created_at_utc": _utc_now_iso(),
        "created_by": {"script": "scripts/policies/convert_hdf5_to_small_files.py"},
        "schema": {"modalities": modalities_schema},
        "shards": shards_out,
    }
    _json_dump(task_dir / "demo_manifest.json", task_manifest)
    return task_manifest


def _collect_demo_refs(input_paths: list[Path]) -> list[DemoRef]:
    refs: list[DemoRef] = []
    for input_path in input_paths:
        filename = input_path.name
        task_slug = _task_slug_from_filename(filename)
        instruction = DEMO_FILE_TO_INSTRUCTION.get(filename, task_slug)
        prefix = Path(filename).stem
        with h5py.File(str(input_path), "r") as fin:
            for demo_key in _iter_demo_keys(fin):
                out_demo_key = f"{prefix}__{demo_key}"
                refs.append(
                    DemoRef(
                        source_path=input_path,
                        source_file=filename,
                        source_task_slug=task_slug,
                        source_task=instruction,
                        source_demo_key=demo_key,
                        out_demo_key=out_demo_key,
                    )
                )
    return refs


def split_hdf5s_to_mixed_shards(
    input_paths: list[Path],
    output_root: Path,
    *,
    mixed_task_slug: str = "mixed",
    demos_per_shard: int = 25,
    hz: float = 50.0,
    save_videos: bool = True,
    video_fps: Optional[float] = None,
    video_chunk: int = 64,
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
    overwrite: bool = False,
    mix_seed: int = 0,
) -> dict:
    """
    Create a *single* task folder (mixed_task_slug) whose shards contain demos mixed across
    all input HDF5 files (tasks). This mitigates shard-level and worker-level task imbalance.
    """
    if demos_per_shard <= 0:
        raise ValueError("demos_per_shard must be > 0")
    if hz <= 0:
        raise ValueError("hz must be > 0")
    if video_fps is None:
        video_fps = float(hz)

    task_slug = str(mixed_task_slug)
    instruction = str(mixed_task_slug)

    task_dir = output_root / task_slug
    shards_dir = task_dir / "shards"
    videos_shards_dir = task_dir / "videos_shards"

    if overwrite and task_dir.exists():
        shutil.rmtree(task_dir)
    _ensure_dir(shards_dir)
    if save_videos:
        for cam in CAMERA_TO_DATASET.keys():
            _ensure_dir(videos_shards_dir / cam)

    if not overwrite:
        existing = []
        if (task_dir / "demo_manifest.json").exists():
            existing.append(str(task_dir / "demo_manifest.json"))
        if list(shards_dir.glob("shard_*.hdf5")):
            existing.append(str(shards_dir / "shard_*.hdf5"))
        if save_videos:
            for cam in CAMERA_TO_DATASET.keys():
                if list((videos_shards_dir / cam).glob("shard_*.mp4")) or list((videos_shards_dir / cam).glob("shard_*.json")):
                    existing.append(str(videos_shards_dir / cam))
        if existing:
            raise FileExistsError(
                "Output directory already contains converted files; refusing to proceed without --overwrite.\n"
                f"  task_dir: {task_dir}\n"
                f"  found: {existing[:5]}{' ...' if len(existing) > 5 else ''}\n"
                "Pass --overwrite to delete existing shard/videos first, or choose a new --output_root."
            )

    demo_refs = _collect_demo_refs(input_paths)
    if not demo_refs:
        raise ValueError("No demos found across input_paths")

    sources_summary: dict[str, dict] = {}
    task_counts: dict[str, int] = {}
    for r in demo_refs:
        task_counts[r.source_task_slug] = int(task_counts.get(r.source_task_slug, 0) + 1)
        s = sources_summary.get(r.source_file)
        if s is None:
            sources_summary[r.source_file] = {
                "source_file": r.source_file,
                "task": r.source_task,
                "task_slug": r.source_task_slug,
                "num_demos": 1,
            }
        else:
            s["num_demos"] = int(s.get("num_demos", 0) + 1)

    rng = random.Random(int(mix_seed))
    rng.shuffle(demo_refs)

    iterator: Iterable[Tuple[int, DemoRef]] = tqdm(
        list(enumerate(demo_refs)), total=len(demo_refs), desc="Mixing demos", unit="demo"
    )

    shard_records: dict[int, dict] = {}
    packed_states: Dict[str, Optional[PackedVideoState]] = {cam: None for cam in CAMERA_TO_DATASET.keys()}
    current_shard_idx: Optional[int] = None

    open_files: dict[Path, h5py.File] = {}
    try:
        for idx, ref in iterator:
            shard_idx = idx // int(demos_per_shard)
            shard_name = f"shard_{shard_idx:03d}.hdf5"
            shard_path = shards_dir / shard_name
            shard_rel = f"shards/{shard_name}"

            shard_rec = shard_records.get(shard_idx)
            if shard_rec is None:
                shard_rec = {"shard_id": int(shard_idx), "hdf5": shard_rel, "demos": [], "demo_keys": [], "videos": {}}
                shard_records[shard_idx] = shard_rec

            fin = open_files.get(ref.source_path)
            if fin is None:
                fin = h5py.File(str(ref.source_path), "r")
                open_files[ref.source_path] = fin

            demo = fin[f"data/{ref.source_demo_key}"]
            payload = _read_numeric_demo(demo)
            act_len = int(payload["actions"].shape[0])
            if payload.get("timestamps", None) is not None:
                action_timestamps = np.asarray(payload["timestamps"]).astype(np.float64, copy=False)
                if action_timestamps.shape != (act_len,):
                    raise ValueError(
                        f"{ref.source_file} {ref.source_demo_key}: timestamps shape {action_timestamps.shape} "
                        f"does not match actions length {act_len}"
                    )
                timestamps_source = str(payload.get("timestamps_source", "source"))
            else:
                action_timestamps = _make_timestamps(act_len, hz)
                timestamps_source = "assumed"

            if save_videos:
                if current_shard_idx is None:
                    current_shard_idx = shard_idx
                if shard_idx != current_shard_idx:
                    for cam in CAMERA_TO_DATASET.keys():
                        st = packed_states.get(cam)
                        if st is not None:
                            st.close_and_write_sidecar()
                            packed_states[cam] = None
                    current_shard_idx = shard_idx

                for cam, dset_name in CAMERA_TO_DATASET.items():
                    if dset_name not in demo:
                        raise ValueError(
                            f"{ref.source_file} {ref.source_demo_key}: missing image dataset '{dset_name}' required for mixed per_shard videos"
                        )
                    dset = demo[dset_name]
                    img_len = int(dset.shape[0])

                    st = packed_states.get(cam)
                    if st is None:
                        rel = f"videos_shards/{cam}/shard_{shard_idx:03d}.mp4"
                        out_path = task_dir / rel
                        backend, writer = _open_video_writer(
                            out_path,
                            fps=float(video_fps),
                            height=int(dset.shape[1]),
                            width=int(dset.shape[2]),
                            overwrite=bool(overwrite),
                        )
                        st = PackedVideoState(
                            cam=cam,
                            shard_name=f"shard_{shard_idx:03d}",
                            out_path=out_path,
                            backend=backend,
                            writer=writer,
                            fps=float(video_fps),
                            height=int(dset.shape[1]),
                            width=int(dset.shape[2]),
                            frame_cursor=0,
                            segments=[],
                        )
                        packed_states[cam] = st

                    if int(dset.shape[1]) != st.height or int(dset.shape[2]) != st.width:
                        raise ValueError(
                            f"{ref.source_file} {ref.source_demo_key}: resolution {dset.shape[1]}x{dset.shape[2]} "
                            f"differs from shard video {st.height}x{st.width} for cam={cam}"
                        )

                    frame_start = int(st.frame_cursor)
                    _T_written, _H, _W = _append_hdf5_frames_to_writer(dset, st.writer, chunk=int(video_chunk))
                    st.frame_cursor += int(img_len)
                    seg = {
                        "demo_key": ref.out_demo_key,
                        "frame_start": int(frame_start),
                        "frame_count": int(img_len),
                        "timestamps_source": "assumed_video_fps",
                        "timestamps_dataset": f"/data/{ref.out_demo_key}/modalities/images/{cam}/timestamps",
                        "source_file": ref.source_file,
                        "source_demo_key": ref.source_demo_key,
                        "task_slug": ref.source_task_slug,
                    }
                    st.segments.append(seg)
                    mp4_rel = f"videos_shards/{cam}/shard_{shard_idx:03d}.mp4"
                    shard_rec["videos"][cam] = {"mp4": mp4_rel, "sidecar": mp4_rel.replace(".mp4", ".json")}

            with h5py.File(str(shard_path), "a") as fout:
                fout.attrs["source_file"] = "mixed"
                fout.attrs["task_instruction"] = instruction
                fout.attrs["timestamps_hz"] = float(hz)
                fout.attrs["video_fps"] = float(video_fps)
                fout.attrs["format_version"] = 1

                g = fout.require_group("data").require_group(ref.out_demo_key)
                g.attrs["instruction"] = ref.source_task
                g.attrs["task_slug"] = ref.source_task_slug
                g.attrs["source_file"] = ref.source_file
                g.attrs["source_demo_key"] = ref.source_demo_key
                g.attrs["length"] = int(act_len)

                def _write_dataset(parent: h5py.Group, name: str, arr: np.ndarray) -> None:
                    if name in parent:
                        del parent[name]
                    kwargs = {}
                    if compression is not None:
                        kwargs["compression"] = compression
                        kwargs["compression_opts"] = int(compression_level)
                    parent.create_dataset(name, data=arr, **kwargs)

                _write_dataset(g, "actions", payload["actions"].astype(np.float32, copy=False))
                obs = g.require_group("obs")
                _write_dataset(obs, "robot0_eef_pos", payload["obs/robot0_eef_pos"].astype(np.float32, copy=False))
                _write_dataset(obs, "robot0_eef_quat", payload["obs/robot0_eef_quat"].astype(np.float32, copy=False))
                _write_dataset(obs, "robot0_gripper_qpos", payload["obs/robot0_gripper_qpos"].astype(np.float32, copy=False))
                _write_dataset(obs, "robot0_joint_pos", payload["obs/robot0_joint_pos"].astype(np.float32, copy=False))
                _write_dataset(obs, "object", payload["obs/object"].astype(np.float32, copy=False))
                _write_dataset(obs, "object_rel_eef", payload["obs/object_rel_eef"].astype(np.float32, copy=False))
                _write_dataset(g, "timestamps", action_timestamps)
                g.attrs["timestamps_source"] = timestamps_source

                mods = g.require_group("modalities")
                m_act = mods.require_group("actions")
                _write_dataset(m_act, "timestamps", action_timestamps)
                m_act.attrs["timestamps_source"] = timestamps_source
                if "data" in m_act:
                    del m_act["data"]
                m_act["data"] = g["actions"]

                m_robot = mods.require_group("robot_state")
                _write_dataset(m_robot, "timestamps", action_timestamps)
                m_robot.attrs["timestamps_source"] = timestamps_source
                for name in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]:
                    if name in m_robot:
                        del m_robot[name]
                m_robot["robot0_eef_pos"] = obs["robot0_eef_pos"]
                m_robot["robot0_eef_quat"] = obs["robot0_eef_quat"]
                m_robot["robot0_gripper_qpos"] = obs["robot0_gripper_qpos"]
                m_robot["robot0_joint_pos"] = obs["robot0_joint_pos"]

                m_env = mods.require_group("env_state")
                _write_dataset(m_env, "timestamps", action_timestamps)
                m_env.attrs["timestamps_source"] = timestamps_source
                if "object" in m_env:
                    del m_env["object"]
                m_env["object"] = obs["object"]
                if "object_rel_eef" in m_env:
                    del m_env["object_rel_eef"]
                m_env["object_rel_eef"] = obs["object_rel_eef"]

                if save_videos:
                    m_img_root = mods.require_group("images")
                    for cam in CAMERA_TO_DATASET.keys():
                        m_cam = m_img_root.require_group(cam)
                        if CAMERA_TO_DATASET[cam] in demo:
                            img_len = int(demo[CAMERA_TO_DATASET[cam]].shape[0])
                            img_ts = _make_timestamps(img_len, float(video_fps))
                            _write_dataset(m_cam, "timestamps", img_ts)
                            m_cam.attrs["timestamps_source"] = "assumed_video_fps"

            shard_rec["demo_keys"].append(ref.out_demo_key)
            shard_rec["demos"].append(
                {
                    "demo_key": ref.out_demo_key,
                    "length": int(act_len),
                    "timestamps_source": str(timestamps_source),
                    "task": ref.source_task,
                    "task_slug": ref.source_task_slug,
                    "source_file": ref.source_file,
                    "source_demo_key": ref.source_demo_key,
                    "modalities": {
                        "actions": {
                            "length": int(act_len),
                            "timestamps_dataset": f"/data/{ref.out_demo_key}/modalities/actions/timestamps",
                            "data_dataset": f"/data/{ref.out_demo_key}/actions",
                        },
                        "robot_state": {
                            "length": int(act_len),
                            "timestamps_dataset": f"/data/{ref.out_demo_key}/modalities/robot_state/timestamps",
                            "data_group": "/data/<demo_key>/obs",
                        },
                        "env_state": {
                            "length": int(act_len),
                            "timestamps_dataset": f"/data/{ref.out_demo_key}/modalities/env_state/timestamps",
                            "data_dataset": "/data/<demo_key>/obs/object",
                            "data_dataset_rel_eef": "/data/<demo_key>/obs/object_rel_eef",
                        },
                        **(
                            {
                                "images": {
                                    cam: {
                                        "frame_count": int(demo[CAMERA_TO_DATASET[cam]].shape[0]) if (CAMERA_TO_DATASET[cam] in demo) else None,
                                        "timestamps_dataset": f"/data/{ref.out_demo_key}/modalities/images/{cam}/timestamps",
                                    }
                                    for cam in CAMERA_TO_DATASET.keys()
                                }
                            }
                            if save_videos
                            else {}
                        ),
                    },
                }
            )
    finally:
        if save_videos:
            for cam in CAMERA_TO_DATASET.keys():
                st = packed_states.get(cam)
                if st is not None:
                    st.close_and_write_sidecar()
                    packed_states[cam] = None
        for f in open_files.values():
            f.close()

    shards_out = [shard_records[k] for k in sorted(shard_records.keys())]
    modalities_schema = {
        "actions": {
            "storage": "hdf5",
            "data_dataset": "/data/<demo_key>/actions",
            "timestamps_dataset": "/data/<demo_key>/modalities/actions/timestamps",
            "dtype": "float32",
            "timestamps_dtype": "float64",
            "timestamps_hz_if_assumed": float(hz),
        },
        "robot_state": {
            "storage": "hdf5",
            "timestamps_dataset": "/data/<demo_key>/modalities/robot_state/timestamps",
            "timestamps_dtype": "float64",
            "datasets": {
                "robot0_eef_pos": "/data/<demo_key>/obs/robot0_eef_pos",
                "robot0_eef_quat": "/data/<demo_key>/obs/robot0_eef_quat",
                "robot0_gripper_qpos": "/data/<demo_key>/obs/robot0_gripper_qpos",
                "robot0_joint_pos": "/data/<demo_key>/obs/robot0_joint_pos",
            },
            "note": "RobotDataset concatenates robot0_* into 16D robot_state.",
        },
        "env_state": {
            "storage": "hdf5",
            "timestamps_dataset": "/data/<demo_key>/modalities/env_state/timestamps",
            "timestamps_dtype": "float64",
            "datasets": {
                "object": "/data/<demo_key>/obs/object",
                "object_rel_eef": "/data/<demo_key>/obs/object_rel_eef",
            },
        },
    }
    if save_videos:
        modalities_schema["images"] = {
            "storage": "mp4_shard",
            "pack": "per_shard",
            "video_files_field": "videos",
            "segments_field": "segments",
            "segment_key_field": "demo_key",
            "segment_start_field": "frame_start",
            "segment_count_field": "frame_count",
            "timestamps_dataset": "/data/<demo_key>/modalities/images/<cam>/timestamps",
            "timestamps_dtype": "float64",
            "timestamps_hz_if_assumed": float(video_fps),
        }
    task_manifest = {
        "version": 4,
        "task": instruction,
        "task_slug": task_slug,
        "source_file": "mixed",
        "timestamps_hz": float(hz),
        "video_fps": float(video_fps),
        "demos_per_shard": int(demos_per_shard),
        "video_pack": "per_shard" if save_videos else None,
        "mix_seed": int(mix_seed),
        "sources": sorted(sources_summary.values(), key=lambda x: (x.get("task_slug", ""), x.get("source_file", ""))),
        "task_counts": dict(sorted(task_counts.items(), key=lambda kv: kv[0])),
        "demo_key_format": {"type": "prefixed", "pattern": "<source_file_stem>__<source_demo_key>", "delimiter": "__"},
        "cameras": list(CAMERA_TO_DATASET.keys()),
        "camera_datasets": dict(CAMERA_TO_DATASET),
        "created_at_utc": _utc_now_iso(),
        "created_by": {"script": "scripts/policies/convert_hdf5_to_small_files.py"},
        "schema": {"modalities": modalities_schema},
        "shards": shards_out,
    }
    _json_dump(task_dir / "demo_manifest.json", task_manifest)
    return task_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Split large robomimic HDF5 files into shards + MP4 videos.")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing input .hdf5 files")
    ap.add_argument("--output_root", type=str, required=True, help="Root directory for the output dataset")
    ap.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated list of filenames to process (default: all *.hdf5 in input_dir)",
    )
    ap.add_argument("--demos_per_shard", type=int, default=25, help="How many demos per shard HDF5 (default: 25)")
    ap.add_argument(
        "--timestamps_hz",
        dest="timestamps_hz",
        type=float,
        default=50.0,
        help=(
            "Assumed timestamp rate (Hz) for the SOURCE trajectories, used ONLY when source timestamps "
            "are not present in the input HDF5 demos. Default: 50."
            "Note: It is not related to the data. Data is stored in index format. It is only used to calculate the timestamps."
        ),
    )
    ap.add_argument(
        "--no_images",
        action="store_true",
        help="Skip image/video modalities entirely (no MP4s and removes images from manifests).",
    )
    ap.add_argument("--video_fps", type=float, default=None, help="FPS to encode MP4 videos (default: hz)")
    ap.add_argument(
        "--video_read_chunk_frames",
        dest="video_read_chunk_frames",
        type=int,
        default=64,
        help=(
            "How many frames to read from the SOURCE HDF5 per chunk when encoding MP4 videos. "
            "This is a performance/memory knob only (does not change timing). Default: 64."
        ),
    )

    ap.add_argument(
        "--hdf5_compression",
        dest="hdf5_compression",
        type=str,
        default="gzip",
        help="HDF5 compression for numeric shard datasets (gzip/lzf/None). Default: gzip",
    )
    ap.add_argument(
        "--hdf5_compression_level",
        dest="hdf5_compression_level",
        type=int,
        default=4,
        help="HDF5 gzip compression level for numeric shard datasets (default: 4)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete all existing files and folders under --output_root before converting.",
    )
    ap.add_argument(
        "--mix_tasks",
        action="store_true",
        help="If set, create a single mixed-task dataset under <output_root>/mixed/ with shards containing demos from all input files. Otherwise, create one dataset per input file.",
    )
    ap.add_argument("--mixed_task_slug", type=str, default="mixed", help="Folder name for mixed dataset (default: mixed)")
    ap.add_argument("--mix_seed", type=int, default=0, help="RNG seed for mixing demos across tasks (default: 0)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    if input_dir.resolve() == output_root.resolve():
        raise ValueError("--output_root must be different from --input_dir when using this converter.")
    _ensure_dir(output_root)
    if bool(args.overwrite):
        _clear_directory_contents(output_root)
    if (output_root / "dataset_manifest.json").exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Refusing to overwrite existing dataset manifest: {output_root / 'dataset_manifest.json'}. "
            "Pass --overwrite or choose a fresh --output_root."
        )

    if args.files is None:
        files = sorted([p.name for p in input_dir.glob("*.hdf5")])
    else:
        files = [x.strip() for x in args.files.split(",") if x.strip()]

    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {input_dir}")

    compression = args.hdf5_compression
    if compression is not None and compression.lower() in {"none", "null", "false"}:
        compression = None

    print("=" * 80)
    print("Convert HDF5 -> shard HDF5s + MP4 videos")
    print("=" * 80)
    print(f"Input dir: {input_dir}")
    print(f"Output root: {output_root}")
    print(f"Files: {files}")
    print(f"demos_per_shard: {args.demos_per_shard}")
    print(f"timestamps_hz: {args.timestamps_hz} (used only if source timestamps missing)")
    save_videos = not bool(args.no_images)
    print(f"images/videos: {save_videos} (encode_fps={args.video_fps or args.timestamps_hz})")
    print(f"video_read_chunk_frames: {args.video_read_chunk_frames}")
    print(f"hdf5_compression: {compression} (level={args.hdf5_compression_level})")
    print(f"mix_tasks: {bool(args.mix_tasks)} (mixed_task_slug={args.mixed_task_slug}, mix_seed={args.mix_seed})")
    print("=" * 80)

    tasks: list[dict] = []
    if args.mix_tasks:
        input_paths = [input_dir / fn for fn in files]
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing input file: {p}")
        task_manifest = split_hdf5s_to_mixed_shards(
            input_paths,
            output_root,
            mixed_task_slug=str(args.mixed_task_slug),
            demos_per_shard=int(args.demos_per_shard),
            hz=float(args.timestamps_hz),
            save_videos=bool(save_videos),
            video_fps=None if args.video_fps is None else float(args.video_fps),
            video_chunk=int(args.video_read_chunk_frames),
            compression=compression,
            compression_level=int(args.hdf5_compression_level),
            overwrite=bool(args.overwrite),
            mix_seed=int(args.mix_seed),
        )
        tasks.append(
            {
                "task": task_manifest.get("task"),
                "task_slug": task_manifest.get("task_slug"),
                "source_file": task_manifest.get("source_file"),
                "manifest": f"{task_manifest.get('task_slug')}/demo_manifest.json",
            }
        )
    else:
        for fn in files:
            in_path = input_dir / fn
            if not in_path.exists():
                raise FileNotFoundError(f"Missing input file: {in_path}")
            task_manifest = split_hdf5_to_shards(
                in_path,
                output_root,
                demos_per_shard=int(args.demos_per_shard),
                hz=float(args.timestamps_hz),
                save_videos=bool(save_videos),
                video_fps=None if args.video_fps is None else float(args.video_fps),
                video_chunk=int(args.video_read_chunk_frames),
                compression=compression,
                compression_level=int(args.hdf5_compression_level),
                overwrite=bool(args.overwrite),
            )
            tasks.append(
                {
                    "task": task_manifest.get("task"),
                    "task_slug": task_manifest.get("task_slug"),
                    "source_file": task_manifest.get("source_file"),
                    "manifest": f"{task_manifest.get('task_slug')}/demo_manifest.json",
                }
            )

    dataset_manifest = {
        "version": 1,
        "format": "small-files",
        "created_at_utc": _utc_now_iso(),
        "created_by": {"script": "scripts/policies/convert_hdf5_to_small_files.py"},
        "tasks": sorted(tasks, key=lambda x: x.get("task_slug", "")),
    }
    _json_dump(output_root / "dataset_manifest.json", dataset_manifest)
    print("\nDone.")


if __name__ == "__main__":
    main()

