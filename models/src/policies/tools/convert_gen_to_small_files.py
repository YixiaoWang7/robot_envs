"""
Convert a *folder-based* generated dataset into the \"small-files\" format consumed by:
  `src/policies/datasets/robot_datasetv2.py`

Source layout (this is what you have under /mnt/ssd/.../task):

  <input_root>/<task_slug>/
    demo_000000/
      demo.hdf5
      agentview.mp4
      robot0_eye_in_hand.mp4
    demo_000001/
      ...

Where `demo.hdf5` contains:
  /actions (T,7)
  /obs/*  (T, D)

Target layout (per task OR one mixed folder when --mix_tasks):

  <output_root>/<task_slug>/
    demo_manifest.json
    shards/shard_000.hdf5
    videos_shards/<cam>/shard_000.mp4
    videos_shards/<cam>/shard_000.json

Notes:
  - Videos must be \"per_shard\" for RobotDatasetV2, so we concatenate per-demo MP4s.
  - Numeric data is copied efficiently using HDF5 group copy (no full array materialization).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


CAMERAS_DEFAULT = ("agentview", "robot0_eye_in_hand")
CAMERA_TO_DEMO_MP4_NAME: dict[str, str] = {
    "agentview": "agentview.mp4",
    "robot0_eye_in_hand": "robot0_eye_in_hand.mp4",
}


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

def _infer_instruction_from_task_slug(task_slug: str) -> str:
    s = str(task_slug).strip().lower()
    if "_into_" in s:
        obj, cont = s.split("_into_", 1)
        obj = obj.replace("_", " ").strip()
        cont = cont.replace("_", " ").strip()
        if obj and cont:
            return f"place the {obj} into the {cont}"
    return str(task_slug)


def _make_timestamps(length: int, hz: float) -> np.ndarray:
    return np.arange(int(length), dtype=np.float64) / float(hz)


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


def _ffprobe_stream_info(mp4_path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe not found on PATH; required for MP4 probing in this script.")
    cmd = [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(mp4_path),
    ]
    p = subprocess.run(cmd, check=True, capture_output=True)
    info = json.loads(p.stdout.decode("utf-8"))
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no streams for: {mp4_path}")
    st = streams[0]

    def _parse_ratio(x: object) -> Optional[float]:
        if x is None:
            return None
        s = str(x)
        if "/" in s:
            a, b = s.split("/", 1)
            try:
                num = float(a)
                den = float(b)
                return None if den == 0 else (num / den)
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    fps = _parse_ratio(st.get("avg_frame_rate")) or _parse_ratio(st.get("r_frame_rate"))
    frames = st.get("nb_read_frames") or st.get("nb_frames")
    try:
        frames_i = int(frames) if frames is not None else None
    except Exception:
        frames_i = None
    return {
        "width": int(st.get("width", 0) or 0),
        "height": int(st.get("height", 0) or 0),
        "fps": None if fps is None else float(fps),
        "frames": frames_i,
    }


def _write_concat_mp4(
    *,
    mp4_paths: list[Path],
    out_path: Path,
    overwrite: bool,
    reencode_fps: Optional[float],
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required for MP4 writing in this script.")
    if not mp4_paths:
        raise ValueError("_write_concat_mp4 requires at least one input mp4")
    _ensure_dir(out_path.parent)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing video file: {out_path}. "
            "Pass --overwrite or choose a fresh --output_root."
        )

    # Fast-path: if there's only one input, avoid ffmpeg entirely.
    # This matters a lot when demos_per_shard == 1 (otherwise we'd spawn ffmpeg per demo).
    if len(mp4_paths) == 1:
        src = mp4_paths[0]
        if out_path.exists() and overwrite:
            out_path.unlink()
        try:
            os.link(src, out_path)  # O(1) when on same filesystem
        except Exception:
            shutil.copy2(src, out_path)
        return

    list_path = out_path.with_suffix(".concat.txt")
    content = "".join([f"file '{p.as_posix()}'\n" for p in mp4_paths])
    list_path.write_text(content, encoding="utf-8")
    try:
        # Try stream copy first (fast, no re-encode)
        cmd_copy = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(out_path),
        ]
        p = subprocess.run(cmd_copy, capture_output=True)
        if p.returncode == 0:
            return

        # Fallback: re-encode to H.264 for compatibility
        cmd_enc = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
        if reencode_fps is not None:
            cmd_enc += ["-r", str(float(reencode_fps))]
        cmd_enc += ["-movflags", "+faststart", str(out_path)]
        subprocess.run(cmd_enc, check=True)
    finally:
        try:
            list_path.unlink()
        except Exception:
            pass


@dataclass(frozen=True)
class DemoSource:
    task_slug: str
    demo_key: str
    demo_dir: Path
    hdf5_path: Path

    def mp4_path(self, cam: str) -> Path:
        return self.demo_dir / CAMERA_TO_DEMO_MP4_NAME[str(cam)]


def _discover_task_demos(input_root: Path, *, task_slug: str) -> list[DemoSource]:
    task_dir = input_root / task_slug
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    demos: list[DemoSource] = []
    for d in sorted(task_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("demo_")):
            continue
        h5 = d / "demo.hdf5"
        if not h5.exists():
            continue
        demos.append(DemoSource(task_slug=task_slug, demo_key=d.name, demo_dir=d, hdf5_path=h5))
    if not demos:
        raise ValueError(f"No demos found under: {task_dir}")
    return demos


def _discover_all_tasks(input_root: Path, *, allow: Optional[set[str]]) -> list[str]:
    tasks = []
    for p in sorted(input_root.iterdir()):
        if not p.is_dir():
            continue
        if allow is not None and p.name not in allow:
            continue
        # treat as task if it contains at least one demo_*/demo.hdf5
        if any((c.is_dir() and c.name.startswith("demo_") and (c / "demo.hdf5").exists()) for c in p.iterdir()):
            tasks.append(p.name)
    if not tasks:
        raise ValueError(f"No task folders discovered under: {input_root}")
    return tasks


def _copy_obs_group(fin: h5py.File, dest_demo: h5py.Group) -> None:
    if "obs" not in fin:
        return
    if "obs" in dest_demo:
        del dest_demo["obs"]
    fin.copy("obs", dest_demo, name="obs")


def _ensure_object_rel_eef(dest_demo: h5py.Group) -> None:
    # Compute obs/object_rel_eef when obs/object is a packed pose array (T, 7*N)
    try:
        obs = dest_demo["obs"]
    except Exception:
        return
    if "object" not in obs:
        return
    if "robot0_eef_pos" not in obs or "robot0_eef_quat" not in obs:
        return
    obj = obs["object"][()]
    if obj.ndim != 2 or (obj.shape[1] % 7) != 0:
        return
    eef_pos = obs["robot0_eef_pos"][()].astype(np.float32, copy=False)
    eef_quat = obs["robot0_eef_quat"][()].astype(np.float32, copy=False)
    obj = obj.astype(np.float32, copy=False)
    rel = _object_world_to_eef_relative(eef_pos=eef_pos, eef_quat=eef_quat, objects=obj)
    if "object_rel_eef" in obs:
        del obs["object_rel_eef"]
    obs.create_dataset("object_rel_eef", data=rel)


def _write_dataset(group: h5py.Group, name: str, arr: np.ndarray, *, compression: Optional[str], compression_level: int) -> None:
    if name in group:
        del group[name]
    kwargs = {}
    if compression is not None:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = int(compression_level)
    group.create_dataset(name, data=arr, **kwargs)


def _write_timestamps_and_modalities(
    demo: h5py.Group,
    *,
    act_len: int,
    hz: float,
    video_fps: float,
    cameras: tuple[str, ...],
    image_lengths: Optional[dict[str, int]],
    compression: Optional[str],
    compression_level: int,
) -> None:
    action_ts = _make_timestamps(act_len, hz)
    _write_dataset(demo, "timestamps", action_ts, compression=compression, compression_level=compression_level)
    demo.attrs["timestamps_source"] = "assumed"

    mods = demo.require_group("modalities")
    m_act = mods.require_group("actions")
    _write_dataset(m_act, "timestamps", action_ts, compression=compression, compression_level=compression_level)
    m_act.attrs["timestamps_source"] = "assumed"
    if "data" in m_act:
        del m_act["data"]
    m_act["data"] = demo["actions"]

    # For this dataset, we align robot_state/env_state timestamps with actions.
    m_robot = mods.require_group("robot_state")
    _write_dataset(m_robot, "timestamps", action_ts, compression=compression, compression_level=compression_level)
    m_robot.attrs["timestamps_source"] = "assumed"

    m_env = mods.require_group("env_state")
    _write_dataset(m_env, "timestamps", action_ts, compression=compression, compression_level=compression_level)
    m_env.attrs["timestamps_source"] = "assumed"

    # Images: assumed fixed video_fps. Length can be read from MP4 probing results.
    if image_lengths is not None:
        m_img_root = mods.require_group("images")
        for cam in cameras:
            m_cam = m_img_root.require_group(cam)
            n = int(image_lengths.get(cam, 0))
            img_ts = _make_timestamps(n, float(video_fps))
            _write_dataset(m_cam, "timestamps", img_ts, compression=compression, compression_level=compression_level)
            m_cam.attrs["timestamps_source"] = "assumed_video_fps"


def _convert_task_sources_to_shards(
    sources: list[DemoSource],
    *,
    output_root: Path,
    out_task_slug: str,
    out_task_name: str,
    demos_per_shard: int,
    timestamps_hz: float,
    save_videos: bool,
    video_fps: Optional[float],
    cameras: tuple[str, ...],
    overwrite: bool,
    compression: Optional[str],
    compression_level: int,
    max_demos: Optional[int],
) -> dict:
    task_dir = output_root / out_task_slug
    shards_dir = task_dir / "shards"
    videos_dir = task_dir / "videos_shards"

    if overwrite and task_dir.exists():
        shutil.rmtree(task_dir)
    _ensure_dir(shards_dir)
    if save_videos:
        for cam in cameras:
            _ensure_dir(videos_dir / cam)

    if max_demos is not None:
        sources = sources[: int(max_demos)]
    if not sources:
        raise ValueError("No demos to convert (after max_demos filter).")

    # Choose a stable encode fps. If not provided, inherit from the first video when possible.
    encode_fps = float(video_fps) if video_fps is not None else float(timestamps_hz)
    if save_videos and video_fps is None:
        # Best effort: use the first demo's agentview fps if available.
        try:
            info0 = _ffprobe_stream_info(sources[0].mp4_path(cameras[0]))
            if info0.get("fps") is not None:
                encode_fps = float(info0["fps"])
        except Exception:
            encode_fps = float(timestamps_hz)

    shard_records: dict[int, dict] = {}

    iterator: Iterable[Tuple[int, DemoSource]] = list(enumerate(sources))
    iterator = tqdm(iterator, total=len(sources), desc=f"Convert {out_task_slug}", unit="demo")

    # Numeric conversion
    for idx, src in iterator:
        shard_idx = idx // int(demos_per_shard)
        shard_name = f"shard_{shard_idx:03d}.hdf5"
        shard_path = shards_dir / shard_name
        shard_rel = f"shards/{shard_name}"
        shard_rec = shard_records.get(shard_idx)
        if shard_rec is None:
            shard_rec = {"shard_id": int(shard_idx), "hdf5": shard_rel, "demos": [], "demo_keys": [], "videos": {}}
            shard_records[shard_idx] = shard_rec

        with h5py.File(str(src.hdf5_path), "r") as fin, h5py.File(str(shard_path), "a") as fout:
            fout.attrs["source"] = "folder_dataset"
            fout.attrs["task_instruction"] = str(out_task_name)
            fout.attrs["timestamps_hz"] = float(timestamps_hz)
            fout.attrs["video_fps"] = float(encode_fps)
            fout.attrs["format_version"] = 2

            g = fout.require_group("data").require_group(str(src.demo_key))
            g.attrs["instruction"] = str(out_task_name)
            g.attrs["task_slug"] = str(out_task_slug)

            # Copy actions and obs efficiently.
            if "actions" in g:
                del g["actions"]
            fin.copy("actions", g, name="actions")
            _copy_obs_group(fin, g)

            act_len = int(g["actions"].shape[0])
            g.attrs["length"] = int(act_len)

            # Compute object_rel_eef when possible.
            _ensure_object_rel_eef(g)

            # Optional: per-demo frame counts for per-modality timestamps.
            image_lengths: Optional[dict[str, int]] = None
            if save_videos:
                image_lengths = {}
                for cam in cameras:
                    mp4 = src.mp4_path(cam)
                    try:
                        info = _ffprobe_stream_info(mp4)
                        n = info.get("frames")
                        image_lengths[cam] = int(n) if n is not None else int(act_len)
                    except Exception:
                        image_lengths[cam] = int(act_len)

            _write_timestamps_and_modalities(
                g,
                act_len=int(act_len),
                hz=float(timestamps_hz),
                video_fps=float(encode_fps),
                cameras=cameras,
                image_lengths=image_lengths,
                compression=compression,
                compression_level=int(compression_level),
            )

        shard_rec["demo_keys"].append(str(src.demo_key))
        shard_rec["demos"].append(
            {
                "demo_key": str(src.demo_key),
                "length": int(act_len),
                "timestamps_source": "assumed",
                "task": str(out_task_name),
                "task_slug": str(out_task_slug),
            }
        )

    # Video concatenation + sidecars
    if save_videos:
        shard_items = sorted(shard_records.items(), key=lambda kv: kv[0])
        for shard_idx, shard_rec in tqdm(shard_items, desc=f"Videos {out_task_slug}", unit="shard"):
            # demos in this shard in order
            s = int(shard_idx) * int(demos_per_shard)
            e = min(len(sources), s + int(demos_per_shard))
            shard_sources = sources[s:e]
            for cam in cameras:
                mp4_rel = f"videos_shards/{cam}/shard_{shard_idx:03d}.mp4"
                out_mp4 = task_dir / mp4_rel
                in_mp4s = [ds.mp4_path(cam) for ds in shard_sources]
                _write_concat_mp4(mp4_paths=in_mp4s, out_path=out_mp4, overwrite=bool(overwrite), reencode_fps=float(encode_fps))

                # Build sidecar segments (frame_start based on per-demo frame_count)
                segments = []
                cursor = 0
                for ds in shard_sources:
                    act_len = None
                    try:
                        with h5py.File(str(ds.hdf5_path), "r") as f:
                            act_len = int(f["actions"].shape[0])
                    except Exception:
                        act_len = None
                    try:
                        info = _ffprobe_stream_info(ds.mp4_path(cam))
                        frame_count = int(info["frames"]) if info.get("frames") is not None else int(act_len or 0)
                    except Exception:
                        frame_count = int(act_len or 0)
                    segments.append(
                        {
                            "demo_key": str(ds.demo_key),
                            "frame_start": int(cursor),
                            "frame_count": int(frame_count),
                            "timestamps_source": "assumed_video_fps",
                            "timestamps_dataset": f"/data/{ds.demo_key}/modalities/images/{cam}/timestamps",
                        }
                    )
                    cursor += int(frame_count)

                sidecar = {
                    "video_backend": "ffmpeg_concat",
                    "fps": float(encode_fps),
                    "frames": int(cursor),
                    "cam": str(cam),
                    "shard": f"shard_{shard_idx:03d}",
                    "segments": segments,
                    "time_axis": {"type": "hdf5", "dataset_path": "/data/<demo_key>/timestamps"},
                }
                sidecar_rel = mp4_rel.replace(".mp4", ".json")
                _json_dump(task_dir / sidecar_rel, sidecar)
                shard_rec["videos"][cam] = {"mp4": mp4_rel, "sidecar": sidecar_rel}

    shards_out = [shard_records[k] for k in sorted(shard_records.keys())]
    task_manifest = {
        "version": 4,
        "task": str(out_task_name),
        "task_slug": str(out_task_slug),
        "source": "folder_dataset",
        "timestamps_hz": float(timestamps_hz),
        "video_fps": float(encode_fps),
        "demos_per_shard": int(demos_per_shard),
        "video_pack": "per_shard" if save_videos else None,
        "cameras": list(cameras),
        "created_at_utc": _utc_now_iso(),
        "created_by": {"script": "src/policies/tools/convert_gen_to_small_files.py"},
        "shards": shards_out,
    }
    _json_dump(task_dir / "demo_manifest.json", task_manifest)
    return task_manifest
def main() -> None:
    ap = argparse.ArgumentParser(description="Convert folder-based demos into small-files (datasetv2) format.")
    ap.add_argument("--input_root", type=str, required=True, help="Folder containing task subfolders (e.g. .../task)")
    ap.add_argument("--output_root", type=str, required=True, help="Root directory for the output dataset")
    ap.add_argument(
        "--task_slugs",
        type=str,
        default=None,
        help="Comma-separated list of task folder names to process (default: auto-discover all tasks)",
    )
    ap.add_argument("--demos_per_shard", type=int, default=25, help="How many demos per shard HDF5 (default: 25)")
    ap.add_argument(
        "--timestamps_hz", dest="timestamps_hz",
        type=float,
        default=50.0,
        help=(
            "Assumed timestamp rate (Hz) for the SOURCE trajectories, used ONLY when source timestamps "
            "are not present in the input HDF5 demos. Default: 50."
            "Note: It is not related to the data. Data is stored in index format. It is only used to calculate the timestamps."
        ),
    )
    ap.add_argument(
        "--no_images", action="store_true", help="Skip image/video modalities entirely (no MP4 concat, no sidecars)."
    )
    ap.add_argument("--video_fps", type=float, default=None, help="FPS to use when re-encoding shard videos (default: inherit or hz)")

    ap.add_argument(
        "--hdf5_compression", dest="hdf5_compression",
        type=str,
        default="gzip",
        help="HDF5 compression for numeric shard datasets (gzip/lzf/None). Default: gzip",
    )
    ap.add_argument(
        "--hdf5_compression_level", dest="hdf5_compression_level",
        type=int,
        default=4,
        help="HDF5 gzip compression level for numeric shard datasets (default: 4)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Delete all existing output task folders before converting.",
    )
    ap.add_argument(
        "--mix_tasks",
        action="store_true",
        help="If set, create a single mixed-task dataset under <output_root>/mixed/ with demos from all tasks.",
    )
    ap.add_argument("--mixed_task_slug", type=str, default="mixed", help="Folder name for mixed dataset (default: mixed)")
    ap.add_argument("--mix_seed", type=int, default=0, help="RNG seed for mixing demos across tasks (default: 0)")
    ap.add_argument("--max_demos_per_task", type=int, default=None, help="Optional limit for quick tests (per task)")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("--output_root must be different from --input_root.")
    _ensure_dir(output_root)
    if (output_root / "dataset_manifest.json").exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Refusing to overwrite existing dataset manifest: {output_root / 'dataset_manifest.json'}. "
            "Pass --overwrite or choose a fresh --output_root."
        )

    allow = None
    if args.task_slugs is not None:
        allow = {x.strip() for x in str(args.task_slugs).split(",") if x.strip()}
    task_slugs = _discover_all_tasks(input_root, allow=allow)

    compression = args.hdf5_compression
    if compression is not None and compression.lower() in {"none", "null", "false"}:
        compression = None

    print("=" * 80)
    print("Convert folder dataset -> shard HDF5s + shard MP4 videos")
    print("=" * 80)
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Tasks: {task_slugs}")
    print(f"demos_per_shard: {args.demos_per_shard}")
    print(f"timestamps_hz: {args.timestamps_hz} (used only if source timestamps missing)")
    save_videos = not bool(args.no_images)
    print(f"images/videos: {save_videos} (encode_fps={args.video_fps or 'inherit-or-hz'})")
    print(f"hdf5_compression: {compression} (level={args.hdf5_compression_level})")
    print(f"mix_tasks: {bool(args.mix_tasks)} (mixed_task_slug={args.mixed_task_slug}, mix_seed={args.mix_seed})")
    print(f"max_demos_per_task: {args.max_demos_per_task}")
    print("=" * 80)

    dataset_tasks: list[dict] = []
    if bool(args.mix_tasks):
        all_sources: list[DemoSource] = []
        for ts in task_slugs:
            all_sources.extend(_discover_task_demos(input_root, task_slug=str(ts)))
        rng = random.Random(int(args.mix_seed))
        rng.shuffle(all_sources)
        out_slug = str(args.mixed_task_slug)
        out_task = str(args.mixed_task_slug)
        _convert_task_sources_to_shards(
            all_sources,
            output_root=output_root,
            out_task_slug=out_slug,
            out_task_name=out_task,
            demos_per_shard=int(args.demos_per_shard),
            timestamps_hz=float(args.timestamps_hz),
            save_videos=bool(save_videos),
            video_fps=None if args.video_fps is None else float(args.video_fps),
            cameras=tuple(CAMERAS_DEFAULT),
            overwrite=bool(args.overwrite),
            compression=compression,
            compression_level=int(args.hdf5_compression_level),
            max_demos=None if args.max_demos_per_task is None else int(args.max_demos_per_task),
        )
        dataset_tasks.append({"task": out_task, "task_slug": out_slug, "manifest": f"{out_slug}/demo_manifest.json"})
    else:
        for ts in task_slugs:
            srcs = _discover_task_demos(input_root, task_slug=str(ts))
            out_slug = str(ts)
            out_task = _infer_instruction_from_task_slug(out_slug)
            _convert_task_sources_to_shards(
                srcs,
                output_root=output_root,
                out_task_slug=out_slug,
                out_task_name=out_task,
                demos_per_shard=int(args.demos_per_shard),
                timestamps_hz=float(args.timestamps_hz),
                save_videos=bool(save_videos),
                video_fps=None if args.video_fps is None else float(args.video_fps),
                cameras=tuple(CAMERAS_DEFAULT),
                overwrite=bool(args.overwrite),
                compression=compression,
                compression_level=int(args.hdf5_compression_level),
                max_demos=None if args.max_demos_per_task is None else int(args.max_demos_per_task),
            )
            dataset_tasks.append({"task": out_task, "task_slug": out_slug, "manifest": f"{out_slug}/demo_manifest.json"})

    dataset_manifest = {
        "version": 1,
        "format": "small-files",
        "created_at_utc": _utc_now_iso(),
        "created_by": {"script": "src/policies/tools/convert_gen_to_small_files.py"},
        "tasks": sorted(dataset_tasks, key=lambda x: x.get("task_slug", "")),
    }
    _json_dump(output_root / "dataset_manifest.json", dataset_manifest)
    print("\nDone.")


if __name__ == "__main__":
    main()

