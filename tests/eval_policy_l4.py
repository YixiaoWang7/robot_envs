#!/usr/bin/env python
"""
Multi-episode policy evaluation on CG_L4.

Loads a RobotFlowPolicyWrapper checkpoint and runs MPC inference.

Outputs (under --out-dir / eval_{timestamp}/):
  videos/ep{i:03d}_{Success|Failure}_{task}.mp4   (up to --max-videos)
  cross_attn_videos/ep{i:03d}_{Success|Failure}_{task}_crossattn.mp4   (optional)
  rollouts/ep{i:03d}.npz                           (actions, rewards, success)
  eval_summary.json                                 (aggregated + per-episode + per-task metrics)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from CG_L2_image_wrapper import ImageBasedCGWrapper
from CG_L2_state_wrapper import StateBasedCGWrapper

# CG_L4 task vocabulary
_L4_OBJECTS = ("cross", "cube", "cylinder", "milk")
_L4_CONTAINERS = ("bin", "mug", "plate", "mug_no_handle")
_ALL_TASKS = [f"place the {o} into the {c}" for o in _L4_OBJECTS for c in _L4_CONTAINERS]


@dataclass(frozen=True)
class EvalSeedPlan:
    root_seed: int
    global_seed: int
    episode_task_seeds: list[int]
    episode_reset_seeds: list[int]
    batch_policy_seeds: list[int]


def _seedseq_to_int(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _build_seed_plan(*, root_seed: int, n_episodes: int, num_envs: int) -> EvalSeedPlan:
    n_batches = math.ceil(n_episodes / max(num_envs, 1))
    root = np.random.SeedSequence(int(root_seed))
    global_seq, task_root, reset_root, policy_root = root.spawn(4)
    return EvalSeedPlan(
        root_seed=int(root_seed),
        global_seed=_seedseq_to_int(global_seq),
        episode_task_seeds=[_seedseq_to_int(seq) for seq in task_root.spawn(n_episodes)],
        episode_reset_seeds=[_seedseq_to_int(seq) for seq in reset_root.spawn(n_episodes)],
        batch_policy_seeds=[_seedseq_to_int(seq) for seq in policy_root.spawn(n_batches)],
    )


def _set_global_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch  # type: ignore
    except Exception:
        return

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _set_torch_seed(seed: int) -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_episode_tasks_full_grid(*, n_trials_per_task: int) -> list[str]:
    """
    Deterministic evaluation schedule: full 4×4 task grid, repeated N times.

    Order is stable across runs to make comparisons easier:
      objects major, containers minor (same as `_ALL_TASKS`).
    """
    n = int(n_trials_per_task)
    if n <= 0:
        raise ValueError(f"n_trials_per_task must be > 0, got {n_trials_per_task}")
    return [task for task in _ALL_TASKS for _ in range(n)]


def _load_tasks_from_config(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tasks = cfg.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid tasks list in {p}: expected list, got {type(tasks)}")
    return [str(t) for t in tasks]


def _interpolate_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    tt = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(c0[0] + (c1[0] - c0[0]) * tt)),
        int(round(c0[1] + (c1[1] - c0[1]) * tt)),
        int(round(c0[2] + (c1[2] - c0[2]) * tt)),
    )


def _save_success_grid_image(
    *,
    out_path: Path,
    per_task: dict[str, dict],
    train_tasks: set[str],
) -> None:
    """
    Save a 4×4 success-rate grid image (objects × containers).

    Cell color encodes success rate within each group:
      Train tasks  — blue  scale (soft periwinkle → sky blue)
      Eval tasks   — amber scale (soft peach      → warm amber)
      Missing data — neutral gray
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise RuntimeError("PIL (Pillow) is required to save the task grid image") from e

    def _load_nice_font(size: int) -> "ImageFont.ImageFont":
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(size))
        except Exception:
            return ImageFont.load_default()

    cell = 140
    pad = 22
    left_label_w = 110
    top_label_h = 62
    gap = 4           # white gap between cells — no hard borders
    bg = (250, 250, 248)  # warm white background
    w = pad * 2 + left_label_w + cell * 4
    h = pad * 2 + top_label_h + cell * 4
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    label_font = _load_nice_font(16)
    value_font = _load_nice_font(15)

    x0 = pad + left_label_w
    y0 = pad + top_label_h

    # Palettes — moderate contrast so 0% and 100% are distinct but not harsh.
    # Train: soft periwinkle-blue → clear sky blue
    train_lo = (214, 234, 252)
    train_hi = ( 66, 165, 245)   # Material Blue 400
    # Eval:  soft peach          → warm amber
    eval_lo  = (255, 232, 186)
    eval_hi  = (255, 167,  38)   # Material Orange 400
    missing  = (225, 225, 225)   # neutral gray for truly missing data

    def _text_fill_for_bg(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        # W3C-ish relative luminance; choose white text on dark cells for contrast.
        r, g, b = [c / 255.0 for c in rgb]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # Keep the overall look light; only flip to white on genuinely dark fills.
        return (250, 250, 250) if lum < 0.42 else (60, 60, 60)

    cont_display = {"mug_no_handle": "mug\n(no handle)"}

    # Column labels (containers)
    for ci, cont in enumerate(_L4_CONTAINERS):
        x1 = x0 + ci * cell
        label = cont_display.get(cont, cont)
        if hasattr(draw, "multiline_textbbox"):
            tb = draw.multiline_textbbox((0, 0), label, font=label_font, spacing=2, align="center")  # type: ignore[attr-defined]
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
        else:
            lines = label.splitlines() or [label]
            line_bbs = [draw.textbbox((0, 0), ln, font=label_font) for ln in lines]
            line_ws = [bb[2] - bb[0] for bb in line_bbs]
            line_hs = [bb[3] - bb[1] for bb in line_bbs]
            tw = max(line_ws) if line_ws else 0
            th = sum(line_hs) + max(0, (len(lines) - 1)) * 2
        draw.multiline_text(
            (x1 + (cell - tw) / 2, pad + (top_label_h - th) / 2),
            label,
            fill=(80, 80, 80),
            font=label_font,
            spacing=2,
            align="center",
        )

    for oi, obj in enumerate(_L4_OBJECTS):
        y1 = y0 + oi * cell
        ob = draw.textbbox((0, 0), f"{obj}", font=label_font)
        ow = ob[2] - ob[0]
        oh = ob[3] - ob[1]
        draw.text(
            (pad + (left_label_w - ow) / 2, y1 + (cell - oh) / 2),
            f"{obj}",
            fill=(80, 80, 80),
            font=label_font,
        )
        for ci, cont in enumerate(_L4_CONTAINERS):
            task = f"place the {obj} into the {cont}"
            stats = per_task.get(task)
            if stats is None:
                fill = missing
                text = "-"
                text_fill = (110, 110, 110)
            else:
                sr = float(stats.get("pc_success", 0.0)) / 100.0
                text = f"{float(stats.get('pc_success', 0.0)):.1f}%"
                is_train = task in train_tasks
                if is_train:
                    fill = _interpolate_rgb(train_lo, train_hi, sr)
                else:
                    fill = _interpolate_rgb(eval_lo, eval_hi, sr)
                text_fill = _text_fill_for_bg(fill)

            x1 = x0 + ci * cell
            y1 = y0 + oi * cell
            x2 = x1 + cell
            y2 = y1 + cell

            # No per-cell border; use a subtle white gap instead.
            draw.rectangle([x1 + gap, y1 + gap, x2 - gap, y2 - gap], fill=fill)

            # Center text
            vb = draw.textbbox((0, 0), text, font=value_font)
            tw = vb[2] - vb[0]
            th = vb[3] - vb[1]
            draw.text(
                (x1 + (cell - tw) / 2, y1 + (cell - th) / 2),
                text,
                fill=text_fill,
                font=value_font,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def _make_video_writer(path: Path, fps: int):
    """Returns (append_frame_fn, close_fn). Writes H.264 via imageio-ffmpeg."""
    import imageio.v2 as iio  # type: ignore

    state: dict = {"obj": None}

    def append_frame(rgb_u8: np.ndarray):
        if state["obj"] is None:
            state["obj"] = iio.get_writer(
                str(path),
                fps=fps,
                codec="libx264",
                output_params=["-pix_fmt:v", "yuv420p", "-crf", "18"],
            )
        state["obj"].append_data(rgb_u8)

    def close():
        if state["obj"] is not None:
            state["obj"].close()

    return append_frame, close


# ---------------------------------------------------------------------------
# Cross-attention visualization
# ---------------------------------------------------------------------------

def _overlay_attn_on_rgb(
    rgb_u8: np.ndarray,
    attn_hw: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int | None = None,
) -> np.ndarray:
    """
    Overlay an attention heatmap on an RGB uint8 image.

    - rgb_u8: (H,W,3) RGB uint8
    - attn_hw: (Hf,Wf) float (any range); normalized per-frame
    """
    import cv2  # type: ignore

    if rgb_u8.ndim != 3 or rgb_u8.shape[-1] != 3:
        raise ValueError(f"Expected rgb_u8 shape (H,W,3), got {tuple(rgb_u8.shape)}")

    H, W = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    a = attn_hw.astype(np.float32)
    a = a - float(np.min(a))
    denom = float(np.max(a))
    if denom > 1e-8:
        a = a / denom
    else:
        a = np.zeros_like(a, dtype=np.float32)

    heat = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)  # (H,W)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    cmap = int(colormap) if colormap is not None else int(cv2.COLORMAP_TURBO)
    heat_bgr = cv2.applyColorMap(heat_u8, cmap)  # (H,W,3) BGR
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    out = (rgb_u8.astype(np.float32) * (1.0 - float(alpha))) + (heat_rgb.astype(np.float32) * float(alpha))
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_cross_attn_frame(
    *,
    obs: dict,
    env_idx: int,
    attn_last_step: np.ndarray,
    step: int,
) -> np.ndarray:
    """
    Render a single RGB frame:
      rows = cameras (agentview, eye_in_hand)
      cols = [raw, obj_query_overlay, cont_query_overlay]

    - attn_last_step: (Nc, Tq=2, Hf, Wf)
    """
    import cv2  # type: ignore

    cam_keys = [
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    ]
    imgs = []
    for k in cam_keys:
        if k in obs:
            imgs.append(obs[k][env_idx].astype(np.uint8))  # RGB (H,W,3)
    if not imgs:
        # Fallback: still return something with the same writer contract.
        return obs["observation.images.agentview"][env_idx].astype(np.uint8)

    Nc = int(attn_last_step.shape[0])
    if Nc != len(imgs):
        # Best-effort: clip to the smaller count.
        n = min(Nc, len(imgs))
        imgs = imgs[:n]
        attn_last_step = attn_last_step[:n]

    rows = []
    for cam_i, rgb in enumerate(imgs):
        a_obj = attn_last_step[cam_i, 0]  # (Hf,Wf)
        a_cont = attn_last_step[cam_i, 1]  # (Hf,Wf)
        ov_obj = _overlay_attn_on_rgb(rgb, a_obj)
        ov_cont = _overlay_attn_on_rgb(rgb, a_cont)

        # Add lightweight labels (draw on copies).
        def _label(im: np.ndarray, text: str) -> np.ndarray:
            out = im.copy()
            cv2.putText(
                out,
                text,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )
            return out

        cam_name = "agentview" if cam_i == 0 else f"cam{cam_i}"
        if cam_i == 1:
            cam_name = "eye_in_hand"

        p0 = _label(rgb, f"{cam_name} | raw")
        p1 = _label(ov_obj, f"{cam_name} | obj query")
        p2 = _label(ov_cont, f"{cam_name} | cont query")
        rows.append(np.concatenate([p0, p1, p2], axis=1))

    frame = np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    cv2.putText(
        frame,
        f"step {int(step)}",
        (8, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"step {int(step)}",
        (8, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    return frame.astype(np.uint8)


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def _parse_task_indices(task: str | None) -> tuple[int, int]:
    """
    Map a CG language task string → (object_idx, container_idx).

    CG_L4 only: (cross/cube/cylinder/milk) × (bin/mug/plate/mug_no_handle).
    """
    from CG_L2_image_wrapper import parse_task_indices  # local import (script-friendly)

    return parse_task_indices(task, level="l4")


# ---------------------------------------------------------------------------
# Image preprocessing (matches training pipeline)
# ---------------------------------------------------------------------------

def _preprocess_images(obs: dict, env_idx: int) -> np.ndarray:
    """
    Extract images for one environment and return (n_cams, 3, H, W) uint8.

    - Wrapper outputs RGB from robosuite/MuJoCo.
    - Training data was decoded via cv2.VideoCapture → BGR.
    - Convert RGB→BGR to match training channel order.
    - Do NOT resize/normalize — DinoImageEncoder center-crops 256→224 and
      normalises (/255 + ImageNet) internally.
    """
    import cv2  # type: ignore

    cams = [
        obs["observation.images.agentview"][env_idx],
        obs["observation.images.robot0_eye_in_hand"][env_idx],
    ]
    out = []
    for im in cams:
        im = im.astype(np.uint8)
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        out.append(np.transpose(im_bgr, (2, 0, 1)))  # (3, H, W) uint8
    return np.stack(out, axis=0)  # (n_cams, 3, H, W) uint8


def _get_video_frame(env, obs: dict, env_idx: int) -> np.ndarray:
    """Return an RGB uint8 frame for video logging."""
    if "observation.images.agentview" in obs:
        return obs["observation.images.agentview"][env_idx].astype(np.uint8)
    return env.render()[env_idx].astype(np.uint8)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(task: str, *, horizon: int):
    env_task = task
    if task == "all":
        # CG_L4 requires a concrete task at construction time; the wrapper will
        # sample tasks by calling `env.update_task(...)` between resets.
        env_task = _ALL_TASKS[0]
    from robosuite.environments.manipulation.CG_L4 import CG_L4
    from robosuite.controllers import load_composite_controller_config
    from robosuite.utils.placement_samplers import UniformApartRandomSampler

    # Use the same samplers as `tests/generate_cg_l4_motion_planning.py` to reduce
    # reset placement failures (RandomizationError).
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    bootstrap_sampler_a = UniformApartRandomSampler(
        name="FullDesk_Object_A_Sampler",
        x_range=[-0.18, 0.20],
        y_range=[-0.28, -0.05],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        min_distance=0.01,
    )
    bootstrap_sampler_b = UniformApartRandomSampler(
        name="FullDesk_Object_B_Sampler",
        x_range=[-0.18, 0.20],
        y_range=[0.05, 0.28],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        min_distance=0.005,
    )

    return CG_L4(
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        task=env_task,
        horizon=horizon,
        hard_reset=False,
        placement_initializer_A=bootstrap_sampler_a,
        placement_initializer_B=bootstrap_sampler_b,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        render_camera=["agentview", "robot0_eye_in_hand"],
    )

# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def build_policy(*, checkpoint: Path, device: str):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("torch is required") from e

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "models" / "src"))
    from policies.training.policy_loading import load_robot_flow_policy  # type: ignore

    policy = load_robot_flow_policy(checkpoint, device=device)
    policy.model.eval()
    return policy


# ---------------------------------------------------------------------------
# Single batched rollout
# ---------------------------------------------------------------------------

def run_batch(
    env,
    policy,
    *,
    horizon: int,
    n_execute: int,
    device: str,
    record_video: bool,
    record_cross_attn_video: bool,
    fps: int,
    video_paths: list[Path],
    cross_attn_video_paths: list[Path],
    episode_offset: int,
    episode_tasks: list[str],
    reset_seeds: list[int],
    policy_seed: int,
) -> dict:
    """
    Run one full batched rollout (all envs until done or horizon).
    Supports both image-conditioned and env_state-conditioned checkpoints.
    """
    num_envs = env.num_envs
    if len(episode_tasks) != num_envs:
        raise ValueError(f"episode_tasks must have length {num_envs}, got {len(episode_tasks)}")
    if len(reset_seeds) != num_envs:
        raise ValueError(f"reset_seeds must have length {num_envs}, got {len(reset_seeds)}")

    obs, _ = env.reset(tasks=episode_tasks, reset_seeds=reset_seeds)

    done = [False] * num_envs
    sum_rewards = [0.0] * num_envs
    max_rewards = [-1e9] * num_envs
    lengths = [0] * num_envs
    tasks = [getattr(env.envs[i], "task", "unknown") for i in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_success = [False] * num_envs

    # Stream videos to disk to avoid keeping all frames in RAM.
    video_append = [None for _ in range(num_envs)]
    video_close = [None for _ in range(num_envs)]
    if record_video:
        for i in range(num_envs):
            a, c = _make_video_writer(video_paths[i], fps)
            video_append[i] = a
            video_close[i] = c
            video_append[i](_get_video_frame(env, obs, i))  # type: ignore[misc]

    import torch  # type: ignore

    use_images = policy.image_feature is not None
    use_env_state = policy.env_state_feature is not None
    n_obs_steps = int(policy.state_feature.window_size)

    can_cross_attn = (
        bool(record_cross_attn_video)
        and bool(use_images)
        and hasattr(policy, "model")
        and hasattr(policy.model, "get_cross_attention_maps")
    )
    cross_append = [None for _ in range(num_envs)]
    cross_close = [None for _ in range(num_envs)]
    last_cross_frame = [None for _ in range(num_envs)]
    if can_cross_attn:
        for i in range(num_envs):
            a, c = _make_video_writer(cross_attn_video_paths[i], fps)
            cross_append[i] = a
            cross_close[i] = c

    task_idx_tensors = []
    for i in range(num_envs):
        obj_i, cont_i = _parse_task_indices(tasks[i])
        task_idx_tensors.append(torch.tensor([[obj_i, cont_i]], dtype=torch.long, device=device))

    state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)]
    env_state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)] if use_env_state else []
    img_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)] if use_images else []
    action_queues: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    policy_rng = np.random.default_rng(int(policy_seed))

    def _seed_history(i: int):
        state_hists[i].append(obs["observation.state"][i].astype(np.float32))
        if use_env_state:
            env_state_hists[i].append(obs["observation.environment_state"][i].astype(np.float32))
        if use_images:
            img_hists[i].append(_preprocess_images(obs, i))
        while len(state_hists[i]) < n_obs_steps:
            state_hists[i].append(state_hists[i][-1].copy())
            if use_env_state:
                env_state_hists[i].append(env_state_hists[i][-1].copy())
            if use_images:
                img_hists[i].append(img_hists[i][-1].copy())

    for i in range(num_envs):
        _seed_history(i)

    # Seed the cross-attention video with an initial frame that matches the reset observation.
    if can_cross_attn:
        try:
            robot_state_init = torch.from_numpy(
                np.stack([np.stack(list(state_hists[i]), axis=0) for i in range(num_envs)], axis=0)
            ).to(torch.float32).to(device)
            images_init = torch.from_numpy(
                np.stack([np.stack(list(img_hists[i]), axis=0) for i in range(num_envs)], axis=0)
            ).to(torch.uint8).to(device)
            task_idx_init = torch.cat(task_idx_tensors, dim=0)
            with torch.no_grad():
                attn = policy.model.get_cross_attention_maps(
                    robot_state=robot_state_init,
                    images=images_init,
                    task_indices=task_idx_init,
                )["query_attn"].detach().cpu().numpy()  # (B,n_obs_steps,Nc,Tq,Hf,Wf)
            for i in range(num_envs):
                last = attn[i, -1]  # (Nc,Tq,Hf,Wf)
                fr = _render_cross_attn_frame(obs=obs, env_idx=i, attn_last_step=last, step=0)
                last_cross_frame[i] = fr
                cross_append[i](fr)  # type: ignore[misc]
        except Exception as e:
            print(f"[warn] Cross-attention video disabled (failed initial render): {e}")
            can_cross_attn = False

    step = 0
    while not all(done) and step < horizon:
        need_query = [i for i in range(num_envs) if not done[i] and not action_queues[i]]

        if need_query:
            for i in need_query:
                state_hists[i].append(obs["observation.state"][i].astype(np.float32))
                if use_env_state:
                    env_state_hists[i].append(obs["observation.environment_state"][i].astype(np.float32))
                if use_images:
                    img_hists[i].append(_preprocess_images(obs, i))

            B_q = len(need_query)
            robot_state_batch = torch.from_numpy(
                np.stack([np.stack(list(state_hists[i]), axis=0) for i in need_query], axis=0)
            ).to(torch.float32).to(device)

            raw_batch: dict[str, torch.Tensor] = {"state": robot_state_batch}
            if use_env_state:
                env_state_batch = torch.from_numpy(
                    np.stack([np.stack(list(env_state_hists[i]), axis=0) for i in need_query], axis=0)
                ).to(torch.float32).to(device)
                raw_batch["env_state"] = env_state_batch
            if use_images:
                images_batch = torch.from_numpy(
                    np.stack([np.stack(list(img_hists[i]), axis=0) for i in need_query], axis=0)
                ).to(torch.uint8).to(device)
                
            task_idx_batch = torch.cat([task_idx_tensors[i] for i in need_query], dim=0)

            norm_batch = policy.processor(raw_batch)
            model_kwargs = {
                "robot_state": norm_batch["state"],
                "task_indices": task_idx_batch,
                "flow_algo": policy.model.flow_algo,
                "batch_size": B_q,
            }
            if use_env_state:
                model_kwargs["env_state"] = norm_batch["env_state"]
            if use_images:
                model_kwargs["images"] = images_batch
            with torch.no_grad():
                _set_torch_seed(int(policy_rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64)))
                actions_norm = policy.model.generate_actions(**model_kwargs)
                actions_denorm = policy.denormalize_actions(actions_norm)

            # Update cross-attention frames at replan/query points.
            if can_cross_attn:
                try:
                    with torch.no_grad():
                        attn = policy.model.get_cross_attention_maps(
                            robot_state=norm_batch["state"],
                            images=images_batch,  # type: ignore[arg-type]
                            task_indices=task_idx_batch,
                        )["query_attn"].detach().cpu().numpy()  # (Bq,n_obs_steps,Nc,Tq,Hf,Wf)
                    for qi, i in enumerate(need_query):
                        last = attn[qi, -1]  # (Nc,Tq,Hf,Wf)
                        last_cross_frame[i] = _render_cross_attn_frame(
                            obs=obs, env_idx=i, attn_last_step=last, step=step
                        )
                except Exception as e:
                    print(f"[warn] Cross-attention render failed at step={step} (disabling): {e}")
                    can_cross_attn = False

            acts_np = actions_denorm[:, :n_execute].detach().cpu().numpy()
            for qi, i in enumerate(need_query):
                for t in range(acts_np.shape[1]):
                    action_queues[i].append(acts_np[qi, t, :7].astype(np.float32))

        action_mat = np.zeros((num_envs, 7), dtype=np.float32)
        for i in range(num_envs):
            if not done[i] and action_queues[i]:
                action_mat[i] = action_queues[i].pop(0)

        obs, rew, terminated, truncated, info = env.step(action_mat.astype(np.float32))
        successes = np.asarray(info.get("is_success", [False] * num_envs))

        for i in range(num_envs):
            if done[i]:
                continue
            r = float(rew[i]) if hasattr(rew, "__len__") else float(rew)
            sum_rewards[i] += r
            max_rewards[i] = max(max_rewards[i], r)
            lengths[i] += 1
            ep_actions[i].append(action_mat[i].copy())
            ep_rewards[i].append(r)
            if bool(successes[i]):
                ep_success[i] = True

            step_done = bool(terminated[i] if hasattr(terminated, "__len__") else terminated) or bool(
                truncated[i] if hasattr(truncated, "__len__") else truncated
            )
            if step_done or ep_success[i]:
                done[i] = True

            if record_video and not done[i]:
                video_append[i](_get_video_frame(env, obs, i))  # type: ignore[misc]
            if can_cross_attn and not done[i]:
                # For intermediate MPC steps, reuse the last attention map (updated on query steps).
                if last_cross_frame[i] is not None:
                    cross_append[i](last_cross_frame[i])  # type: ignore[misc]

        step += 1
        running_sr = np.mean(ep_success[:num_envs]) * 100
        print(
            f"\r  step {step:4d}/{horizon} | done {sum(done)}/{num_envs} | "
            f"running SR={running_sr:.0f}%",
            end="",
            flush=True,
        )

    print()

    if record_video:
        for i in range(num_envs):
            if video_close[i] is not None:
                video_close[i]()  # type: ignore[misc]
    if can_cross_attn:
        for i in range(num_envs):
            if cross_close[i] is not None:
                cross_close[i]()  # type: ignore[misc]

    return {
        "success": ep_success,
        "sum_reward": sum_rewards,
        "max_reward": max_rewards,
        "length": lengths,
        "task": tasks,
        "actions": [np.stack(a, axis=0) if a else np.zeros((0, 7), dtype=np.float32) for a in ep_actions],
        "rewards": [np.array(r, dtype=np.float32) for r in ep_rewards],
    }


# ---------------------------------------------------------------------------
# Multi-episode evaluator
# ---------------------------------------------------------------------------

def eval_policy(
    env: ImageBasedCGWrapper,
    policy,
    *,
    episode_tasks: list[str],
    horizon: int,
    n_execute: int,
    device: str,
    fps: int,
    out_dir: Path,
    max_videos: int,
    max_cross_attn_videos: int,
    result_config: dict,
    seed_plan: EvalSeedPlan,
) -> dict:
    """
    Run ceil(n_episodes / num_envs) batched rollouts and aggregate metrics.
    Returns the full info dict (also written to eval_summary.json).
    """
    n_episodes = len(episode_tasks)
    num_envs   = env.num_envs
    n_batches  = math.ceil(n_episodes / num_envs)
    videos_dir = out_dir / "videos"
    cross_dir = out_dir / "cross_attn_videos"
    rollouts_dir = out_dir / "rollouts"
    videos_dir.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: list[dict] = []
    n_episodes_rendered = 0
    n_cross_rendered = 0
    start_time = time.time()

    save_cross_attn = bool(result_config.get("save_cross_attention_videos", False))

    for batch_ix in range(n_batches):
        ep_offset = batch_ix * num_envs
        remaining = n_episodes - ep_offset
        if remaining <= 0:
            break

        # How many of this batch's episodes should be recorded?
        can_record = max(0, max_videos - n_episodes_rendered)
        record_this_batch = min(can_record, num_envs) > 0

        can_record_cross = max(0, max_cross_attn_videos - n_cross_rendered)
        record_cross_this_batch = save_cross_attn and (min(can_record_cross, num_envs) > 0)

        # Pre-build video paths
        video_paths_batch: list[Path] = []
        cross_paths_batch: list[Path] = []
        for i in range(num_envs):
            ep_ix = ep_offset + i
            video_paths_batch.append(videos_dir / f"ep{ep_ix:03d}_PENDING.mp4")
            cross_paths_batch.append(cross_dir / f"ep{ep_ix:03d}_PENDING_crossattn.mp4")

        print(
            f"\n[batch {batch_ix + 1}/{n_batches}]  "
            f"episodes {ep_offset}–{ep_offset + min(num_envs, remaining) - 1}  "
            f"(record_video={record_this_batch}, record_cross_attn={record_cross_this_batch})"
        )

        batch_tasks = episode_tasks[ep_offset : ep_offset + num_envs]
        batch_reset_seeds = seed_plan.episode_reset_seeds[ep_offset : ep_offset + num_envs]
        batch_policy_seed = seed_plan.batch_policy_seeds[batch_ix]
        if len(batch_tasks) < num_envs:
            pad = num_envs - len(batch_tasks)
            batch_tasks.extend(batch_tasks[-1:] * pad)
            batch_reset_seeds.extend(batch_reset_seeds[-1:] * pad)

        batch = run_batch(
            env,
            policy,
            horizon=horizon,
            n_execute=n_execute,
            device=device,
            record_video=record_this_batch and (n_episodes_rendered < max_videos),
            record_cross_attn_video=record_cross_this_batch and (n_cross_rendered < max_cross_attn_videos),
            fps=fps,
            video_paths=video_paths_batch,
            cross_attn_video_paths=cross_paths_batch,
            episode_offset=ep_offset,
            episode_tasks=batch_tasks,
            reset_seeds=batch_reset_seeds,
            policy_seed=batch_policy_seed,
        )

        # Rename video files to include success/failure and task
        for i in range(num_envs):
            ep_ix = ep_offset + i
            if ep_ix >= n_episodes:
                # Clean up any padded env outputs.
                try:
                    if video_paths_batch[i].exists():
                        video_paths_batch[i].unlink()
                except Exception:
                    pass
                try:
                    if cross_paths_batch[i].exists():
                        cross_paths_batch[i].unlink()
                except Exception:
                    pass
                break
            success_str = "Success" if batch["success"][i] else "Failure"
            safe_task   = batch["task"][i].replace(" ", "_")
            final_path  = videos_dir / f"ep{ep_ix:03d}_{success_str}_{safe_task}.mp4"
            final_cross = cross_dir / f"ep{ep_ix:03d}_{success_str}_{safe_task}_crossattn.mp4"
            if record_this_batch and n_episodes_rendered < max_videos:
                video_paths_batch[i].rename(final_path)
                n_episodes_rendered += 1
                video_path_str = str(final_path)
            else:
                # Remove placeholder path (no video written)
                video_path_str = None
                try:
                    if video_paths_batch[i].exists():
                        video_paths_batch[i].unlink()
                except Exception:
                    pass

            cross_path_str = None
            if record_cross_this_batch and n_cross_rendered < max_cross_attn_videos:
                if cross_paths_batch[i].exists():
                    cross_paths_batch[i].rename(final_cross)
                    n_cross_rendered += 1
                    cross_path_str = str(final_cross)
            else:
                try:
                    if cross_paths_batch[i].exists():
                        cross_paths_batch[i].unlink()
                except Exception:
                    pass

            # Save per-episode rollout NPZ
            np.savez_compressed(
                rollouts_dir / f"ep{ep_ix:03d}.npz",
                actions=batch["actions"][i],
                rewards=batch["rewards"][i],
                success=np.array([batch["success"][i]]),
                task=np.array([batch["task"][i]]),
            )

            all_episodes.append({
                "episode_ix":  ep_ix,
                "task":        batch["task"][i],
                "success":     bool(batch["success"][i]),
                "sum_reward":  float(batch["sum_reward"][i]),
                "max_reward":  float(batch["max_reward"][i]),
                "length":      int(batch["length"][i]),
                "seed":        int(seed_plan.episode_reset_seeds[ep_ix]),
                "root_seed":   int(seed_plan.root_seed),
                "task_seed":   int(seed_plan.episode_task_seeds[ep_ix]),
                "policy_seed": int(batch_policy_seed),
                "video":       video_path_str,
                "cross_attn_video": cross_path_str,
            })

        # Running summary after each batch
        done_eps = all_episodes[:n_episodes]
        sr = np.mean([e["success"] for e in done_eps]) * 100
        print(
            f"  → after {len(done_eps)} episodes: "
            f"SR={sr:.1f}%  "
            f"avg_len={np.mean([e['length'] for e in done_eps]):.1f}"
        )

    # Trim to exactly n_episodes
    all_episodes = all_episodes[:n_episodes]
    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Per-task breakdown
    # ------------------------------------------------------------------
    per_task: dict[str, dict] = {}
    task_successes: dict[str, list[bool]] = defaultdict(list)
    task_rewards: dict[str, list[float]] = defaultdict(list)
    task_lengths: dict[str, list[int]] = defaultdict(list)
    
    for ep in all_episodes:
        task_successes[ep["task"]].append(ep["success"])
        task_rewards[ep["task"]].append(ep["sum_reward"])
        task_lengths[ep["task"]].append(ep["length"])
    
    for task, succs in task_successes.items():
        per_task_stats = {
            "n_episodes":  len(succs),
            "n_success":   int(sum(succs)),
            "pc_success":  float(np.mean(succs) * 100),
        }
        
        if result_config.get("include_reward_statistics", True):
            per_task_stats.update({
                "avg_sum_reward": float(np.mean(task_rewards[task])),
                "std_sum_reward": float(np.std(task_rewards[task])),
                "min_sum_reward": float(np.min(task_rewards[task])),
                "max_sum_reward": float(np.max(task_rewards[task])),
            })
        
        if result_config.get("include_action_statistics", True):
            per_task_stats.update({
                "avg_length": float(np.mean(task_lengths[task])),
                "std_length": float(np.std(task_lengths[task])),
                "min_length": int(np.min(task_lengths[task])),
                "max_length": int(np.max(task_lengths[task])),
            })
        
        per_task[task] = per_task_stats

    aggregated = {
        "n_episodes":    len(all_episodes),
        "pc_success":    float(np.mean([e["success"]    for e in all_episodes]) * 100),
        "avg_sum_reward":float(np.mean([e["sum_reward"] for e in all_episodes])),
        "avg_max_reward":float(np.mean([e["max_reward"] for e in all_episodes])),
        "avg_ep_length": float(np.mean([e["length"]     for e in all_episodes])),
        "std_ep_length": float(np.std([e["length"]      for e in all_episodes])),
        "eval_s":        elapsed,
        "eval_ep_s":     elapsed / max(len(all_episodes), 1),
    }

    info = {
        "aggregated":  aggregated,
        "per_task":    per_task if result_config.get("save_per_task_breakdown", True) else {},
        "per_episode": all_episodes if result_config.get("save_per_episode_details", True) else [],
    }

    (out_dir / "eval_summary.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )

    return info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load evaluation configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-episode policy evaluation on CG_L4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON eval config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--n-trials-per-task",
        "--n_trials_per_task",
        dest="n_trials_per_task",
        type=int,
        default=None,
        help="Override eval_config.n_trials_per_task from the config JSON",
    )
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Parallel environments per batch (must be even)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Maximum steps per episode")
    # Policy
    parser.add_argument("--n-execute", type=int, default=None,
                        help="Actions to execute per policy query (MPC horizon)")
    # Output
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of episode videos to save")
    parser.add_argument(
        "--save-cross-attn-videos",
        action="store_true",
        help="Save an extra per-episode MP4 with cross-attention overlays (requires an image-conditioned checkpoint)",
    )
    parser.add_argument(
        "--cross-attn-max-videos",
        type=int,
        default=None,
        help="Maximum number of cross-attention videos to save (defaults to --max-videos)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    eval_config = config.get("eval_config", {})
    result_config = config.get("result_config", {})

    # Optional: use a train task config to color the 4×4 grid patches.
    train_tasks_cfg_path = config.get("train_tasks_config", None)
    train_tasks_list = config.get("train_tasks", None)
    if train_tasks_list is not None:
        if not isinstance(train_tasks_list, list):
            raise ValueError(f"train_tasks must be a list, got {type(train_tasks_list)}")
        train_tasks = {str(t) for t in train_tasks_list}
    else:
        train_tasks = set(_load_tasks_from_config(train_tasks_cfg_path))

    checkpoint = args.checkpoint or eval_config.get("checkpoint")
    device = args.device or eval_config.get("device", "cuda")
    num_envs = args.num_envs if args.num_envs is not None else eval_config.get("num_envs", 2)
    horizon = args.horizon if args.horizon is not None else eval_config.get("horizon", 200)
    n_execute = args.n_execute if args.n_execute is not None else eval_config.get("n_execute", 8)
    out_dir = args.out_dir or eval_config.get("out_dir", "artifacts")
    max_videos = args.max_videos if args.max_videos is not None else eval_config.get("max_videos", 10)
    max_cross_attn_videos = (
        args.cross_attn_max_videos
        if args.cross_attn_max_videos is not None
        else eval_config.get("cross_attn_max_videos", max_videos)
    )
    seed = args.seed if args.seed is not None else eval_config.get("seed", 0)

    if not checkpoint:
        raise ValueError("eval_config.checkpoint (or --checkpoint) is required")

    if args.save_cross_attn_videos:
        result_config = dict(result_config)
        result_config["save_cross_attention_videos"] = True

    n_trials_per_task = args.n_trials_per_task if args.n_trials_per_task is not None else eval_config.get("n_trials_per_task")
    if n_trials_per_task is None:
        raise ValueError("eval_config.n_trials_per_task is required — total episodes = n_trials_per_task × 16 tasks")
    episode_tasks = _build_episode_tasks_full_grid(n_trials_per_task=int(n_trials_per_task))
    n_episodes = len(episode_tasks)

    os.environ.setdefault("MUJOCO_GL", "egl")
    deterministic_eval = True
    seed_plan = _build_seed_plan(root_seed=seed, n_episodes=n_episodes, num_envs=num_envs)
    _set_global_seed(seed_plan.global_seed, deterministic=deterministic_eval)

    policy = build_policy(
        checkpoint=Path(checkpoint),
        device=device,
    )

    wrapper_cls = ImageBasedCGWrapper if policy.image_feature is not None else StateBasedCGWrapper
    env = wrapper_cls(
        make_env_fn=lambda: make_env("all", horizon=horizon),
        num_envs=num_envs,
        use_relative_coordinates=(policy.env_state_feature is not None),
    )

    fps = int(getattr(env.envs[0], "control_freq", 20))
    print(f"env control_freq={fps} Hz | num_envs={num_envs}")

    probe_task = str(_ALL_TASKS[0])
    probe_tasks = [probe_task for _ in range(num_envs)]
    probe_reset_seeds = [int(seed_plan.global_seed + i) for i in range(num_envs)]
    obs, _ = env.reset(tasks=probe_tasks, reset_seeds=probe_reset_seeds)
    policy_state_dim = int(np.prod(policy.state_feature.shape))
    env_state_dim = int(obs["observation.state"].shape[-1])
    if policy_state_dim != env_state_dim:
        env.close()
        raise ValueError(
            f"State dim mismatch: policy expects {policy_state_dim}, "
            f"env provides {env_state_dim}. Check the checkpoint config."
        )
    if policy.env_state_feature is not None:
        policy_env_state_dim = int(np.prod(policy.env_state_feature.shape))
        env_env_state_dim = int(obs["observation.environment_state"].shape[-1])
        if policy_env_state_dim != env_env_state_dim:
            env.close()
            raise ValueError(
                f"Env-state dim mismatch: policy expects {policy_env_state_dim}, "
                f"env provides {env_env_state_dim}. Check the wrapper output."
            )
    if policy.image_feature is not None:
        print(f"state_dim={policy_state_dim}  image_shape={tuple(policy.image_feature.shape)}")
    else:
        print(
            f"state_dim={policy_state_dim}  "
            f"env_state_dim={int(np.prod(policy.env_state_feature.shape)) if policy.env_state_feature is not None else 'n/a'}"
        )

    # Output directory
    run_id  = time.strftime("%Y%m%d-%H%M%S")
    out_dir_path = Path(out_dir) / f"eval_{run_id}"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir_path}")

    # Save configuration used for evaluation
    eval_params = {
        "checkpoint": checkpoint,
        "device": device,
        "n_episodes": n_episodes,
        "n_trials_per_task": int(n_trials_per_task),
        "num_envs": num_envs,
        "horizon": horizon,
        "n_execute": n_execute,
        "max_videos": max_videos,
        "seed": seed,
        "eval_tasks": _ALL_TASKS,
        "result_config": result_config,
        "train_tasks_config": train_tasks_cfg_path,
        "train_tasks": sorted(train_tasks),
        "reproducibility": {
            "deterministic_torch": deterministic_eval,
            "seed_plan": asdict(seed_plan),
        },
    }
    (out_dir_path / "eval_config_used.json").write_text(
        json.dumps(eval_params, indent=2) + "\n", encoding="utf-8"
    )

    # Run evaluation
    info = eval_policy(
        env,
        policy,
        episode_tasks=episode_tasks,
        horizon=horizon,
        n_execute=n_execute,
        device=device,
        fps=fps,
        out_dir=out_dir_path,
        max_videos=max_videos,
        max_cross_attn_videos=int(max_cross_attn_videos),
        result_config=result_config,
        seed_plan=seed_plan,
    )

    env.close()

    # Print summary
    agg = info["aggregated"]
    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY  ({agg['n_episodes']} episodes, {agg['eval_s']:.1f}s)")
    print("=" * 60)
    print(f"  Success rate : {agg['pc_success']:.1f}%")
    print(f"  Avg sum rew  : {agg['avg_sum_reward']:.3f}")
    print(f"  Avg max rew  : {agg['avg_max_reward']:.3f}")
    print(f"  Avg length   : {agg['avg_ep_length']:.1f} ± {agg['std_ep_length']:.1f} steps")
    print()
    if info["per_task"]:
        print("Per-task breakdown:")
        for task_name, stats in sorted(info["per_task"].items()):
            print(f"  {task_name:<40s}  {stats['pc_success']:5.1f}%  ({stats['n_success']}/{stats['n_episodes']})")
            if result_config.get("include_reward_statistics", True):
                print(f"    {'':40s}  avg_rew={stats['avg_sum_reward']:.3f} ± {stats['std_sum_reward']:.3f}")
            if result_config.get("include_action_statistics", True):
                print(f"    {'':40s}  avg_len={stats['avg_length']:.1f} ± {stats['std_length']:.1f}")

        # Save 4×4 success-rate grid image (train vs eval bordered).
        try:
            grid_path = out_dir_path / "task_grid_success.png"
            _save_success_grid_image(
                out_path=grid_path,
                per_task=info["per_task"],
                train_tasks=train_tasks,
            )
            print(f"\nSaved task grid image: {grid_path}")
        except Exception as e:
            print(f"\n[warn] Could not save task grid image: {e}")
    print("=" * 60)
    print(f"Full results: {out_dir_path / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
