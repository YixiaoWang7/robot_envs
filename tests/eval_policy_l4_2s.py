#!/usr/bin/env python
"""
Multi-episode policy evaluation on CG_L4.

Loads a RobotFlowPolicyWrapper checkpoint and runs MPC inference.

Outputs (under --out-dir / eval_{timestamp}/):
  videos/ep{i:03d}_{Success|Failure}_{task}.mp4   (up to --max-videos)
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

# Ensure imports work regardless of current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "CG"))
sys.path.insert(0, str(_REPO_ROOT / "CG" / "robosuite"))

from CG_L2_image_wrapper import ImageBasedCGWrapper
from CG_L2_state_wrapper import StateBasedCGWrapper


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
# Two-stage task helpers
# ---------------------------------------------------------------------------

_L4_OBJECTS = ("cross", "cube", "cylinder", "milk")
_L4_CONTAINERS = ("bin", "mug", "plate", "mug_no_handle")


def _stage_task_string(*, obj_name: str, cont_name: str) -> str:
    return f"place the {obj_name} into the {cont_name}"


def _parse_stage_slug(stage_slug: str) -> tuple[str, str]:
    st = str(stage_slug).strip().lower()
    if "_into_" not in st:
        raise ValueError(f"Expected stage slug '<obj>_into_<cont>', got: {stage_slug!r}")
    obj, cont = st.split("_into_", 1)
    obj = obj.strip()
    cont = cont.strip()
    if obj not in _L4_OBJECTS:
        raise ValueError(f"Unknown object in stage slug {stage_slug!r}: {obj!r} (expected one of {_L4_OBJECTS})")
    if cont not in _L4_CONTAINERS:
        raise ValueError(
            f"Unknown container in stage slug {stage_slug!r}: {cont!r} (expected one of {_L4_CONTAINERS})"
        )
    return obj, cont


def _stage_task_to_stage_slug(task: str) -> str:
    """
    Convert a single-stage language task to a stage slug.
    Expected format: "place the <obj> into the <cont>".
    """
    tl = str(task).strip().lower()
    if not tl:
        raise ValueError("Empty task string")

    # Check longer tokens first so "mug_no_handle" doesn't match "mug".
    cont_candidates = sorted(_L4_CONTAINERS, key=len, reverse=True)
    obj_candidates = sorted(_L4_OBJECTS, key=len, reverse=True)

    obj = next((o for o in obj_candidates if o in tl), None)
    cont = next((c for c in cont_candidates if c in tl), None)
    if obj is None or cont is None:
        raise ValueError(
            f"Could not parse stage task {task!r}. Expected something like "
            f"'place the cross into the bin' with obj in {_L4_OBJECTS} and cont in {_L4_CONTAINERS}."
        )
    return f"{obj}_into_{cont}"


def _two_stage_slug_to_stage_tasks(task_slug: str) -> tuple[str, str]:
    """
    Parse "<obj0>_into_<cont0>__<obj1>_into_<cont1>" into two CG_L4-compatible *stage* tasks:
      ("place the <obj0> into the <cont0>", "place the <obj1> into the <cont1>")

    Important: CG_L4 only understands single-stage instructions; two-stage evaluation must
    switch the env target mid-episode rather than passing a combined string with "__".
    """
    s = str(task_slug).strip().lower()
    if not s:
        raise ValueError("task_slug is empty")
    parts = s.split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected two-stage slug with exactly one '__', got: {s!r}")

    o0, c0 = _parse_stage_slug(parts[0])
    o1, c1 = _parse_stage_slug(parts[1])
    return _stage_task_string(obj_name=o0, cont_name=c0), _stage_task_string(obj_name=o1, cont_name=c1)


def _load_two_stage_task_slugs_from_checkpoint(checkpoint_path: Path) -> list[str]:
    """
    Load two-stage task slugs from the dataset referenced by the checkpoint run_config.
    """
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("torch is required to read checkpoint run_config") from e

    ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    run_cfg = ckpt.get("run_config", {}) or {}
    ds = (run_cfg.get("dataset") or {}) if isinstance(run_cfg, dict) else {}
    data_dir = str(ds.get("data_dir", "")).strip()
    if not data_dir:
        raise ValueError("Checkpoint run_config.dataset.data_dir is missing; cannot discover two-stage tasks.")
    ds_path = Path(data_dir) / "dataset_manifest.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"Expected dataset_manifest.json at: {ds_path}")
    with ds_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    slugs = [str(t["task_slug"]) for t in (manifest.get("tasks") or []) if isinstance(t, dict) and "task_slug" in t]
    if not slugs:
        raise ValueError(f"No task_slug entries found in {ds_path}")
    return slugs


def _validate_two_stage_eval_config(cfg: dict) -> dict:
    """
    Strict config validation for two-stage eval.

    Requirements:
    - cfg must include non-empty `task_slugs` (no defaulting to checkpoint dataset).
    - if optional alignment fields are present, they must match `task_slugs`.

    Supported optional alignment fields:
    - stage0_tasks: list[str] same length as task_slugs
    - stage1_tasks: list[str] same length as task_slugs
    - stage0_train_tasks + stage0_train_task_slugs
    - stage1_train_tasks + stage1_train_task_slugs
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"Eval config must be a JSON object, got {type(cfg)}")
    if "eval_config" not in cfg or "result_config" not in cfg:
        raise ValueError("Eval config missing required keys: 'eval_config' and/or 'result_config'")

    if "task_slugs" not in cfg:
        raise ValueError(
            "Missing required key 'task_slugs' in eval config. "
            "For two-stage evaluation this must be explicitly provided (no default). "
            "Generate it with tests/test_configs/2s/generate_eval_task_sets.py."
        )
    task_slugs = cfg.get("task_slugs")
    if not isinstance(task_slugs, list) or not task_slugs:
        raise ValueError("'task_slugs' must be a non-empty list of two-stage slugs.")

    slugs: list[str] = []
    seen: set[str] = set()
    for i, s in enumerate(task_slugs):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"task_slugs[{i}] must be a non-empty string, got {s!r}")
        slug = str(s).strip().lower()
        # Validate slug format + vocab.
        _two_stage_slug_to_stage_tasks(slug)
        if slug in seen:
            raise ValueError(f"Duplicate task slug in task_slugs: {slug!r}")
        seen.add(slug)
        slugs.append(slug)

    # Optional: stage task strings (per-eval-task) must align with slugs.
    for k in ("stage0_tasks", "stage1_tasks"):
        if k in cfg and cfg[k] is not None:
            if not isinstance(cfg[k], list):
                raise ValueError(f"'{k}' must be a list[str] when provided.")
            if len(cfg[k]) != len(slugs):
                raise ValueError(f"'{k}' must have same length as task_slugs ({len(slugs)}), got {len(cfg[k])}.")

    if "stage0_tasks" in cfg or "stage1_tasks" in cfg:
        stage0_tasks = cfg.get("stage0_tasks", None)
        stage1_tasks = cfg.get("stage1_tasks", None)
        if stage0_tasks is None or stage1_tasks is None:
            raise ValueError("If providing stage task strings, both 'stage0_tasks' and 'stage1_tasks' must be set.")
        for i, slug in enumerate(slugs):
            exp0, exp1 = _two_stage_slug_to_stage_tasks(slug)
            got0 = str(stage0_tasks[i]).strip().lower()
            got1 = str(stage1_tasks[i]).strip().lower()
            if got0 != exp0:
                raise ValueError(
                    f"stage0_tasks[{i}] does not match task_slugs[{i}]={slug!r}. "
                    f"Expected {exp0!r}, got {stage0_tasks[i]!r}."
                )
            if got1 != exp1:
                raise ValueError(
                    f"stage1_tasks[{i}] does not match task_slugs[{i}]={slug!r}. "
                    f"Expected {exp1!r}, got {stage1_tasks[i]!r}."
                )

    # Optional: training set alignment (language tasks <-> slugs).
    for stage in ("stage0", "stage1"):
        tasks_k = f"{stage}_train_tasks"
        slugs_k = f"{stage}_train_task_slugs"
        if (tasks_k in cfg) != (slugs_k in cfg):
            raise ValueError(f"If providing '{tasks_k}', you must also provide '{slugs_k}' (and vice versa).")
        if tasks_k in cfg:
            tasks_v = cfg.get(tasks_k)
            slugs_v = cfg.get(slugs_k)
            if not isinstance(tasks_v, list) or not tasks_v:
                raise ValueError(f"'{tasks_k}' must be a non-empty list[str].")
            if not isinstance(slugs_v, list) or not slugs_v:
                raise ValueError(f"'{slugs_k}' must be a non-empty list[str].")

            exp = [_stage_task_to_stage_slug(t) for t in tasks_v]
            got = [str(s).strip().lower() for s in slugs_v]
            if set(exp) != set(got):
                missing = sorted(set(exp) - set(got))
                extra = sorted(set(got) - set(exp))
                raise ValueError(
                    f"Mismatch between {tasks_k} and {slugs_k}.\n"
                    f"- missing_in_{slugs_k}: {missing}\n"
                    f"- extra_in_{slugs_k}: {extra}"
                )

    cfg = dict(cfg)
    cfg["task_slugs"] = slugs
    return cfg


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
        task=str(task),
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
    fps: int,
    video_paths: list[Path],
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

    task_slugs = [str(t) for t in episode_tasks]
    stage0_tasks: list[str] = []
    stage1_tasks: list[str] = []
    for slug in task_slugs:
        t0, t1 = _two_stage_slug_to_stage_tasks(slug)
        stage0_tasks.append(t0)
        stage1_tasks.append(t1)

    obs, _ = env.reset(tasks=stage0_tasks, reset_seeds=reset_seeds)

    done = [False] * num_envs
    sum_rewards = [0.0] * num_envs
    max_rewards = [-1e9] * num_envs
    lengths = [0] * num_envs
    tasks = task_slugs
    ep_frames = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_success = [False] * num_envs

    if record_video:
        for i in range(num_envs):
            ep_frames[i].append(_get_video_frame(env, obs, i))

    import torch  # type: ignore

    use_images = policy.image_feature is not None
    use_env_state = policy.env_state_feature is not None
    n_obs_steps = int(policy.state_feature.window_size)

    # Build per-env task indices in the shape expected by the model:
    # - 1-stage policies: (obj, cont) -> (1,2)
    # - 2-stage full policies: (obj0, cont0, obj1, cont1) -> (1,4)
    num_query_tokens = int(getattr(policy.model, "num_query_tokens", 2))
    state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)]
    env_state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)] if use_env_state else []
    img_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)] if use_images else []
    action_queues: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    policy_rng = np.random.default_rng(int(policy_seed))
    current_stage = [0] * num_envs
    stage0_done = [False] * num_envs
    stage1_done = [False] * num_envs

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

            raw_batch: dict = {"state": robot_state_batch, "task_slug": [task_slugs[i] for i in need_query]}
            if use_env_state:
                env_state_batch = torch.from_numpy(
                    np.stack([np.stack(list(env_state_hists[i]), axis=0) for i in need_query], axis=0)
                ).to(torch.float32).to(device)
                raw_batch["env_state"] = env_state_batch
            if use_images:
                images_batch = torch.from_numpy(
                    np.stack([np.stack(list(img_hists[i]), axis=0) for i in need_query], axis=0)
                ).to(torch.uint8).to(device)

            norm_batch = policy.processor(raw_batch)
            if "task_indices" not in norm_batch:
                raise KeyError(
                    "policy.processor did not produce 'task_indices'. "
                    "For two-stage evaluation, the checkpoint processor should have "
                    "processor.add_task_indices=true and processor.task_indices_mode='task_slug_two_stage_full'."
                )
            task_idx_batch = norm_batch["task_indices"].to(device)
            model_kwargs = {
                "robot_state": norm_batch["state"].to(device),
                "task_indices": task_idx_batch,
                "flow_algo": policy.model.flow_algo,
                "batch_size": B_q,
            }
            if use_env_state:
                model_kwargs["env_state"] = norm_batch["env_state"].to(device)
            if use_images:
                model_kwargs["images"] = images_batch
            with torch.no_grad():
                _set_torch_seed(int(policy_rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64)))
                actions_norm = policy.model.generate_actions(**model_kwargs)
                actions_denorm = policy.denormalize_actions(actions_norm)

            acts_np = actions_denorm[:, :n_execute].detach().cpu().numpy()
            for qi, i in enumerate(need_query):
                for t in range(acts_np.shape[1]):
                    action_queues[i].append(acts_np[qi, t, :7].astype(np.float32))

        action_mat = np.zeros((num_envs, 7), dtype=np.float32)
        for i in range(num_envs):
            if not done[i] and action_queues[i]:
                action_mat[i] = action_queues[i].pop(0)

        obs, rew, terminated, truncated, info = env.step(action_mat.astype(np.float32))

        for i in range(num_envs):
            if done[i]:
                continue
            r = float(rew[i]) if hasattr(rew, "__len__") else float(rew)
            sum_rewards[i] += r
            max_rewards[i] = max(max_rewards[i], r)
            lengths[i] += 1
            ep_actions[i].append(action_mat[i].copy())
            ep_rewards[i].append(r)

            # NOTE: CG wrapper returns a *latched* success flag (once True, stays True).
            # For two-stage evaluation we need stage-local success, so read directly from the env.
            try:
                stage_success = bool(env.envs[i]._check_success())
            except Exception:
                stage_success = False
            if int(current_stage[i]) == 0 and stage_success:
                # Stage 0 complete → retarget env to stage 1 *without resetting*.
                stage0_done[i] = True
                current_stage[i] = 1
                env.envs[i].update_task(stage1_tasks[i])
                if hasattr(env.envs[i], "_refresh_task_pointers") and callable(env.envs[i]._refresh_task_pointers):
                    env.envs[i]._refresh_task_pointers()
                # Reset wrapper's latched success so it keeps applying actions for stage 1.
                if hasattr(env, "is_success") and isinstance(getattr(env, "is_success"), list):
                    try:
                        env.is_success[i] = False
                    except Exception:
                        pass

                # Refresh wrapper's cached raw obs so "target-relative" observables update immediately.
                if hasattr(env.envs[i], "_get_observations") and hasattr(env, "obs"):
                    try:
                        env.obs[i] = env.envs[i]._get_observations()
                        if hasattr(env, "_compute_observation") and callable(env._compute_observation):
                            obs_i = env._compute_observation(i)
                            for k in list(obs.keys()):
                                obs[k][i] = obs_i[k]
                    except Exception:
                        # If this fails, we still continue; the next env.step() will update obs normally.
                        pass

            elif int(current_stage[i]) == 1 and stage_success:
                stage1_done[i] = True

            ep_success[i] = bool(stage0_done[i] and stage1_done[i])

            step_done = bool(terminated[i] if hasattr(terminated, "__len__") else terminated) or bool(
                truncated[i] if hasattr(truncated, "__len__") else truncated
            )
            if step_done or ep_success[i]:
                done[i] = True

            if record_video and not done[i]:
                ep_frames[i].append(_get_video_frame(env, obs, i))

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
        import threading

        def _write_video(frames: list[np.ndarray], path: Path):
            append, close = _make_video_writer(path, fps)
            for f in frames:
                append(f)
            close()

        threads = []
        for i in range(num_envs):
            if ep_frames[i]:
                t = threading.Thread(target=_write_video, args=(ep_frames[i], video_paths[i]))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

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
    rollouts_dir = out_dir / "rollouts"
    videos_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: list[dict] = []
    n_episodes_rendered = 0
    start_time = time.time()

    for batch_ix in range(n_batches):
        ep_offset = batch_ix * num_envs
        remaining = n_episodes - ep_offset
        if remaining <= 0:
            break

        # How many of this batch's episodes should be recorded?
        can_record = max(0, max_videos - n_episodes_rendered)
        record_this_batch = min(can_record, num_envs) > 0

        # Pre-build video paths
        video_paths_batch: list[Path] = []
        for i in range(num_envs):
            ep_ix = ep_offset + i
            video_paths_batch.append(videos_dir / f"ep{ep_ix:03d}_PENDING.mp4")

        print(
            f"\n[batch {batch_ix + 1}/{n_batches}]  "
            f"episodes {ep_offset}–{ep_offset + min(num_envs, remaining) - 1}  "
            f"(record_video={record_this_batch})"
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
            fps=fps,
            video_paths=video_paths_batch,
            episode_offset=ep_offset,
            episode_tasks=batch_tasks,
            reset_seeds=batch_reset_seeds,
            policy_seed=batch_policy_seed,
        )

        # Rename video files to include success/failure and task
        for i in range(num_envs):
            ep_ix = ep_offset + i
            if ep_ix >= n_episodes:
                break
            success_str = "Success" if batch["success"][i] else "Failure"
            safe_task   = batch["task"][i].replace(" ", "_")
            final_path  = videos_dir / f"ep{ep_ix:03d}_{success_str}_{safe_task}.mp4"
            if record_this_batch and n_episodes_rendered < max_videos:
                video_paths_batch[i].rename(final_path)
                n_episodes_rendered += 1
                video_path_str = str(final_path)
            else:
                # Remove placeholder path (no video written)
                video_path_str = None

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
        cfg = json.load(f)
    return _validate_two_stage_eval_config(cfg)


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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    eval_config = config.get("eval_config", {})
    result_config = config.get("result_config", {})

    checkpoint = args.checkpoint or eval_config.get("checkpoint")
    device = args.device or eval_config.get("device", "cuda")
    num_envs = args.num_envs if args.num_envs is not None else eval_config.get("num_envs", 2)
    horizon = args.horizon if args.horizon is not None else eval_config.get("horizon", 200)
    n_execute = args.n_execute if args.n_execute is not None else eval_config.get("n_execute", 8)
    out_dir = args.out_dir or eval_config.get("out_dir", "artifacts")
    max_videos = args.max_videos if args.max_videos is not None else eval_config.get("max_videos", 10)
    seed = args.seed if args.seed is not None else eval_config.get("seed", 0)

    if not checkpoint:
        raise ValueError("eval_config.checkpoint (or --checkpoint) is required")

    n_trials_per_task = args.n_trials_per_task if args.n_trials_per_task is not None else eval_config.get("n_trials_per_task")
    if n_trials_per_task is None:
        raise ValueError("eval_config.n_trials_per_task is required")

    os.environ.setdefault("MUJOCO_GL", "egl")
    deterministic_eval = True
    policy = build_policy(
        checkpoint=Path(checkpoint),
        device=device,
    )

    # Two-stage only: require explicit task slugs from config.
    # (No defaulting to checkpoint dataset manifests.)
    slugs = [str(s) for s in config["task_slugs"]]
    episode_tasks = [slug for slug in slugs for _ in range(int(n_trials_per_task))]
    n_episodes = len(episode_tasks)
    print(f"Two-stage evaluation: {len(slugs)} tasks × {int(n_trials_per_task)} trials = {n_episodes} episodes.")

    seed_plan = _build_seed_plan(root_seed=seed, n_episodes=n_episodes, num_envs=num_envs)
    _set_global_seed(seed_plan.global_seed, deterministic=deterministic_eval)

    wrapper_cls = ImageBasedCGWrapper if policy.image_feature is not None else StateBasedCGWrapper
    t0_probe, _t1_probe = _two_stage_slug_to_stage_tasks(str(slugs[0]))
    env = wrapper_cls(
        make_env_fn=lambda: make_env(t0_probe, horizon=horizon),
        num_envs=num_envs,
        use_relative_coordinates=(policy.env_state_feature is not None),
    )

    fps = int(getattr(env.envs[0], "control_freq", 20))
    print(f"env control_freq={fps} Hz | num_envs={num_envs}")

    probe_tasks = [t0_probe for _ in range(num_envs)]
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
        "eval_task_slugs": slugs,
        "result_config": result_config,
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

        pass
    print("=" * 60)
    print(f"Full results: {out_dir_path / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
