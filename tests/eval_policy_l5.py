#!/usr/bin/env python
"""
Multi-episode policy evaluation on CG_L5.

Evaluates tasks of the form:
  place the <object> into the <container> then press the <color> button
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "CG"))
sys.path.insert(0, str(_REPO_ROOT / "CG" / "robosuite"))

from CG_L2_image_wrapper import ImageBasedCGWrapper, _transform_to_relative  # type: ignore


OBJECTS = ("cross", "cube", "cylinder", "milk")
CONTAINERS = ("bin", "mug", "plate", "mug_no_handle")
BUTTON_COLORS = ("red", "green", "blue", "yellow")
BUTTON_ENTITY_NAMES = tuple(f"button_{color}" for color in BUTTON_COLORS)
POSED_ENTITY_NAMES = OBJECTS + CONTAINERS + BUTTON_ENTITY_NAMES


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


def _make_video_writer(path: Path, fps: int):
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


def _slug_to_task(slug: str) -> str:
    obj, rest = str(slug).strip().lower().split("_into_", 1)
    cont, color = rest.rsplit("_press_", 1)
    return f"place the {obj} into the {cont} then press the {color} button"


def _validate_l5_slug(slug: str) -> str:
    s = str(slug).strip().lower()
    if "_into_" not in s or "_press_" not in s:
        raise ValueError(f"Expected L5 slug '<obj>_into_<container>_press_<color>', got: {slug!r}")
    obj, rest = s.split("_into_", 1)
    cont, color = rest.rsplit("_press_", 1)
    if obj not in OBJECTS:
        raise ValueError(f"Unknown object in L5 slug {slug!r}: {obj!r}")
    if cont not in CONTAINERS:
        raise ValueError(f"Unknown container in L5 slug {slug!r}: {cont!r}")
    if color not in BUTTON_COLORS:
        raise ValueError(f"Unknown button color in L5 slug {slug!r}: {color!r}")
    return s


def _build_all_task_slugs() -> list[str]:
    # Stable canonical order: objects major, containers minor, colors minor-most.
    return [
        f"{obj}_into_{cont}_press_{color}"
        for obj in OBJECTS
        for cont in CONTAINERS
        for color in BUTTON_COLORS
    ]


def _preprocess_images(obs: dict, env_idx: int) -> np.ndarray:
    import cv2  # type: ignore

    cams = [
        obs["observation.images.agentview"][env_idx],
        obs["observation.images.robot0_eye_in_hand"][env_idx],
    ]
    out = []
    for im in cams:
        im = im.astype(np.uint8)
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        out.append(np.transpose(im_bgr, (2, 0, 1)))
    return np.stack(out, axis=0)


def _get_video_frame(env, obs: dict, env_idx: int) -> np.ndarray:
    if "observation.images.agentview" in obs:
        return obs["observation.images.agentview"][env_idx].astype(np.uint8)
    return env.render()[env_idx].astype(np.uint8)


class L5ImageEvalWrapper(ImageBasedCGWrapper):
    """Batched wrapper with CG_L5 observation layout matching generated L5 data."""

    def _entity_pose_world(self, env, raw: dict, name: str) -> np.ndarray:
        if name in BUTTON_ENTITY_NAMES:
            idx = BUTTON_ENTITY_NAMES.index(name)
            body_id = env.button_body_ids[idx]
            pos = env.sim.data.body_xpos[body_id]
            quat = env.sim.data.body_xquat[body_id]
            quat = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        else:
            pos = raw[f"{name}_pos"]
            quat = raw[f"{name}_quat"]
        return np.concatenate([np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)])

    def _compute_observation(self, idx: int) -> dict:
        raw = self.obs[idx]
        env = self.envs[idx]
        eef_pos = raw["robot0_eef_pos"].astype(np.float32)
        eef_quat = raw["robot0_eef_quat"].astype(np.float32)
        gripper = raw["robot0_gripper_qpos"].astype(np.float32)

        object_poses_world = []
        object_poses_rel = []
        for name in POSED_ENTITY_NAMES:
            pose = self._entity_pose_world(env, raw, name)
            object_poses_world.append(pose)
            if self.use_relative_coordinates:
                object_poses_rel.append(_transform_to_relative(eef_pos, eef_quat, pose))
            else:
                object_poses_rel.append(pose)

        obj_pos = env.sim.data.body_xpos[env.object_A_body_id].astype(np.float32)
        button_pos = env.sim.data.body_xpos[env.target_button_body_id].astype(np.float32)
        environment_state = np.concatenate(
            object_poses_rel + [(obj_pos - eef_pos).astype(np.float32), (button_pos - eef_pos).astype(np.float32)]
        ).astype(np.float32)

        return {
            "observation.state": np.concatenate([eef_pos, eef_quat, gripper]).astype(np.float32),
            "observation.environment_state": environment_state,
            "observation.images.agentview": np.flipud(raw["agentview_image"]),
            "observation.images.robot0_eye_in_hand": np.flipud(raw["robot0_eye_in_hand_image"]),
        }


def make_env(task: str, *, horizon: int):
    from robosuite.controllers import load_composite_controller_config
    from robosuite.environments.manipulation.CG_L5 import CG_L5
    from robosuite.utils.placement_samplers import UniformApartRandomSampler

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    sampler_a = UniformApartRandomSampler(
        name="L5_Object_A_Sampler",
        x_range=[-0.12, 0.20],
        y_range=[-0.20, -0.05],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        min_distance=0.02,
    )
    sampler_b = UniformApartRandomSampler(
        name="L5_Object_B_Sampler",
        x_range=[-0.18, 0.20],
        y_range=[0.05, 0.20],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        min_distance=0.01,
    )
    return CG_L5(
        robots="Panda",
        controller_configs=controller_config,
        gripper_types="PandaGripper",
        task=str(task),
        horizon=horizon,
        hard_reset=False,
        placement_initializer_A=sampler_a,
        placement_initializer_B=sampler_b,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        render_camera=["agentview", "robot0_eye_in_hand"],
    )


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
    del episode_offset
    num_envs = env.num_envs
    if len(episode_tasks) != num_envs:
        raise ValueError(f"episode_tasks must have length {num_envs}, got {len(episode_tasks)}")

    task_slugs = [_validate_l5_slug(t) for t in episode_tasks]
    task_strings = [_slug_to_task(slug) for slug in task_slugs]

    from robosuite.utils.errors import RandomizationError  # noqa: PLC0415

    _MAX_RESET_RETRIES = 5
    for _attempt in range(_MAX_RESET_RETRIES):
        try:
            obs, _ = env.reset(tasks=task_strings, reset_seeds=reset_seeds)
            break
        except RandomizationError:
            if _attempt == _MAX_RESET_RETRIES - 1:
                raise
            reset_seeds = [s + 1 for s in reset_seeds]
            print(
                f"[warn] RandomizationError on reset, retrying with bumped seeds"
                f" (attempt {_attempt + 2}/{_MAX_RESET_RETRIES})"
            )

    done = [False] * num_envs
    sum_rewards = [0.0] * num_envs
    max_rewards = [-1e9] * num_envs
    lengths = [0] * num_envs
    ep_frames = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_success = [False] * num_envs
    success_step: list[int | None] = [None] * num_envs

    if record_video:
        for i in range(num_envs):
            ep_frames[i].append(_get_video_frame(env, obs, i))

    import torch  # type: ignore

    use_images = policy.image_feature is not None
    use_env_state = policy.env_state_feature is not None
    n_obs_steps = int(policy.state_feature.window_size)

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

            bq = len(need_query)
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
                    "For L5 evaluation, use processor.add_task_indices=true and "
                    "processor.task_indices_mode='task_slug_l5'."
                )

            model_kwargs = {
                "robot_state": norm_batch["state"].to(device),
                "task_indices": norm_batch["task_indices"].to(device),
                "flow_algo": policy.model.flow_algo,
                "batch_size": bq,
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
            ep_success[i] = bool(info["is_success"][i])
            if ep_success[i] and success_step[i] is None:
                success_step[i] = step

            step_done = bool(terminated[i] if hasattr(terminated, "__len__") else terminated) or bool(
                truncated[i] if hasattr(truncated, "__len__") else truncated
            )
            if step_done or (success_step[i] is not None and step >= success_step[i] + 10):
                done[i] = True

            if record_video and not done[i]:
                ep_frames[i].append(_get_video_frame(env, obs, i))

        step += 1
        print(
            f"\r  step {step:4d}/{horizon} | done {sum(done)}/{num_envs} | "
            f"running SR={np.mean(ep_success[:num_envs]) * 100:.0f}%",
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
        "task": task_slugs,
        "actions": [np.stack(a, axis=0) if a else np.zeros((0, 7), dtype=np.float32) for a in ep_actions],
        "rewards": [np.array(r, dtype=np.float32) for r in ep_rewards],
    }


def eval_policy(
    env: L5ImageEvalWrapper,
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
    train_task_slugs: set[str] | None = None,
) -> dict:
    n_episodes = len(episode_tasks)
    num_envs = env.num_envs
    n_batches = math.ceil(n_episodes / num_envs)
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
        can_record = max(0, max_videos - n_episodes_rendered)
        record_this_batch = min(can_record, num_envs) > 0
        video_paths_batch = [videos_dir / f"ep{ep_offset + i:03d}_PENDING.mp4" for i in range(num_envs)]

        batch_elapsed = time.time() - start_time
        print(
            f"\n[batch {batch_ix + 1}/{n_batches}] episodes "
            f"{ep_offset}-{ep_offset + min(num_envs, remaining) - 1} (record_video={record_this_batch})"
            f"  [{batch_elapsed:.0f}s elapsed]"
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

        for i in range(num_envs):
            ep_ix = ep_offset + i
            if ep_ix >= n_episodes:
                break
            success_str = "Success" if batch["success"][i] else "Failure"
            safe_task = batch["task"][i].replace(" ", "_")
            final_path = videos_dir / f"ep{ep_ix:03d}_{success_str}_{safe_task}.mp4"
            if record_this_batch and n_episodes_rendered < max_videos:
                video_paths_batch[i].rename(final_path)
                n_episodes_rendered += 1
                video_path_str = str(final_path)
            else:
                video_path_str = None

            np.savez_compressed(
                rollouts_dir / f"ep{ep_ix:03d}.npz",
                actions=batch["actions"][i],
                rewards=batch["rewards"][i],
                success=np.array([batch["success"][i]]),
                task=np.array([batch["task"][i]]),
            )
            all_episodes.append(
                {
                    "episode_ix": ep_ix,
                    "task": batch["task"][i],
                    "success": bool(batch["success"][i]),
                    "sum_reward": float(batch["sum_reward"][i]),
                    "max_reward": float(batch["max_reward"][i]),
                    "length": int(batch["length"][i]),
                    "seed": int(seed_plan.episode_reset_seeds[ep_ix]),
                    "root_seed": int(seed_plan.root_seed),
                    "task_seed": int(seed_plan.episode_task_seeds[ep_ix]),
                    "policy_seed": int(batch_policy_seed),
                    "video": video_path_str,
                }
            )

        done_eps = all_episodes[:n_episodes]
        elapsed_so_far = time.time() - start_time
        ep_per_s = len(done_eps) / max(elapsed_so_far, 1e-6)
        eta = (n_episodes - len(done_eps)) / max(ep_per_s, 1e-6)
        print(
            f"  -> after {len(done_eps)}/{n_episodes} episodes: "
            f"SR={np.mean([e['success'] for e in done_eps]) * 100:.1f}% "
            f"avg_len={np.mean([e['length'] for e in done_eps]):.1f} "
            f"[{elapsed_so_far:.0f}s elapsed, ~{eta:.0f}s remaining]"
        )

    all_episodes = all_episodes[:n_episodes]
    elapsed = time.time() - start_time
    task_successes: dict[str, list[bool]] = defaultdict(list)
    task_rewards: dict[str, list[float]] = defaultdict(list)
    task_lengths: dict[str, list[int]] = defaultdict(list)
    for ep in all_episodes:
        task_successes[ep["task"]].append(ep["success"])
        task_rewards[ep["task"]].append(ep["sum_reward"])
        task_lengths[ep["task"]].append(ep["length"])

    per_task: dict[str, dict] = {}
    for task, succs in task_successes.items():
        stats = {
            "n_episodes": len(succs),
            "n_success": int(sum(succs)),
            "pc_success": float(np.mean(succs) * 100),
        }
        if result_config.get("include_reward_statistics", True):
            stats.update(
                {
                    "avg_sum_reward": float(np.mean(task_rewards[task])),
                    "std_sum_reward": float(np.std(task_rewards[task])),
                    "min_sum_reward": float(np.min(task_rewards[task])),
                    "max_sum_reward": float(np.max(task_rewards[task])),
                }
            )
        if result_config.get("include_action_statistics", True):
            stats.update(
                {
                    "avg_length": float(np.mean(task_lengths[task])),
                    "std_length": float(np.std(task_lengths[task])),
                    "min_length": int(np.min(task_lengths[task])),
                    "max_length": int(np.max(task_lengths[task])),
                }
            )
        per_task[task] = stats

    aggregated = {
        "n_episodes": len(all_episodes),
        "pc_success": float(np.mean([e["success"] for e in all_episodes]) * 100),
        "avg_sum_reward": float(np.mean([e["sum_reward"] for e in all_episodes])),
        "avg_max_reward": float(np.mean([e["max_reward"] for e in all_episodes])),
        "avg_ep_length": float(np.mean([e["length"] for e in all_episodes])),
        "std_ep_length": float(np.std([e["length"] for e in all_episodes])),
        "eval_s": elapsed,
        "eval_ep_s": elapsed / max(len(all_episodes), 1),
    }
    if train_task_slugs:
        train_set = set(train_task_slugs)
        id_succs = [bool(e["success"]) for e in all_episodes if str(e["task"]) in train_set]
        ood_succs = [bool(e["success"]) for e in all_episodes if str(e["task"]) not in train_set]
        if id_succs:
            aggregated["pc_success_in_dist"] = float(np.mean(id_succs) * 100)
            aggregated["n_episodes_in_dist"] = int(len(id_succs))
        if ood_succs:
            aggregated["pc_success_ood"] = float(np.mean(ood_succs) * 100)
            aggregated["n_episodes_ood"] = int(len(ood_succs))
    info = {
        "aggregated": aggregated,
        "per_task": per_task if result_config.get("save_per_task_breakdown", True) else {},
        "per_episode": all_episodes if result_config.get("save_per_episode_details", True) else [],
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "eval_config" not in cfg or "result_config" not in cfg:
        raise ValueError("Eval config missing required keys: 'eval_config' and/or 'result_config'")
    slugs = cfg.get("task_slugs")
    if not isinstance(slugs, list) or not slugs:
        raise ValueError("'task_slugs' must be a non-empty list of L5 task slugs (used as the train/ID set).")
    cfg["task_slugs"] = [_validate_l5_slug(s) for s in slugs]

    train_slugs = cfg.get("train_task_slugs")
    if train_slugs is not None:
        if not isinstance(train_slugs, list) or not train_slugs:
            raise ValueError("'train_task_slugs' must be a non-empty list when provided.")
        cfg["train_task_slugs"] = [_validate_l5_slug(s) for s in train_slugs]

    eval_all = cfg.get("eval_all_tasks", False)
    if not isinstance(eval_all, bool):
        raise ValueError("'eval_all_tasks' must be a boolean when provided.")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Multi-episode policy evaluation on CG_L5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-trials-per-task", "--n_trials_per_task", dest="n_trials_per_task", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-execute", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_config = config.get("eval_config", {})
    result_config = config.get("result_config", {})

    checkpoint = args.checkpoint or eval_config.get("checkpoint")
    device = args.device or eval_config.get("device", "cuda")
    num_envs = args.num_envs if args.num_envs is not None else eval_config.get("num_envs", 2)
    horizon = args.horizon if args.horizon is not None else eval_config.get("horizon", 500)
    n_execute = args.n_execute if args.n_execute is not None else eval_config.get("n_execute", 8)
    out_dir = args.out_dir or eval_config.get("out_dir", "artifacts")
    max_videos = args.max_videos if args.max_videos is not None else eval_config.get("max_videos", 10)
    seed = args.seed if args.seed is not None else eval_config.get("seed", 0)
    n_trials_per_task = args.n_trials_per_task if args.n_trials_per_task is not None else eval_config.get("n_trials_per_task")
    if not checkpoint:
        raise ValueError("eval_config.checkpoint (or --checkpoint) is required")
    if n_trials_per_task is None:
        raise ValueError("eval_config.n_trials_per_task is required")

    os.environ.setdefault("MUJOCO_GL", "egl")
    seed_plan = _build_seed_plan(
        root_seed=int(seed),
        n_episodes=len(config["task_slugs"]) * int(n_trials_per_task),
        num_envs=int(num_envs),
    )
    _set_global_seed(seed_plan.global_seed, deterministic=True)

    policy = build_policy(checkpoint=Path(checkpoint), device=device)
    train_slugs = [str(s) for s in (config.get("train_task_slugs") or config["task_slugs"])]
    train_set = set(train_slugs)

    if bool(config.get("eval_all_tasks", False)):
        eval_slugs = _build_all_task_slugs()
        eval_mode = "all_64"
    else:
        eval_slugs = [str(s) for s in config["task_slugs"]]
        eval_mode = "config"

    episode_tasks = [slug for slug in eval_slugs for _ in range(int(n_trials_per_task))]
    print(
        f"CG_L5 evaluation ({eval_mode}): {len(eval_slugs)} tasks x {int(n_trials_per_task)} trials "
        f"= {len(episode_tasks)} episodes."
    )

    t_probe = _slug_to_task(eval_slugs[0])
    env = L5ImageEvalWrapper(
        make_env_fn=lambda: make_env(t_probe, horizon=horizon),
        num_envs=int(num_envs),
        use_relative_coordinates=(policy.env_state_feature is not None),
    )
    fps = int(getattr(env.envs[0], "control_freq", 20))
    print(f"env control_freq={fps} Hz | num_envs={num_envs}")

    probe_tasks = [_slug_to_task(eval_slugs[0]) for _ in range(int(num_envs))]
    probe_reset_seeds = [int(seed_plan.global_seed + i) for i in range(int(num_envs))]
    obs, _ = env.reset(tasks=probe_tasks, reset_seeds=probe_reset_seeds)
    policy_state_dim = int(np.prod(policy.state_feature.shape))
    env_state_dim = int(obs["observation.state"].shape[-1])
    if policy_state_dim != env_state_dim:
        env.close()
        raise ValueError(f"State dim mismatch: policy expects {policy_state_dim}, env provides {env_state_dim}.")
    if policy.env_state_feature is not None:
        policy_env_state_dim = int(np.prod(policy.env_state_feature.shape))
        env_env_state_dim = int(obs["observation.environment_state"].shape[-1])
        if policy_env_state_dim != env_env_state_dim:
            env.close()
            raise ValueError(
                f"Env-state dim mismatch: policy expects {policy_env_state_dim}, env provides {env_env_state_dim}."
            )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir_path = Path(out_dir) / f"eval_{run_id}"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir_path}")
    eval_params = {
        "checkpoint": checkpoint,
        "device": device,
        "n_episodes": len(episode_tasks),
        "n_trials_per_task": int(n_trials_per_task),
        "num_envs": int(num_envs),
        "horizon": int(horizon),
        "n_execute": int(n_execute),
        "max_videos": int(max_videos),
        "seed": int(seed),
        "eval_task_slugs": eval_slugs,
        "eval_mode": eval_mode,
        "train_task_slugs": train_slugs,
        "result_config": result_config,
        "reproducibility": {"deterministic_torch": True, "seed_plan": asdict(seed_plan)},
    }
    (out_dir_path / "eval_config_used.json").write_text(json.dumps(eval_params, indent=2) + "\n", encoding="utf-8")

    info = eval_policy(
        env,
        policy,
        episode_tasks=episode_tasks,
        horizon=int(horizon),
        n_execute=int(n_execute),
        device=str(device),
        fps=fps,
        out_dir=out_dir_path,
        max_videos=int(max_videos),
        result_config=result_config,
        seed_plan=seed_plan,
        train_task_slugs=train_set,
    )
    env.close()

    agg = info["aggregated"]
    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY ({agg['n_episodes']} episodes, {agg['eval_s']:.1f}s)")
    print("=" * 60)
    print(f"  Success rate : {agg['pc_success']:.1f}%")
    if "pc_success_in_dist" in agg:
        print(f"  ID success   : {agg['pc_success_in_dist']:.1f}%  ({agg.get('n_episodes_in_dist', 0)} episodes)")
    if "pc_success_ood" in agg:
        print(f"  OOD success  : {agg['pc_success_ood']:.1f}%  ({agg.get('n_episodes_ood', 0)} episodes)")
    print(f"  Avg sum rew  : {agg['avg_sum_reward']:.3f}")
    print(f"  Avg max rew  : {agg['avg_max_reward']:.3f}")
    print(f"  Avg length   : {agg['avg_ep_length']:.1f} +/- {agg['std_ep_length']:.1f} steps")
    print("=" * 60)
    print(f"Full results: {out_dir_path / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
