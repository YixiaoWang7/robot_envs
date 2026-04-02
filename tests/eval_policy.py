#!/usr/bin/env python
"""
Multi-episode policy evaluation on CG_L2.

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
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.CG_L2 import CG_L2
from CG_L2_image_wrapper import ImageBasedCGWrapper
from CG_L2_state_wrapper import StateBasedCGWrapper


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
# Task helpers
# ---------------------------------------------------------------------------

_OBJ_IDX   = {"cross": 0, "cube": 1, "cylinder": 2}
_CONT_IDX  = {"bin": 0, "cup": 1, "plate": 2}

_ALL_TASKS = [
    f"place the {obj} into the {cont}"
    for obj in ("cross", "cube", "cylinder")
    for cont in ("bin", "cup", "plate")
]


def _parse_task_indices(task: str | None) -> tuple[int, int]:
    """Map a CG_L2 language task string → (object_idx, container_idx)."""
    tl = (task or "").lower()
    obj = next((v for k, v in _OBJ_IDX.items() if k in tl), 0)
    cont = next((v for k, v in _CONT_IDX.items() if k in tl), 0)
    
    # obj = 0 
    # cont = 1
    return obj, cont


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
    cam_names = ["agentview", "robot0_eye_in_hand"]
    debug_image = False
    for cam_name, im in zip(cam_names, cams):
        im = im.astype(np.uint8)
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        out.append(np.transpose(im_bgr, (2, 0, 1)))  # (3, H, W) uint8

        # Save debug images for the first call (env_idx == 0) only.
        if debug_image and env_idx == 0:
            import os
            from PIL import Image as _PIL
            debug_dir = "debug_preprocess"
            os.makedirs(debug_dir, exist_ok=True)
            # Save raw RGB via PIL — correct colors if the wrapper outputs proper RGB.
            _PIL.fromarray(im).save(f"{debug_dir}/{cam_name}_rgb.png")
            # Save the BGR version via PIL — colors will look swapped (R↔B) if conversion is correct.
            _PIL.fromarray(im_bgr).save(f"{debug_dir}/{cam_name}_bgr.png")
    
        
    return np.stack(out, axis=0)  # (n_cams, 3, H, W) uint8


def _get_video_frame(env, obs: dict, env_idx: int) -> np.ndarray:
    """Return an RGB uint8 frame for video logging."""
    if "observation.images.agentview" in obs:
        return obs["observation.images.agentview"][env_idx].astype(np.uint8)
    return env.render()[env_idx].astype(np.uint8)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(task: str, *, horizon: int) -> CG_L2:
    strategy = "fixed"
    env_task: str | None = task
    if task == "all":
        strategy = "all"
        env_task = None
    return CG_L2(
        robots="Panda",
        gripper_types="PandaGripper",
        strategy=strategy,
        task=env_task,
        horizon=horizon,
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
) -> dict:
    """
    Run one full batched rollout (all envs until done or horizon).
    Supports both image-conditioned and env_state-conditioned checkpoints.
    """
    num_envs = env.num_envs
    obs, _ = env.reset()

    done = [False] * num_envs
    sum_rewards = [0.0] * num_envs
    max_rewards = [-1e9] * num_envs
    lengths = [0] * num_envs
    tasks = [getattr(env.envs[i], "task", "unknown") for i in range(num_envs)]
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

    task_idx_tensors = []
    for i in range(num_envs):
        obj_i, cont_i = _parse_task_indices(tasks[i])
        task_idx_tensors.append(torch.tensor([[obj_i, cont_i]], dtype=torch.long, device=device))

    state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)]
    env_state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)] if use_env_state else []
    img_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)] if use_images else []
    action_queues: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

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
            # print(model_kwargs)
            with torch.no_grad():
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
    n_episodes: int,
    horizon: int,
    n_execute: int,
    device: str,
    fps: int,
    out_dir: Path,
    max_videos: int,
    seed: int,
) -> dict:
    """
    Run ceil(n_episodes / num_envs) batched rollouts and aggregate metrics.
    Returns the full info dict (also written to eval_summary.json).
    """
    num_envs   = env.num_envs
    n_batches  = math.ceil(n_episodes / num_envs)
    videos_dir = out_dir / "videos"
    rollouts_dir = out_dir / "rollouts"
    videos_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

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
                "seed":        seed + ep_ix,
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
    for ep in all_episodes:
        task_successes[ep["task"]].append(ep["success"])
    for task, succs in task_successes.items():
        per_task[task] = {
            "n_episodes":  len(succs),
            "n_success":   int(sum(succs)),
            "pc_success":  float(np.mean(succs) * 100),
        }

    aggregated = {
        "n_episodes":    len(all_episodes),
        "pc_success":    float(np.mean([e["success"]    for e in all_episodes]) * 100),
        "avg_sum_reward":float(np.mean([e["sum_reward"] for e in all_episodes])),
        "avg_max_reward":float(np.mean([e["max_reward"] for e in all_episodes])),
        "avg_ep_length": float(np.mean([e["length"]     for e in all_episodes])),
        "eval_s":        elapsed,
        "eval_ep_s":     elapsed / max(len(all_episodes), 1),
    }

    info = {
        "aggregated":  aggregated,
        "per_task":    per_task,
        "per_episode": all_episodes,
    }

    (out_dir / "eval_summary.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )

    return info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-episode policy evaluation on CG_L2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    # Evaluation scale
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Total number of episodes to evaluate")
    parser.add_argument("--num-envs", type=int, default=2,
                        help="Parallel environments per batch (must be even)")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Maximum steps per episode")
    # Policy
    parser.add_argument("--n-execute", type=int, default=8,
                        help="Actions to execute per policy query (MPC horizon)")
    # Environment
    parser.add_argument("--task", type=str, default="all",
                        help='Task string or "all" for the wrapper to sample tasks')
    # Output
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--max-videos", type=int, default=10,
                        help="Maximum number of episode videos to save")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    # if args.num_envs % 2 != 0:
    #     raise ValueError("--num-envs must be even (ImageBasedCGWrapper requirement)")

    policy = build_policy(
        checkpoint=Path(args.checkpoint),
        device=args.device,
    )

    # Build the environment wrapper that matches the checkpoint features.
    env_task = args.task if policy.image_feature is None else (args.task if args.task != "all" else _ALL_TASKS[0])
    wrapper_cls = ImageBasedCGWrapper if policy.image_feature is not None else StateBasedCGWrapper
    env = wrapper_cls(
        make_env_fn=lambda: make_env(env_task, horizon=args.horizon),
        num_envs=args.num_envs,
        use_relative_coordinates=(policy.env_state_feature is not None),
    )
    if wrapper_cls is StateBasedCGWrapper:
        env.train_task = "all"
    else:
        env.train_task = args.task

    fps = int(getattr(env.envs[0], "control_freq", 20))
    print(f"env control_freq={fps} Hz | num_envs={args.num_envs}")

    obs, _ = env.reset()
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
    out_dir = Path(args.out_dir) / f"eval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Save args
    (out_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    # Run evaluation
    info = eval_policy(
        env,
        policy,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        n_execute=args.n_execute,
        device=args.device,
        fps=fps,
        out_dir=out_dir,
        max_videos=args.max_videos,
        seed=args.seed,
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
    print(f"  Avg length   : {agg['avg_ep_length']:.1f} steps")
    print()
    print("Per-task breakdown:")
    for task, stats in sorted(info["per_task"].items()):
        print(f"  {task:<40s}  {stats['pc_success']:5.1f}%  ({stats['n_success']}/{stats['n_episodes']})")
    print("=" * 60)
    print(f"Full results: {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
