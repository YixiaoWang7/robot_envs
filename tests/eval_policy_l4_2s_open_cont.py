#!/usr/bin/env python
"""
Two-stage CG_L4 evaluation where the policy receives [obj0, cont0, obj1].

Stage 0 remains exact: place obj0 into cont0. Stage 1 is open-container:
placing obj1 into any container except cont0 counts as success.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval_policy_l4_2s import (
    ImageBasedCGWrapper,
    StateBasedCGWrapper,
    _L4_CONTAINERS,
    _build_seed_plan,
    _get_video_frame,
    _make_video_writer,
    _parse_stage_slug,
    _preprocess_images,
    _set_global_seed,
    _set_torch_seed,
    _stage_task_string,
    _validate_two_stage_eval_config,
    build_policy,
    make_env,
)


def _open_cont_stage_tasks(task_slug: str) -> tuple[str, str, list[str]]:
    parts = str(task_slug).strip().lower().split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected two-stage slug with exactly one '__', got: {task_slug!r}")
    obj0, cont0 = _parse_stage_slug(parts[0])
    obj1, _hidden_cont1 = _parse_stage_slug(parts[1])
    allowed_containers = [c for c in _L4_CONTAINERS if c != cont0]
    if not allowed_containers:
        raise ValueError(f"No stage-1 containers remain after excluding stage-0 container {cont0!r}")
    return (
        _stage_task_string(obj_name=obj0, cont_name=cont0),
        _stage_task_string(obj_name=obj1, cont_name=allowed_containers[0]),
        allowed_containers,
    )


def _refresh_obs_for_env(env, obs: dict, env_idx: int) -> None:
    if not (hasattr(env.envs[env_idx], "_get_observations") and hasattr(env, "obs")):
        return
    try:
        env.obs[env_idx] = env.envs[env_idx]._get_observations()
        if hasattr(env, "_compute_observation") and callable(env._compute_observation):
            obs_i = env._compute_observation(env_idx)
            for key in list(obs.keys()):
                obs[key][env_idx] = obs_i[key]
    except Exception:
        pass


def _check_success_any_allowed_container(env_i, allowed_containers: list[str]) -> str | None:
    original_index = int(getattr(env_i, "object_B_index", 0))
    try:
        for cont_name in allowed_containers:
            env_i.object_B_index = int(_L4_CONTAINERS.index(cont_name))
            if hasattr(env_i, "_refresh_task_pointers") and callable(env_i._refresh_task_pointers):
                env_i._refresh_task_pointers()
            try:
                if bool(env_i._check_success()):
                    return cont_name
            except Exception:
                continue
    finally:
        env_i.object_B_index = original_index
        if hasattr(env_i, "_refresh_task_pointers") and callable(env_i._refresh_task_pointers):
            env_i._refresh_task_pointers()
    return None


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
    num_envs = env.num_envs
    if len(episode_tasks) != num_envs:
        raise ValueError(f"episode_tasks must have length {num_envs}, got {len(episode_tasks)}")
    if len(reset_seeds) != num_envs:
        raise ValueError(f"reset_seeds must have length {num_envs}, got {len(reset_seeds)}")

    task_slugs = [str(t) for t in episode_tasks]
    stage0_tasks: list[str] = []
    stage1_probe_tasks: list[str] = []
    allowed_stage1_containers: list[list[str]] = []
    for slug in task_slugs:
        t0, t1_probe, allowed = _open_cont_stage_tasks(slug)
        stage0_tasks.append(t0)
        stage1_probe_tasks.append(t1_probe)
        allowed_stage1_containers.append(allowed)

    obs, _ = env.reset(tasks=stage0_tasks, reset_seeds=reset_seeds)

    done = [False] * num_envs
    sum_rewards = [0.0] * num_envs
    max_rewards = [-1e9] * num_envs
    lengths = [0] * num_envs
    ep_frames = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_success = [False] * num_envs
    stage1_success_container: list[str | None] = [None] * num_envs

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
    current_stage = [0] * num_envs
    stage0_done = [False] * num_envs
    stage1_done = [False] * num_envs

    def _seed_history(i: int) -> None:
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
                    "For open-container two-stage evaluation, the checkpoint processor should use "
                    "processor.task_indices_mode='task_slug_two_stage_open_cont'."
                )
            task_idx_batch = norm_batch["task_indices"].to(device)
            if int(task_idx_batch.shape[-1]) != 3:
                raise ValueError(
                    "Open-container two-stage evaluation expects 3 task indices "
                    f"[obj0, cont0, obj1], got shape {tuple(task_idx_batch.shape)}."
                )

            model_kwargs = {
                "robot_state": norm_batch["state"].to(device),
                "task_indices": task_idx_batch,
                "flow_algo": policy.model.flow_algo,
                "batch_size": len(need_query),
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

            try:
                stage_success = bool(env.envs[i]._check_success())
            except Exception:
                stage_success = False

            if int(current_stage[i]) == 0 and stage_success:
                stage0_done[i] = True
                current_stage[i] = 1
                env.envs[i].update_task(stage1_probe_tasks[i])
                if hasattr(env.envs[i], "_refresh_task_pointers") and callable(env.envs[i]._refresh_task_pointers):
                    env.envs[i]._refresh_task_pointers()
                if hasattr(env, "is_success") and isinstance(getattr(env, "is_success"), list):
                    try:
                        env.is_success[i] = False
                    except Exception:
                        pass
                _refresh_obs_for_env(env, obs, i)
            elif int(current_stage[i]) == 1:
                success_container = _check_success_any_allowed_container(env.envs[i], allowed_stage1_containers[i])
                if success_container is not None:
                    stage1_done[i] = True
                    stage1_success_container[i] = success_container

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

        def _write_video(frames: list[np.ndarray], path: Path) -> None:
            append, close = _make_video_writer(path, fps)
            for frame in frames:
                append(frame)
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
        "stage1_success_container": stage1_success_container,
        "actions": [np.stack(a, axis=0) if a else np.zeros((0, 7), dtype=np.float32) for a in ep_actions],
        "rewards": [np.array(r, dtype=np.float32) for r in ep_rewards],
    }


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
    seed_plan,
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

        print(
            f"\n[batch {batch_ix + 1}/{n_batches}]  "
            f"episodes {ep_offset}-{ep_offset + min(num_envs, remaining) - 1}  "
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
                stage1_success_container=np.array([batch["stage1_success_container"][i] or ""]),
            )

            all_episodes.append(
                {
                    "episode_ix": ep_ix,
                    "task": batch["task"][i],
                    "success": bool(batch["success"][i]),
                    "stage1_success_container": batch["stage1_success_container"][i],
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
        sr = np.mean([e["success"] for e in done_eps]) * 100
        print(
            f"  -> after {len(done_eps)} episodes: "
            f"SR={sr:.1f}%  avg_len={np.mean([e['length'] for e in done_eps]):.1f}"
        )

    all_episodes = all_episodes[:n_episodes]
    elapsed = time.time() - start_time

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
            "n_episodes": len(succs),
            "n_success": int(sum(succs)),
            "pc_success": float(np.mean(succs) * 100),
        }
        if result_config.get("include_reward_statistics", True):
            per_task_stats.update(
                {
                    "avg_sum_reward": float(np.mean(task_rewards[task])),
                    "std_sum_reward": float(np.std(task_rewards[task])),
                    "min_sum_reward": float(np.min(task_rewards[task])),
                    "max_sum_reward": float(np.max(task_rewards[task])),
                }
            )
        if result_config.get("include_action_statistics", True):
            per_task_stats.update(
                {
                    "avg_length": float(np.mean(task_lengths[task])),
                    "std_length": float(np.std(task_lengths[task])),
                    "min_length": int(np.min(task_lengths[task])),
                    "max_length": int(np.max(task_lengths[task])),
                }
            )
        per_task[task] = per_task_stats

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

    info = {
        "aggregated": aggregated,
        "per_task": per_task if result_config.get("save_per_task_breakdown", True) else {},
        "per_episode": all_episodes if result_config.get("save_per_episode_details", True) else [],
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return _validate_two_stage_eval_config(json.load(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-container two-stage policy evaluation on CG_L4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to JSON eval config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
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
    horizon = args.horizon if args.horizon is not None else eval_config.get("horizon", 200)
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
    policy = build_policy(checkpoint=Path(checkpoint), device=device)
    slugs = [str(s) for s in config["task_slugs"]]
    episode_tasks = [slug for slug in slugs for _ in range(int(n_trials_per_task))]
    n_episodes = len(episode_tasks)
    print(
        "Open-container two-stage evaluation: "
        f"{len(slugs)} tasks x {int(n_trials_per_task)} trials = {n_episodes} episodes."
    )

    seed_plan = _build_seed_plan(root_seed=seed, n_episodes=n_episodes, num_envs=num_envs)
    _set_global_seed(seed_plan.global_seed, deterministic=True)

    wrapper_cls = ImageBasedCGWrapper if policy.image_feature is not None else StateBasedCGWrapper
    t0_probe, _t1_probe, _allowed_probe = _open_cont_stage_tasks(str(slugs[0]))
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

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir_path = Path(out_dir) / f"eval_{run_id}"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir_path}")

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
        "stage1_success_rule": "any container except stage0 container",
        "result_config": result_config,
        "reproducibility": {
            "deterministic_torch": True,
            "seed_plan": asdict(seed_plan),
        },
    }
    (out_dir_path / "eval_config_used.json").write_text(
        json.dumps(eval_params, indent=2) + "\n", encoding="utf-8"
    )

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

    agg = info["aggregated"]
    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY  ({agg['n_episodes']} episodes, {agg['eval_s']:.1f}s)")
    print("=" * 60)
    print(f"  Success rate : {agg['pc_success']:.1f}%")
    print(f"  Avg sum rew  : {agg['avg_sum_reward']:.3f}")
    print(f"  Avg max rew  : {agg['avg_max_reward']:.3f}")
    print(f"  Avg length   : {agg['avg_ep_length']:.1f} +/- {agg['std_ep_length']:.1f} steps")
    print()
    if info["per_task"]:
        print("Per-task breakdown:")
        for task_name, stats in sorted(info["per_task"].items()):
            print(f"  {task_name:<40s}  {stats['pc_success']:5.1f}%  ({stats['n_success']}/{stats['n_episodes']})")
            if result_config.get("include_reward_statistics", True):
                print(f"    {'':40s}  avg_rew={stats['avg_sum_reward']:.3f} +/- {stats['std_sum_reward']:.3f}")
            if result_config.get("include_action_statistics", True):
                print(f"    {'':40s}  avg_len={stats['avg_length']:.1f} +/- {stats['std_length']:.1f}")
    print("=" * 60)
    print(f"Full results: {out_dir_path / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
