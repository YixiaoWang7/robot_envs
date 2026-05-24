#!/usr/bin/env python
"""
Two-stage CG_L4 evaluation where the policy receives [obj0, obj1, cont1].

Stage 0 is open-container: placing obj0 into any container except cont1 counts
as success.  Stage 1 is exact: placing obj1 into cont1.

Failure-mode metrics tracked per episode:
  - stage0_done / stage1_done
  - stage0_success_container (which container obj0 reached)
  - stage0_step / stage1_step (timestep of completion, -1 if not reached)
  - same_container_violation (obj0 placed into cont1)
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
    _L4_OBJECTS,
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


# ---------------------------------------------------------------------------
# Task helpers  (inverted from open_cont: stage 0 is open, stage 1 is exact)
# ---------------------------------------------------------------------------

def _infer_cont0_stage_tasks(task_slug: str) -> tuple[str, str, str, str, str, list[str]]:
    """Return (stage0_probe_task, stage1_task, obj0_name, obj1_name, cont1_name, allowed_stage0_containers)."""
    parts = str(task_slug).strip().lower().split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected two-stage slug with exactly one '__', got: {task_slug!r}")
    obj0, _hidden_cont0 = _parse_stage_slug(parts[0])
    obj1, cont1 = _parse_stage_slug(parts[1])
    allowed_stage0 = [c for c in _L4_CONTAINERS if c != cont1]
    if not allowed_stage0:
        raise ValueError(f"No stage-0 containers remain after excluding stage-1 container {cont1!r}")
    return (
        _stage_task_string(obj_name=obj0, cont_name=allowed_stage0[0]),
        _stage_task_string(obj_name=obj1, cont_name=cont1),
        obj0,
        obj1,
        cont1,
        allowed_stage0,
    )


def _infer_cont0_visible_key(task_slug: str) -> str:
    """Key for the policy-visible task [obj0, obj1, cont1], ignoring hidden cont0."""
    parts = str(task_slug).strip().lower().split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected two-stage slug with exactly one '__', got: {task_slug!r}")
    obj0, _hidden_cont0 = _parse_stage_slug(parts[0])
    obj1, cont1 = _parse_stage_slug(parts[1])
    return f"{obj0}__{obj1}_into_{cont1}"


def _materialize_hidden_cont0(task_slug: str, cont0_name: str) -> str:
    parts = str(task_slug).strip().lower().split("__")
    if len(parts) != 2:
        raise ValueError(f"Expected two-stage slug with exactly one '__', got: {task_slug!r}")
    obj0, _hidden_cont0 = _parse_stage_slug(parts[0])
    obj1, cont1 = _parse_stage_slug(parts[1])
    if cont0_name == cont1:
        raise ValueError(f"Hidden cont0 must differ from cont1 for infer-cont0 eval, got {task_slug!r}")
    return f"{obj0}_into_{cont0_name}__{obj1}_into_{cont1}"


def _materialize_even_hidden_cont0(task_slug: str, *, task_ix: int, trial_ix: int, seed: int) -> str:
    """Evenly cycle hidden cont0 over all containers except cont1 for a visible query."""
    _t0_probe, _t1, _obj0, _obj1, cont1, allowed = _infer_cont0_stage_tasks(task_slug)
    offset = (int(seed) + int(task_ix)) % len(allowed)
    cont0_name = allowed[(int(trial_ix) + offset) % len(allowed)]
    return _materialize_hidden_cont0(task_slug, cont0_name)


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
    """Cycle through allowed containers and return the name of the first success hit."""
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


def _check_success_specific_container(env_i, cont_name: str) -> bool:
    """Check if current env task succeeds when targeting a specific container."""
    original_index = int(getattr(env_i, "object_B_index", 0))
    try:
        env_i.object_B_index = int(_L4_CONTAINERS.index(cont_name))
        if hasattr(env_i, "_refresh_task_pointers") and callable(env_i._refresh_task_pointers):
            env_i._refresh_task_pointers()
        try:
            return bool(env_i._check_success())
        except Exception:
            return False
    finally:
        env_i.object_B_index = original_index
        if hasattr(env_i, "_refresh_task_pointers") and callable(env_i._refresh_task_pointers):
            env_i._refresh_task_pointers()


def _check_grasp_object(env_i, obj_name: str) -> bool:
    """Return True if the gripper is currently grasping the named object."""
    obj_idx = _L4_OBJECTS.index(obj_name)
    try:
        return bool(env_i._check_grasp(
            gripper=env_i.robots[0].gripper,
            object_geoms=env_i.object_A_list[obj_idx],
        ))
    except Exception:
        return False


def _check_obj_in_any_container(env_i, obj_name: str) -> str | None:
    """Check if obj is placed into any container; return the container name or None."""
    original_a = int(getattr(env_i, "object_A_index", 0))
    original_b = int(getattr(env_i, "object_B_index", 0))
    obj_idx = _L4_OBJECTS.index(obj_name)
    try:
        env_i.object_A_index = obj_idx
        for ci, cont_name in enumerate(_L4_CONTAINERS):
            env_i.object_B_index = ci
            if hasattr(env_i, "_refresh_task_pointers") and callable(env_i._refresh_task_pointers):
                env_i._refresh_task_pointers()
            try:
                if bool(env_i._check_success()):
                    return cont_name
            except Exception:
                continue
    finally:
        env_i.object_A_index = original_a
        env_i.object_B_index = original_b
        if hasattr(env_i, "_refresh_task_pointers") and callable(env_i._refresh_task_pointers):
            env_i._refresh_task_pointers()
    return None


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
    num_envs = env.num_envs
    if len(episode_tasks) != num_envs:
        raise ValueError(f"episode_tasks must have length {num_envs}, got {len(episode_tasks)}")
    if len(reset_seeds) != num_envs:
        raise ValueError(f"reset_seeds must have length {num_envs}, got {len(reset_seeds)}")

    task_slugs = [str(t) for t in episode_tasks]
    stage0_probe_tasks: list[str] = []
    stage1_tasks: list[str] = []
    obj0_names: list[str] = []
    obj1_names: list[str] = []
    cont1_names: list[str] = []
    allowed_stage0_containers: list[list[str]] = []
    for slug in task_slugs:
        t0_probe, t1, o0, o1, c1, allowed = _infer_cont0_stage_tasks(slug)
        stage0_probe_tasks.append(t0_probe)
        stage1_tasks.append(t1)
        obj0_names.append(o0)
        obj1_names.append(o1)
        cont1_names.append(c1)
        allowed_stage0_containers.append(allowed)

    obs, _ = env.reset(tasks=stage0_probe_tasks, reset_seeds=reset_seeds)

    done = [False] * num_envs
    sum_rewards = [0.0] * num_envs
    max_rewards = [-1e9] * num_envs
    lengths = [0] * num_envs
    ep_frames = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_success = [False] * num_envs

    stage0_done = [False] * num_envs
    stage1_done = [False] * num_envs
    stage0_success_container: list[str | None] = [None] * num_envs
    stage0_step: list[int] = [-1] * num_envs
    stage1_step: list[int] = [-1] * num_envs
    same_container_violation: list[bool] = [False] * num_envs
    current_stage = [0] * num_envs

    pick_obj0: list[bool] = [False] * num_envs
    pick_obj0_step: list[int] = [-1] * num_envs
    pick_obj1: list[bool] = [False] * num_envs
    pick_obj1_step: list[int] = [-1] * num_envs
    stage1_actual_container: list[str | None] = [None] * num_envs

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

    debug_printed_eps: set[int] = set()

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
                    "For infer-cont0 two-stage evaluation, the checkpoint processor should use "
                    "processor.task_indices_mode='task_slug_two_stage_infer_cont0'."
                )
            task_idx_batch = norm_batch["task_indices"].to(device)
            if int(task_idx_batch.shape[-1]) != 3:
                raise ValueError(
                    "Infer-cont0 two-stage evaluation expects 3 task indices "
                    f"[obj0, obj1, cont1], got shape {tuple(task_idx_batch.shape)}."
                )
            task_idx_cpu = task_idx_batch.detach().cpu().tolist()
            for qi, i in enumerate(need_query):
                ep_ix = episode_offset + i
                if ep_ix in debug_printed_eps:
                    continue
                debug_printed_eps.add(ep_ix)
                obj0_name = obj0_names[i]
                obj1_name = obj1_names[i]
                cont1_name = cont1_names[i]
                print(
                    "[task-input-debug] "
                    f"ep={ep_ix} slug={task_slugs[i]!r} "
                    f"hidden_cont0_in_slug={_parse_stage_slug(task_slugs[i].split('__', 1)[0])[1]!r} "
                    f"model_task_indices={task_idx_cpu[qi]} "
                    f"decoded_model_input=[obj0={obj0_name}, obj1={obj1_name}, cont1={cont1_name}] "
                    f"stage0_allowed={allowed_stage0_containers[i]}"
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

            if int(current_stage[i]) == 0:
                # Grasp detection for obj0
                if not pick_obj0[i]:
                    if _check_grasp_object(env.envs[i], obj0_names[i]):
                        pick_obj0[i] = True
                        pick_obj0_step[i] = step + 1

                # Stage 0: open-container check -- obj0 into any container except cont1
                s0_hit = _check_success_any_allowed_container(env.envs[i], allowed_stage0_containers[i])
                if s0_hit is not None:
                    stage0_done[i] = True
                    stage0_success_container[i] = s0_hit
                    stage0_step[i] = step + 1
                    if not pick_obj0[i]:
                        pick_obj0[i] = True
                        pick_obj0_step[i] = step + 1
                    current_stage[i] = 1
                    env.envs[i].update_task(stage1_tasks[i])
                    if hasattr(env.envs[i], "_refresh_task_pointers") and callable(env.envs[i]._refresh_task_pointers):
                        env.envs[i]._refresh_task_pointers()
                    if hasattr(env, "is_success") and isinstance(getattr(env, "is_success"), list):
                        try:
                            env.is_success[i] = False
                        except Exception:
                            pass
                    _refresh_obs_for_env(env, obs, i)
                else:
                    # Also check if obj0 was placed into cont1 (violation)
                    if _check_success_specific_container(env.envs[i], cont1_names[i]):
                        same_container_violation[i] = True
                        stage0_done[i] = True
                        stage0_success_container[i] = cont1_names[i]
                        stage0_step[i] = step + 1
                        if not pick_obj0[i]:
                            pick_obj0[i] = True
                            pick_obj0_step[i] = step + 1
            elif int(current_stage[i]) == 1:
                # Grasp detection for obj1
                if not pick_obj1[i]:
                    if _check_grasp_object(env.envs[i], obj1_names[i]):
                        pick_obj1[i] = True
                        pick_obj1_step[i] = step + 1

                # Stage 1: check if obj1 placed into any container
                if stage1_actual_container[i] is None:
                    placed = _check_obj_in_any_container(env.envs[i], obj1_names[i])
                    if placed is not None:
                        stage1_actual_container[i] = placed
                        if not pick_obj1[i]:
                            pick_obj1[i] = True
                            pick_obj1_step[i] = step + 1
                        if placed == cont1_names[i]:
                            stage1_done[i] = True
                            stage1_step[i] = step + 1

            ep_success[i] = bool(stage0_done[i] and stage1_done[i] and not same_container_violation[i])

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
        "stage0_done": stage0_done,
        "stage1_done": stage1_done,
        "stage0_success_container": stage0_success_container,
        "stage0_step": stage0_step,
        "stage1_step": stage1_step,
        "same_container_violation": same_container_violation,
        "cont1_name": cont1_names,
        "pick_obj0": pick_obj0,
        "pick_obj0_step": pick_obj0_step,
        "pick_obj1": pick_obj1,
        "pick_obj1_step": pick_obj1_step,
        "stage1_actual_container": stage1_actual_container,
        "actions": [np.stack(a, axis=0) if a else np.zeros((0, 7), dtype=np.float32) for a in ep_actions],
        "rewards": [np.array(r, dtype=np.float32) for r in ep_rewards],
    }


# ---------------------------------------------------------------------------
# Multi-episode evaluator with failure-mode metrics
# ---------------------------------------------------------------------------

def _aggregate_failure_mode_metrics(episodes: list[dict], *, elapsed: float | None = None) -> dict:
    n = len(episodes)

    n_pick_obj0 = sum(1 for e in episodes if e["pick_obj0"])
    n_stage0 = sum(1 for e in episodes if e["stage0_done"])
    n_stage0_correct = sum(1 for e in episodes if e["stage0_done"] and not e["same_container_violation"])
    n_same_cont = sum(1 for e in episodes if e["same_container_violation"])

    n_pick_obj1 = sum(1 for e in episodes if e["pick_obj1"])
    n_place_obj1 = sum(1 for e in episodes if e["stage1_actual_container"] is not None)
    n_stage1 = sum(1 for e in episodes if e["stage1_done"])
    n_place_obj1_wrong = sum(
        1 for e in episodes
        if e["stage1_actual_container"] is not None and e["stage1_actual_container"] != e["cont1_name"]
    )

    n_success = sum(1 for e in episodes if e["success"])
    n_stage0_only = sum(1 for e in episodes if e["stage0_done"] and not e["stage1_done"] and not e["same_container_violation"])
    n_no_stage0 = sum(1 for e in episodes if not e["stage0_done"])

    stage0_container_counts: dict[str, int] = defaultdict(int)
    for e in episodes:
        c = e["stage0_success_container"]
        if c is not None:
            stage0_container_counts[c] += 1

    stage1_container_counts: dict[str, int] = defaultdict(int)
    for e in episodes:
        c = e["stage1_actual_container"]
        if c is not None:
            stage1_container_counts[c] += 1

    s0_steps = [e["stage0_step"] for e in episodes if e["stage0_step"] >= 0]
    s1_steps = [e["stage1_step"] for e in episodes if e["stage1_step"] >= 0]
    pick0_steps = [e["pick_obj0_step"] for e in episodes if e["pick_obj0_step"] >= 0]
    pick1_steps = [e["pick_obj1_step"] for e in episodes if e["pick_obj1_step"] >= 0]

    aggregated = {
        "n_episodes": n,
        "n_success": n_success,
        "pc_success": float(n_success / max(n, 1) * 100),
        # Stage 0 funnel: pick -> place -> place_correct
        "pc_pick_obj0": float(n_pick_obj0 / max(n, 1) * 100),
        "pc_place_obj0": float(n_stage0 / max(n, 1) * 100),
        "pc_place_obj0_correct": float(n_stage0_correct / max(n, 1) * 100),
        "pc_same_container_violation": float(n_same_cont / max(n, 1) * 100),
        # Stage 1 funnel: pick -> place -> place_correct
        "pc_pick_obj1": float(n_pick_obj1 / max(n, 1) * 100),
        "pc_place_obj1": float(n_place_obj1 / max(n, 1) * 100),
        "pc_place_obj1_correct": float(n_stage1 / max(n, 1) * 100),
        "pc_place_obj1_wrong": float(n_place_obj1_wrong / max(n, 1) * 100),
        # Legacy aliases
        "pc_stage0_success": float(n_stage0 / max(n, 1) * 100),
        "pc_stage1_success": float(n_stage1 / max(n, 1) * 100),
        "pc_stage0_only": float(n_stage0_only / max(n, 1) * 100),
        "pc_no_stage0": float(n_no_stage0 / max(n, 1) * 100),
        "avg_sum_reward": float(np.mean([e["sum_reward"] for e in episodes])) if episodes else 0.0,
        "avg_max_reward": float(np.mean([e["max_reward"] for e in episodes])) if episodes else 0.0,
        "avg_ep_length": float(np.mean([e["length"] for e in episodes])) if episodes else 0.0,
        "std_ep_length": float(np.std([e["length"] for e in episodes])) if episodes else 0.0,
    }
    if elapsed is not None:
        aggregated["eval_s"] = float(elapsed)
        aggregated["eval_ep_s"] = float(elapsed / max(n, 1))

    for cont in _L4_CONTAINERS:
        aggregated[f"pc_stage0_to_{cont}"] = float(stage0_container_counts.get(cont, 0) / max(n, 1) * 100)
    for cont in _L4_CONTAINERS:
        aggregated[f"pc_stage1_to_{cont}"] = float(stage1_container_counts.get(cont, 0) / max(n, 1) * 100)
    if pick0_steps:
        aggregated["avg_pick_obj0_step"] = float(np.mean(pick0_steps))
    if s0_steps:
        aggregated["avg_stage0_step"] = float(np.mean(s0_steps))
    if pick1_steps:
        aggregated["avg_pick_obj1_step"] = float(np.mean(pick1_steps))
    if s1_steps:
        aggregated["avg_stage1_step"] = float(np.mean(s1_steps))
    return aggregated


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
    train_task_slugs: list[str] | None = None,
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
    train_task_slug_set = (
        {_infer_cont0_visible_key(str(s)) for s in train_task_slugs}
        if train_task_slugs is not None
        else None
    )

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
                stage0_success_container=np.array([batch["stage0_success_container"][i] or ""]),
            )

            all_episodes.append(
                {
                    "episode_ix": ep_ix,
                    "task": batch["task"][i],
                    "visible_task": _infer_cont0_visible_key(batch["task"][i]),
                    "hidden_cont0": _parse_stage_slug(batch["task"][i].split("__", 1)[0])[1],
                    "split": (
                        "unknown" if train_task_slug_set is None
                        else "id" if _infer_cont0_visible_key(batch["task"][i]) in train_task_slug_set
                        else "ood"
                    ),
                    "success": bool(batch["success"][i]),
                    "stage0_done": bool(batch["stage0_done"][i]),
                    "stage1_done": bool(batch["stage1_done"][i]),
                    "stage0_success_container": batch["stage0_success_container"][i],
                    "stage0_step": int(batch["stage0_step"][i]),
                    "stage1_step": int(batch["stage1_step"][i]),
                    "same_container_violation": bool(batch["same_container_violation"][i]),
                    "cont1_name": batch["cont1_name"][i],
                    "pick_obj0": bool(batch["pick_obj0"][i]),
                    "pick_obj0_step": int(batch["pick_obj0_step"][i]),
                    "pick_obj1": bool(batch["pick_obj1"][i]),
                    "pick_obj1_step": int(batch["pick_obj1_step"][i]),
                    "stage1_actual_container": batch["stage1_actual_container"][i],
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

    # ------------------------------------------------------------------
    # Aggregate failure-mode metrics
    # ------------------------------------------------------------------
    aggregated = _aggregate_failure_mode_metrics(all_episodes, elapsed=elapsed)
    aggregated_by_split: dict[str, dict] = {}
    if train_task_slug_set is not None:
        split_episodes = {
            "id": [e for e in all_episodes if e["split"] == "id"],
            "ood": [e for e in all_episodes if e["split"] == "ood"],
        }
        aggregated_by_split = {
            split_name: _aggregate_failure_mode_metrics(eps)
            for split_name, eps in split_episodes.items()
        }
        aggregated["n_id_episodes"] = aggregated_by_split["id"]["n_episodes"]
        aggregated["n_ood_episodes"] = aggregated_by_split["ood"]["n_episodes"]

    # ------------------------------------------------------------------
    # Per-task breakdown with failure modes
    # ------------------------------------------------------------------
    per_task: dict[str, dict] = {}
    task_episodes: dict[str, list[dict]] = defaultdict(list)
    for ep in all_episodes:
        task_episodes[ep["visible_task"]].append(ep)

    for task, eps in task_episodes.items():
        nt = len(eps)
        t_success = sum(1 for e in eps if e["success"])
        t_pick0 = sum(1 for e in eps if e["pick_obj0"])
        t_s0 = sum(1 for e in eps if e["stage0_done"])
        t_s0_correct = sum(1 for e in eps if e["stage0_done"] and not e["same_container_violation"])
        t_same = sum(1 for e in eps if e["same_container_violation"])
        t_pick1 = sum(1 for e in eps if e["pick_obj1"])
        t_place1 = sum(1 for e in eps if e["stage1_actual_container"] is not None)
        t_s1 = sum(1 for e in eps if e["stage1_done"])
        t_place1_wrong = sum(
            1 for e in eps
            if e["stage1_actual_container"] is not None and e["stage1_actual_container"] != e["cont1_name"]
        )
        t_s0_only = sum(1 for e in eps if e["stage0_done"] and not e["stage1_done"] and not e["same_container_violation"])
        t_no_s0 = sum(1 for e in eps if not e["stage0_done"])

        per_task_stats: dict = {
            "n_episodes": nt,
            "split": eps[0].get("split", "unknown") if eps else "unknown",
            "n_success": t_success,
            "pc_success": float(t_success / max(nt, 1) * 100),
            "pc_pick_obj0": float(t_pick0 / max(nt, 1) * 100),
            "pc_place_obj0": float(t_s0 / max(nt, 1) * 100),
            "pc_place_obj0_correct": float(t_s0_correct / max(nt, 1) * 100),
            "pc_same_container_violation": float(t_same / max(nt, 1) * 100),
            "pc_pick_obj1": float(t_pick1 / max(nt, 1) * 100),
            "pc_place_obj1": float(t_place1 / max(nt, 1) * 100),
            "pc_place_obj1_correct": float(t_s1 / max(nt, 1) * 100),
            "pc_place_obj1_wrong": float(t_place1_wrong / max(nt, 1) * 100),
            "pc_stage0_success": float(t_s0 / max(nt, 1) * 100),
            "pc_stage1_success": float(t_s1 / max(nt, 1) * 100),
            "pc_stage0_only": float(t_s0_only / max(nt, 1) * 100),
            "pc_no_stage0": float(t_no_s0 / max(nt, 1) * 100),
        }
        # Per-container placement for stage 0
        t_cont0: dict[str, int] = defaultdict(int)
        for e in eps:
            c = e["stage0_success_container"]
            if c is not None:
                t_cont0[c] += 1
        for cont in _L4_CONTAINERS:
            per_task_stats[f"pc_stage0_to_{cont}"] = float(t_cont0.get(cont, 0) / max(nt, 1) * 100)
        # Per-container placement for stage 1
        t_cont1: dict[str, int] = defaultdict(int)
        for e in eps:
            c = e["stage1_actual_container"]
            if c is not None:
                t_cont1[c] += 1
        for cont in _L4_CONTAINERS:
            per_task_stats[f"pc_stage1_to_{cont}"] = float(t_cont1.get(cont, 0) / max(nt, 1) * 100)

        if result_config.get("include_reward_statistics", True):
            rews = [e["sum_reward"] for e in eps]
            per_task_stats.update({
                "avg_sum_reward": float(np.mean(rews)),
                "std_sum_reward": float(np.std(rews)),
                "min_sum_reward": float(np.min(rews)),
                "max_sum_reward": float(np.max(rews)),
            })
        if result_config.get("include_action_statistics", True):
            lens = [e["length"] for e in eps]
            per_task_stats.update({
                "avg_length": float(np.mean(lens)),
                "std_length": float(np.std(lens)),
                "min_length": int(np.min(lens)),
                "max_length": int(np.max(lens)),
            })
        per_task[task] = per_task_stats

    info = {
        "aggregated": aggregated,
        "aggregated_by_split": aggregated_by_split,
        "per_task": per_task if result_config.get("save_per_task_breakdown", True) else {},
        "per_episode": all_episodes if result_config.get("save_per_episode_details", True) else [],
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return _validate_two_stage_eval_config(json.load(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer-cont0 two-stage policy evaluation on CG_L4 (policy sees [obj0, obj1, cont1])",
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
    train_task_slugs = [str(s) for s in config.get("train_task_slugs", [])]
    episode_tasks = [
        _materialize_even_hidden_cont0(
            slug,
            task_ix=task_ix,
            trial_ix=trial_ix,
            seed=int(seed),
        )
        for task_ix, slug in enumerate(slugs)
        for trial_ix in range(int(n_trials_per_task))
    ]
    n_episodes = len(episode_tasks)
    print(
        "Infer-cont0 two-stage evaluation: "
        f"{len(slugs)} tasks x {int(n_trials_per_task)} trials = {n_episodes} episodes."
    )
    print("Hidden cont0 eval sampling: evenly cycles over all containers except cont1 for each visible task.")
    if train_task_slugs:
        train_visible_keys = {_infer_cont0_visible_key(slug) for slug in train_task_slugs}
        n_id_tasks = sum(1 for slug in slugs if _infer_cont0_visible_key(slug) in train_visible_keys)
        print(f"ID/OOD split: {n_id_tasks} ID tasks, {len(slugs) - n_id_tasks} OOD tasks.")

    seed_plan = _build_seed_plan(root_seed=seed, n_episodes=n_episodes, num_envs=num_envs)
    _set_global_seed(seed_plan.global_seed, deterministic=True)

    wrapper_cls = ImageBasedCGWrapper if policy.image_feature is not None else StateBasedCGWrapper
    t0_probe, _t1, _o0, _o1, _c1, _allowed = _infer_cont0_stage_tasks(str(slugs[0]))
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
        "materialized_episode_task_slugs": episode_tasks,
        "train_task_slugs": train_task_slugs,
        "ood_task_slugs": [
            slug for slug in slugs
            if _infer_cont0_visible_key(slug) not in {_infer_cont0_visible_key(s) for s in train_task_slugs}
        ],
        "hidden_cont0_sampling": "even_cycle_over_containers_except_cont1",
        "stage0_success_rule": "any container except stage1 container (cont1)",
        "stage1_success_rule": "exact container (cont1)",
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
        train_task_slugs=train_task_slugs if train_task_slugs else None,
    )

    env.close()

    agg = info["aggregated"]
    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY  ({agg['n_episodes']} episodes, {agg['eval_s']:.1f}s)")
    print("=" * 60)
    print(f"  Overall success        : {agg['pc_success']:.1f}%")
    print(f"  --- Stage 0 (obj0) ---")
    print(f"  Pick obj0              : {agg['pc_pick_obj0']:.1f}%")
    print(f"  Place obj0 (any cont)  : {agg['pc_place_obj0']:.1f}%")
    print(f"  Place obj0 correct     : {agg['pc_place_obj0_correct']:.1f}%")
    print(f"  Same-cont violation    : {agg['pc_same_container_violation']:.1f}%")
    for cont in _L4_CONTAINERS:
        k = f"pc_stage0_to_{cont}"
        if k in agg:
            print(f"    Stage0 -> {cont:<14s}: {agg[k]:.1f}%")
    print(f"  --- Stage 1 (obj1) ---")
    print(f"  Pick obj1              : {agg['pc_pick_obj1']:.1f}%")
    print(f"  Place obj1 (any cont)  : {agg['pc_place_obj1']:.1f}%")
    print(f"  Place obj1 correct     : {agg['pc_place_obj1_correct']:.1f}%")
    print(f"  Place obj1 wrong       : {agg['pc_place_obj1_wrong']:.1f}%")
    for cont in _L4_CONTAINERS:
        k = f"pc_stage1_to_{cont}"
        if k in agg:
            print(f"    Stage1 -> {cont:<14s}: {agg[k]:.1f}%")
    print(f"  --- Timing ---")
    if "avg_pick_obj0_step" in agg:
        print(f"  Avg pick obj0 step     : {agg['avg_pick_obj0_step']:.1f}")
    if "avg_stage0_step" in agg:
        print(f"  Avg stage0 step        : {agg['avg_stage0_step']:.1f}")
    if "avg_pick_obj1_step" in agg:
        print(f"  Avg pick obj1 step     : {agg['avg_pick_obj1_step']:.1f}")
    if "avg_stage1_step" in agg:
        print(f"  Avg stage1 step        : {agg['avg_stage1_step']:.1f}")
    print(f"  Avg sum rew            : {agg['avg_sum_reward']:.3f}")
    print(f"  Avg length             : {agg['avg_ep_length']:.1f} +/- {agg['std_ep_length']:.1f} steps")
    if info.get("aggregated_by_split"):
        print(f"  --- ID/OOD split ---")
        for split_name in ("id", "ood"):
            split = info["aggregated_by_split"].get(split_name)
            if not split:
                continue
            print(
                f"  {split_name.upper():<3s}: n={split['n_episodes']:<4d} "
                f"SR={split['pc_success']:5.1f}% "
                f"pick0={split['pc_pick_obj0']:5.1f}% "
                f"place0={split['pc_place_obj0_correct']:5.1f}% "
                f"pick1={split['pc_pick_obj1']:5.1f}% "
                f"place1={split['pc_place_obj1_correct']:5.1f}% "
                f"viol={split['pc_same_container_violation']:4.1f}%"
            )
    print()
    if info["per_task"]:
        print("Per-task breakdown:")
        for task_name, stats in sorted(info["per_task"].items()):
            print(
                f"  {task_name:<45s}  "
                f"SR={stats['pc_success']:5.1f}%  "
                f"pick0={stats['pc_pick_obj0']:5.1f}%  "
                f"place0={stats['pc_place_obj0_correct']:5.1f}%  "
                f"pick1={stats['pc_pick_obj1']:5.1f}%  "
                f"place1={stats['pc_place_obj1_correct']:5.1f}%  "
                f"viol={stats['pc_same_container_violation']:4.1f}%  "
                f"({stats['n_success']}/{stats['n_episodes']})"
            )
    print("=" * 60)
    print(f"Full results: {out_dir_path / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
