#!/usr/bin/env python
"""
Generate CG_L5 pick-place-then-button-press data using waypoint motion planning.

Each successful episode completes three supervised stages:
1. grasp object_A,
2. place object_A into object_B and lift away,
3. press the requested colored button.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
CG_ROOT = REPO_ROOT / "CG"
ROBOSUITE_ROOT = CG_ROOT / "robosuite"

sys.path.insert(0, str(CG_ROOT))
sys.path.insert(0, str(ROBOSUITE_ROOT))

from robosuite.controllers import load_composite_controller_config
from robosuite.utils.errors import RandomizationError
from robosuite.utils.placement_samplers import UniformApartRandomSampler

from generate_cg_l4_motion_planning import (
    CONTAINER_NAMES,
    OBJECT_NAMES,
    PickPlaceParams,
    SimplePickPlacePlanner,
    SimpleWaypoint,
    StageOffsetRule,
    WaypointSequencer,
    _make_video_frames,
    _setup_logger,
    _write_video,
    _write_video_array,
    quat_wxyz_to_xyzw,
    transform_to_relative_coordinates,
)
from test_motion_planner import ControllerParams as SimpleControllerParams
from test_motion_planner import SimplePoseController


BUTTON_COLORS = ["red", "green", "blue", "yellow"]
BUTTON_ENTITY_NAMES = [f"button_{color}" for color in BUTTON_COLORS]
POSED_ENTITY_NAMES = OBJECT_NAMES + CONTAINER_NAMES + BUTTON_ENTITY_NAMES
ALL_TASKS = [
    f"place the {obj} into the {cont} then press the {color} button"
    for obj in OBJECT_NAMES
    for cont in CONTAINER_NAMES
    for color in BUTTON_COLORS
]


def _parse_l5_task(task: str) -> tuple[int, int, int]:
    words = str(task).lower().split()
    obj_idx = next((i for i, name in enumerate(OBJECT_NAMES) if name in words), None)
    cont_idx = next((i for i, name in enumerate(CONTAINER_NAMES) if name in words), None)
    button_idx = next((i for i, color in enumerate(BUTTON_COLORS) if color in words), None)
    if obj_idx is None or cont_idx is None or button_idx is None:
        raise ValueError(f"Could not parse CG_L5 task: {task!r}")
    return int(obj_idx), int(cont_idx), int(button_idx)


def _task_dir_name(task: str) -> str:
    obj_idx, cont_idx, button_idx = _parse_l5_task(task)
    return f"{OBJECT_NAMES[obj_idx]}_into_{CONTAINER_NAMES[cont_idx]}_press_{BUTTON_COLORS[button_idx]}"


def _task_short(task: str) -> str:
    return _task_dir_name(task).replace("_into_", "_")


def _select_task_shard(tasks: list[str], *, num_shards: int, shard_index: int) -> list[str]:
    return [task for task_id, task in enumerate(tasks) if task_id % num_shards == shard_index]


def _count_saved_demos(task_dir: Path) -> int:
    if not task_dir.exists():
        return 0
    return sum(1 for path in task_dir.glob("demo_*/demo.hdf5") if path.is_file())


def _build_dataset_manifest(
    *,
    out_dir: Path,
    demos_dir: Path,
    tasks: list[str],
    info: dict,
    num_task_shards: int,
    task_shard_index: int,
) -> dict:
    task_entries = []
    for task in tasks:
        task_slug = _task_dir_name(task)
        task_dir = demos_dir / task_slug
        task_entries.append(
            {
                "task": task,
                "task_slug": task_slug,
                "task_indices": list(_parse_l5_task(task)),
                "demo_dir": str(task_dir.relative_to(out_dir)),
                "n_demos": _count_saved_demos(task_dir),
            }
        )
    return {
        "dataset": "CG_L5",
        "task_count": len(task_entries),
        "num_total_tasks": len(ALL_TASKS),
        "num_task_shards": int(num_task_shards),
        "task_shard_index": int(task_shard_index),
        "stage_success_order": ["grasp", "place", "button_press"],
        "hdf5_file": "demo.hdf5",
        "metadata": {
            "n_target_success_episodes": int(info.get("n_target_success_episodes", 0)),
            "n_attempted_episodes": int(info.get("n_attempted_episodes", 0)),
            "n_successful_episodes": int(info.get("n_successful_episodes", 0)),
            "n_saved_episodes": int(info.get("n_saved_episodes", 0)),
            "per_task_success": int(info.get("per_task_success", 0)),
        },
        "tasks": task_entries,
    }


def _write_dataset_manifests(
    *,
    out_dir: Path,
    demos_dir: Path,
    selected_tasks: list[str],
    info: dict,
    num_task_shards: int,
    task_shard_index: int,
    metadata_suffix: str,
) -> None:
    manifest = _build_dataset_manifest(
        out_dir=out_dir,
        demos_dir=demos_dir,
        tasks=selected_tasks,
        info=info,
        num_task_shards=num_task_shards,
        task_shard_index=task_shard_index,
    )

    if num_task_shards > 1:
        shard_path = out_dir / f"dataset_manifest{metadata_suffix}.json"
        shard_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return

    lock_path = out_dir / ".dataset_manifest.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        shard_manifests = []
        for path in sorted(out_dir.glob("dataset_manifest_shard_*_of_*.json")):
            with path.open("r", encoding="utf-8") as f:
                shard_manifests.append(json.load(f))

        tasks_by_slug = {}
        n_attempted = 0
        n_successful = 0
        n_saved = 0
        n_target = 0
        for shard_manifest in shard_manifests:
            metadata = shard_manifest.get("metadata", {})
            n_attempted += int(metadata.get("n_attempted_episodes", 0))
            n_successful += int(metadata.get("n_successful_episodes", 0))
            n_saved += int(metadata.get("n_saved_episodes", 0))
            n_target += int(metadata.get("n_target_success_episodes", 0))
            for task_entry in shard_manifest.get("tasks", []):
                if isinstance(task_entry, dict) and "task_slug" in task_entry:
                    tasks_by_slug[str(task_entry["task_slug"])] = task_entry

        combined = {
            "dataset": "CG_L5",
            "task_count": len(tasks_by_slug),
            "num_total_tasks": len(ALL_TASKS),
            "num_task_shards": int(num_task_shards),
            "completed_shards": len(shard_manifests),
            "stage_success_order": ["grasp", "place", "button_press"],
            "hdf5_file": "demo.hdf5",
            "metadata": {
                "n_target_success_episodes": n_target,
                "n_attempted_episodes": n_attempted,
                "n_successful_episodes": n_successful,
                "n_saved_episodes": n_saved,
            },
            "tasks": [tasks_by_slug[k] for k in sorted(tasks_by_slug)],
        }
        tmp_path = out_dir / "dataset_manifest.json.tmp"
        tmp_path.write_text(json.dumps(combined, indent=2) + "\n")
        tmp_path.replace(out_dir / "dataset_manifest.json")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def fix_env_task_pointers(env) -> None:
    """Update live object and button references after task / index changes."""
    if hasattr(env, "_refresh_task_pointers"):
        env._refresh_task_pointers()
        return
    env.object_A = env.object_A_list[int(env.object_A_index)]
    env.object_B = env.object_B_list[int(env.object_B_index)]
    env.target_button = env.button_list[int(env.target_button_index)]
    env.object_A_body_id = env.sim.model.body_name2id(env.object_A.root_body)
    env.object_B_body_id = env.sim.model.body_name2id(env.object_B.root_body)
    env.button_body_ids = [env.sim.model.body_name2id(button.root_body) for button in env.button_list]
    env.target_button_body_id = env.button_body_ids[int(env.target_button_index)]


def load_cg_l5_env_class():
    module = importlib.import_module("robosuite.environments.manipulation.CG_L5")
    env_cls = getattr(module, "CG_L5", None)
    if env_cls is None:
        raise AttributeError("Could not find CG_L5 class in CG_L5.py")
    return env_cls


def make_env(*, horizon: int):
    env_cls = load_cg_l5_env_class()
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    bootstrap_sampler_a = UniformApartRandomSampler(
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
    bootstrap_sampler_b = UniformApartRandomSampler(
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
    return env_cls(
        robots="Panda",
        controller_configs=controller_config,
        gripper_types="PandaGripper",
        task=ALL_TASKS[0],
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


class L5CGWrapper:
    """Minimal vectorized wrapper around CG_L5 for planner-based generation."""

    MIN_OBJECT_BUTTON_DISTANCE = 0.065

    def __init__(self, make_env_fn, num_envs: int = 1, use_relative_coordinates: bool = True):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = int(num_envs)
        self.use_relative_coordinates = bool(use_relative_coordinates)
        self.obs: list[dict | None] = [None] * self.num_envs
        self.tasks: list[str] = [""] * self.num_envs
        self.obj_indices: list[int] = [0] * self.num_envs
        self.cont_indices: list[int] = [0] * self.num_envs
        self.button_indices: list[int] = [0] * self.num_envs
        self.is_success: list[bool] = [False] * self.num_envs

    def _has_object_button_overlap(self, env) -> bool:
        button_xy = np.array([env.sim.data.body_xpos[bid][:2] for bid in env.button_body_ids], dtype=np.float32)
        for obj in list(env.object_A_list):
            obj_body_id = env.sim.model.body_name2id(obj.root_body)
            obj_xy = env.sim.data.body_xpos[obj_body_id][:2].astype(np.float32)
            if float(np.min(np.linalg.norm(button_xy - obj_xy, axis=1))) < self.MIN_OBJECT_BUTTON_DISTANCE:
                return True
        return False

    def _safe_reset_env(self, env, max_attempts: int = 50, **kwargs):
        last_error = None
        for _ in range(max_attempts):
            try:
                obs = env.reset(**kwargs)
                fix_env_task_pointers(env)
                if self._has_object_button_overlap(env):
                    last_error = RuntimeError("sampled object too close to fixed button cluster")
                    continue
                return env._get_observations()
            except RandomizationError as exc:
                last_error = exc
        raise last_error if last_error is not None else RuntimeError("env.reset() failed without exception")

    def reset(self, *, tasks: list[str] | None = None, **kwargs):
        kwargs.pop("seed", None)
        if tasks is not None and len(tasks) != self.num_envs:
            raise ValueError(f"tasks must have length {self.num_envs}, got {len(tasks)}")
        self.obs = []
        for i, env in enumerate(self.envs):
            task = str(tasks[i]) if tasks is not None else str(np.random.choice(ALL_TASKS))
            obj_idx, cont_idx, button_idx = _parse_l5_task(task)
            self.tasks[i] = task
            self.obj_indices[i] = obj_idx
            self.cont_indices[i] = cont_idx
            self.button_indices[i] = button_idx
            env.strategy = "fixed"
            env.task = task
            obs_i = self._safe_reset_env(env, **kwargs)
            self.obs.append(obs_i)
        self.is_success = [False] * self.num_envs
        obs_dicts = [self._compute_observation(i) for i in range(self.num_envs)]
        batched = {k: np.stack([obs[k] for obs in obs_dicts], axis=0) for k in obs_dicts[0]}
        return batched, [{} for _ in range(self.num_envs)]

    def step(self, actions: np.ndarray):
        next_obs, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs_i, reward, terminated, info = env.step(actions[i].copy())
            self.obs[i] = obs_i
            cur_success = bool(env._check_success())
            self.is_success[i] = bool(self.is_success[i] or cur_success)
            info["is_success"] = bool(self.is_success[i])
            next_obs.append(self._compute_observation(i))
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(False)
            infos.append(info)
        batched = {k: np.stack([obs[k] for obs in next_obs], axis=0) for k in next_obs[0]}
        final_info = [infos[i] if terminateds[i] or truncateds[i] else None for i in range(self.num_envs)]
        return (
            batched,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds),
            np.array(truncateds),
            {"is_success": list(self.is_success), "final_info": final_info},
        )

    def _entity_pose_world(self, env, raw: dict, name: str) -> np.ndarray:
        if name in BUTTON_ENTITY_NAMES:
            idx = BUTTON_ENTITY_NAMES.index(name)
            body_id = env.button_body_ids[idx]
            pos = env.sim.data.body_xpos[body_id]
            quat = quat_wxyz_to_xyzw(env.sim.data.body_xquat[body_id])
        else:
            pos = raw[f"{name}_pos"]
            quat = raw[f"{name}_quat"]
        return np.concatenate([pos, quat]).astype(np.float32)

    def _compute_observation(self, idx: int) -> dict:
        raw = self.obs[idx]
        assert raw is not None
        env = self.envs[idx]
        eef_pos = raw["robot0_eef_pos"].astype(np.float32)
        eef_quat = raw["robot0_eef_quat"].astype(np.float32)
        gripper_q = raw["robot0_gripper_qpos"].astype(np.float32)

        object_poses_world = []
        object_poses_rel = []
        for name in POSED_ENTITY_NAMES:
            pose = self._entity_pose_world(env, raw, name)
            object_poses_world.append(pose)
            if self.use_relative_coordinates:
                object_poses_rel.append(transform_to_relative_coordinates(eef_pos, eef_quat, pose))
            else:
                object_poses_rel.append(pose)

        obj_pos = env.sim.data.body_xpos[env.object_A_body_id].astype(np.float32)
        button_pos = env.sim.data.body_xpos[env.target_button_body_id].astype(np.float32)
        environment_state = np.concatenate(
            object_poses_rel + [(obj_pos - eef_pos).astype(np.float32), (button_pos - eef_pos).astype(np.float32)]
        ).astype(np.float32)

        out = {
            "observation.state": np.concatenate([eef_pos, eef_quat, gripper_q]).astype(np.float32),
            "observation.environment_state": environment_state,
            "observation.object_world": np.concatenate(object_poses_world).astype(np.float32),
        }
        if "agentview_image" in raw:
            out["observation.images.agentview"] = np.flipud(raw["agentview_image"].astype(np.uint8))
        if "robot0_eye_in_hand_image" in raw:
            out["observation.images.robot0_eye_in_hand"] = np.flipud(raw["robot0_eye_in_hand_image"].astype(np.uint8))
        return out

    def close(self):
        for env in self.envs:
            env.close()


class ButtonPressPlanner:
    """Position-only waypoint planner for pressing the target colored button."""

    GRIP_CLOSE = 1.0
    GRIP_OPEN = -1.0

    def __init__(self, *, pos_tol: float = 0.006, hold_steps: int = 28):
        self.seq = WaypointSequencer(pos_tol=pos_tol)
        self.ctrl = SimplePoseController(
            SimpleControllerParams(
                max_step_size=0.025,
                pos_tolerance=float(pos_tol),
                osc_pos_limit=0.05,
                use_orientation=False,
                ignore_position=False,
                max_acceleration=0.08,
                use_smooth_stop=True,
            )
        )
        self.hold_steps = int(hold_steps)
        self.press_tip_xy_offset = np.array([-0.020, 0.0, 0.0], dtype=np.float32)
        self.retry_count = 0
        self.max_retry_count = 4
        self.success_retreat_started = False
        self.done = False
        self.phase = "button_init"

    def reset(self) -> None:
        self.seq.reset([])
        self.ctrl.reset()
        self.retry_count = 0
        self.success_retreat_started = False
        self.done = False
        self.phase = "button_init"

    def _build(self, env, *, retry: bool = False) -> list[SimpleWaypoint]:
        button_pos = env.sim.data.body_xpos[env.target_button_body_id].astype(np.float32)
        press_center = button_pos + self.press_tip_xy_offset
        button_top_z = float(button_pos[2] + env.button_cap_top_offset)
        above = press_center.copy()
        above[2] = button_top_z + 0.12
        press = press_center.copy()
        extra_depth = 0.004 * self.retry_count if retry else 0.0
        press[2] = button_top_z - min(float(env.button_press_depth) + extra_depth, 0.040)
        retreat = above.copy()
        return [
            # Close above the button so the narrower gripper tip, not open fingers, presses the cap.
            SimpleWaypoint(above, None, self.GRIP_CLOSE, self.GRIP_CLOSE, "button_prepress", repeat=4),
            SimpleWaypoint(press, None, self.GRIP_CLOSE, self.GRIP_CLOSE, "button_press", repeat=max(1, self.hold_steps)),
            SimpleWaypoint(retreat, None, self.GRIP_CLOSE, self.GRIP_CLOSE, "button_retreat"),
        ]

    def _build_success_retreat(self, env) -> list[SimpleWaypoint]:
        button_pos = env.sim.data.body_xpos[env.target_button_body_id].astype(np.float32)
        retreat = button_pos + self.press_tip_xy_offset
        retreat[2] = float(button_pos[2] + env.button_cap_top_offset + 0.12)
        return [SimpleWaypoint(retreat, None, self.GRIP_CLOSE, self.GRIP_CLOSE, "button_retreat")]

    def get_action(self, env, *, eef_quat_xyzw: np.ndarray) -> np.ndarray:
        del eef_quat_xyzw
        if self.done:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        if self.seq.is_empty():
            self.seq.reset(self._build(env))
        if env.button_press_success() and not self.success_retreat_started:
            self.success_retreat_started = True
            self.seq.reset(self._build_success_retreat(env))
        cur = self.seq.current()
        if cur is None:
            if env.button_press_success() or self.success_retreat_started or self.retry_count >= self.max_retry_count:
                self.done = True
                self.phase = "done"
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            self.retry_count += 1
            self.seq.reset(self._build(env, retry=True))
            cur = self.seq.current()
            if cur is None:
                self.done = True
                self.phase = "done"
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        self.phase = cur.name
        act, pos_dist, _rot_dist, _err = self.ctrl.compute_action(env, target_pos=cur.pos, target_quat_xyzw=None)
        act = act.astype(np.float32)
        act[6] = float(cur.gripper_hold if (self.seq.is_holding() or (pos_dist <= self.seq.pos_tol and cur.repeat > 1)) else cur.gripper_move)
        self.seq.advance(reached=bool(float(pos_dist) <= float(self.seq.pos_tol)))
        if self.seq.done:
            self.done = bool(env.button_press_success() or self.success_retreat_started or self.retry_count >= self.max_retry_count)
            if self.done:
                self.phase = "done"
        return act


class L5PickPlaceButtonPlanner:
    """Composite planner: pick-place first, then press the target button."""

    def __init__(self, *, pick_params: PickPlaceParams | None = None):
        self.pick_place = SimplePickPlacePlanner(
            waypoint_params=pick_params,
            max_step_size=0.05,
            pos_tol=0.005,
            osc_pos_limit=0.05,
            grasp_hold_steps=25,
            release_hold_steps=25,
            max_acceleration=0.15,
            use_smooth_stop=True,
        )
        self.button_press = ButtonPressPlanner()
        self.done = False
        self.phase = "init"
        self.stage_success = np.zeros(3, dtype=np.bool_)

    def reset(self) -> None:
        self.pick_place.reset()
        self.button_press.reset()
        self.done = False
        self.phase = "pick_place"
        self.stage_success[:] = False

    def post_step(self, env) -> None:
        self.stage_success[0] = bool(
            self.stage_success[0] or env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.object_A)
        )
        self.stage_success[1] = bool(self.stage_success[1] or getattr(env, "has_completed_place", False))
        self.stage_success[2] = bool(self.stage_success[2] or env.button_press_success())

    def get_action(self, env, *, eef_quat_xyzw: np.ndarray) -> np.ndarray:
        if self.done:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        if not bool(self.stage_success[1]):
            self.phase = f"pick_{self.pick_place.phase}"
            return self.pick_place.get_action(env, eef_quat_xyzw=eef_quat_xyzw)
        action = self.button_press.get_action(env, eef_quat_xyzw=eef_quat_xyzw)
        self.phase = self.button_press.phase
        if self.button_press.done and bool(self.stage_success[2]):
            self.done = True
        return action

    def subtask_id(self) -> int:
        if bool(self.stage_success[1]):
            return 2
        if bool(self.stage_success[0]):
            return 1
        return 0


@dataclass
class GenStats:
    attempted: int = 0
    successful: int = 0
    saved: int = 0
    videos: int = 0
    task_attempts: Counter[str] = None  # type: ignore[assignment]
    task_successes: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.task_attempts = Counter()
        self.task_successes = Counter()


def save_demo_triplet(demo_dir: Path, ep: dict, *, fps: int = 20) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(demo_dir / "demo.hdf5", "w") as f:
        obs_grp = f.require_group("obs")
        obs_grp.create_dataset("robot0_eef_pos", data=ep["eef_pos"], compression="gzip")
        obs_grp.create_dataset("robot0_eef_quat", data=ep["eef_quat"], compression="gzip")
        obs_grp.create_dataset("robot0_gripper_qpos", data=ep["gripper_q"], compression="gzip")
        obs_grp.create_dataset("object", data=ep["obj_world"], compression="gzip")
        obs_grp.create_dataset("environment_state", data=ep["env_state"], compression="gzip")
        obs_grp.create_dataset("task_indices", data=np.asarray(ep["task_indices"], dtype=np.int64), compression="gzip")
        obs_grp.create_dataset("subtask_id", data=np.asarray(ep["subtask_id"], dtype=np.int64), compression="gzip")
        obs_grp.create_dataset("stage_success", data=np.asarray(ep["stage_success"], dtype=np.bool_), compression="gzip")
        obs_grp.create_dataset("grasp_success", data=np.asarray(ep["grasp_success"], dtype=np.bool_), compression="gzip")
        obs_grp.create_dataset("place_success", data=np.asarray(ep["place_success"], dtype=np.bool_), compression="gzip")
        obs_grp.create_dataset("button_press_success", data=np.asarray(ep["button_press_success"], dtype=np.bool_), compression="gzip")
        f.create_dataset("actions", data=ep["actions"], compression="gzip")
        f.attrs["task"] = str(ep["task"])
        f.attrs["planner"] = str(ep.get("planner", "l5_pick_place_button"))
        f.attrs["success"] = bool(ep.get("success", False))
        f.attrs["stage_success_order"] = "grasp,place,button_press"
    if "agentview" not in ep or "eye_in_hand" not in ep:
        raise KeyError("episode missing required image streams: agentview and/or eye_in_hand")
    _write_video_array(ep["agentview"], demo_dir / "agentview.mp4", fps=fps)
    _write_video_array(ep["eye_in_hand"], demo_dir / "robot0_eye_in_hand.mp4", fps=fps)


def run_batch(
    env: L5CGWrapper,
    *,
    horizon: int,
    planner_params: PickPlaceParams | None = None,
    tasks: list[str] | None = None,
    fps: int = 20,
    post_success_hold_s: float = 0.5,
) -> tuple[list[dict | None], dict]:
    obs, _ = env.reset(tasks=tasks)
    planners = [L5PickPlaceButtonPlanner(pick_params=planner_params) for _ in range(env.num_envs)]
    for planner in planners:
        planner.reset()

    done = [False] * env.num_envs
    ep_success = [False] * env.num_envs
    hold_remaining = [0] * env.num_envs
    hold_steps = max(0, int(round(float(post_success_hold_s) * float(max(int(fps), 1)))))

    buf_eef_pos = [[] for _ in range(env.num_envs)]
    buf_eef_quat = [[] for _ in range(env.num_envs)]
    buf_gripper_q = [[] for _ in range(env.num_envs)]
    buf_obj_world = [[] for _ in range(env.num_envs)]
    buf_env_state = [[] for _ in range(env.num_envs)]
    buf_agentview = [[] for _ in range(env.num_envs)]
    buf_eye = [[] for _ in range(env.num_envs)]
    buf_actions = [[] for _ in range(env.num_envs)]
    buf_task_indices = [[] for _ in range(env.num_envs)]
    buf_subtask_id = [[] for _ in range(env.num_envs)]
    buf_stage_success = [[] for _ in range(env.num_envs)]

    steps_taken = 0
    for step in range(horizon):
        if all(done):
            break
        steps_taken = step + 1

        for i in range(env.num_envs):
            if done[i]:
                continue
            st = obs["observation.state"][i]
            buf_eef_pos[i].append(st[:3].copy())
            buf_eef_quat[i].append(st[3:7].copy())
            buf_gripper_q[i].append(st[7:9].copy())
            buf_obj_world[i].append(obs["observation.object_world"][i].copy())
            buf_env_state[i].append(obs["observation.environment_state"][i].copy())
            if "observation.images.agentview" in obs:
                buf_agentview[i].append(obs["observation.images.agentview"][i].copy())
            if "observation.images.robot0_eye_in_hand" in obs:
                buf_eye[i].append(obs["observation.images.robot0_eye_in_hand"][i].copy())

        action_mat = np.zeros((env.num_envs, 7), dtype=np.float32)
        for i in range(env.num_envs):
            if done[i]:
                continue
            if bool(ep_success[i]) and bool(planners[i].done):
                if hold_remaining[i] <= 0:
                    hold_remaining[i] = int(hold_steps)
                action_mat[i] = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
                hold_remaining[i] -= 1
                if hold_remaining[i] <= 0:
                    done[i] = True
                continue
            eef_quat_xyzw = obs["observation.state"][i][3:7].astype(np.float32)
            action_mat[i] = planners[i].get_action(env.envs[i], eef_quat_xyzw=eef_quat_xyzw)
            buf_actions[i].append(action_mat[i].copy())
            buf_task_indices[i].append(
                np.array([env.obj_indices[i], env.cont_indices[i], env.button_indices[i]], dtype=np.int64)
            )
            buf_subtask_id[i].append(np.array([planners[i].subtask_id()], dtype=np.int64))

        obs, _reward, terminated, truncated, info = env.step(action_mat)

        for i in range(env.num_envs):
            if done[i]:
                continue
            planners[i].post_step(env.envs[i])
            buf_stage_success[i].append(planners[i].stage_success.astype(np.bool_).copy())
            if bool(info["is_success"][i]) and bool(planners[i].done):
                ep_success[i] = True
            elif bool(terminated[i]) or bool(truncated[i]):
                done[i] = True

    results: list[dict | None] = []
    for i in range(env.num_envs):
        if not buf_actions[i]:
            results.append(None)
            continue
        T = min(len(buf_actions[i]), len(buf_stage_success[i]))
        if T <= 0:
            results.append(None)
            continue
        stage_success = np.stack(buf_stage_success[i][:T]).astype(np.bool_)
        ep = {
            "eef_pos": np.stack(buf_eef_pos[i][:T]),
            "eef_quat": np.stack(buf_eef_quat[i][:T]),
            "gripper_q": np.stack(buf_gripper_q[i][:T]),
            "obj_world": np.stack(buf_obj_world[i][:T]),
            "env_state": np.stack(buf_env_state[i][:T]),
            "actions": np.stack(buf_actions[i][:T]),
            "task_indices": np.stack(buf_task_indices[i][:T]).astype(np.int64, copy=False),
            "subtask_id": np.stack(buf_subtask_id[i][:T]).astype(np.int64, copy=False),
            "stage_success": stage_success,
            "grasp_success": stage_success[:, 0:1],
            "place_success": stage_success[:, 1:2],
            "button_press_success": stage_success[:, 2:3],
            "task": env.tasks[i],
            "success": bool(ep_success[i]),
            "planner": "l5_pick_place_button",
        }
        if buf_agentview[i]:
            ep["agentview"] = np.stack(buf_agentview[i][:T])
        if buf_eye[i]:
            ep["eye_in_hand"] = np.stack(buf_eye[i][:T])
        results.append(ep)

    return results, {
        "steps_taken": steps_taken,
        "batch_successes": int(sum(ep_success)),
        "num_envs": int(env.num_envs),
    }


def generate_dataset(
    env: L5CGWrapper,
    *,
    n_success_episodes: int,
    per_task_success: int,
    tasks_to_run: list[str] | None,
    horizon: int,
    demos_dir: Path,
    planner_params: PickPlaceParams,
    videos_dir: Path | None,
    max_videos: int,
    fps: int,
    logger=None,
) -> dict:
    demos_dir.mkdir(parents=True, exist_ok=True)
    if videos_dir is not None and max_videos > 0:
        videos_dir.mkdir(parents=True, exist_ok=True)

    stats = GenStats()
    batch = 0
    start = time.time()
    log = logger.info if logger is not None else print
    tasks_list = list(tasks_to_run) if tasks_to_run is not None else list(ALL_TASKS)
    target_total = int(per_task_success) * len(tasks_list) if int(per_task_success) > 0 else int(n_success_episodes)
    pbar = tqdm(total=target_total, desc="saved", unit="demo", dynamic_ncols=True) if tqdm is not None else None

    def _maybe_save_review_video(ep: dict, *, attempt_id: int) -> None:
        if videos_dir is None or stats.videos >= max_videos:
            return
        if "agentview" not in ep or "eye_in_hand" not in ep:
            return
        name = f"ep{attempt_id:05d}_{_task_short(ep['task'])}_{'success' if ep.get('success') else 'fail'}.mp4"
        try:
            _write_video(_make_video_frames(ep["agentview"], ep["eye_in_hand"]), videos_dir / name, fps=fps)
            stats.videos += 1
        except Exception as exc:
            if logger is not None:
                logger.warning(f"review video failed ({name}): {exc}")

    def _handle_episode(ep: dict, *, demo_dir: Path, attempt_id: int) -> bool:
        stats.task_attempts[str(ep["task"])] += 1
        if bool(ep.get("success", False)):
            stats.successful += 1
            stats.task_successes[str(ep["task"])] += 1
            save_demo_triplet(demo_dir, ep, fps=fps)
            stats.saved += 1
            if pbar is not None:
                pbar.update(1)
            saved = True
        else:
            saved = False
        _maybe_save_review_video(ep, attempt_id=attempt_id)
        return saved

    if int(per_task_success) > 0:
        for task in tasks_list:
            saved_for_task = 0
            task_dir = demos_dir / _task_dir_name(task)
            task_dir.mkdir(parents=True, exist_ok=True)
            log(f"Task: {task} | target={int(per_task_success)} demos")
            while saved_for_task < int(per_task_success):
                batch += 1
                episodes, batch_info = run_batch(
                    env,
                    horizon=horizon,
                    planner_params=planner_params,
                    tasks=[task] * int(env.num_envs),
                    fps=fps,
                )
                stats.attempted += int(batch_info["num_envs"])
                base_attempt_id = stats.attempted - int(batch_info["num_envs"])
                batch_successes = int(batch_info["batch_successes"])
                for i, ep in enumerate(episodes):
                    if ep is None or saved_for_task >= int(per_task_success):
                        continue
                    attempt_id = base_attempt_id + i + 1
                    if bool(ep.get("success", False)):
                        if _handle_episode(ep, demo_dir=task_dir / f"demo_{saved_for_task:06d}", attempt_id=attempt_id):
                            saved_for_task += 1
                    else:
                        stats.task_attempts[str(task)] += 1
                        _maybe_save_review_video(ep, attempt_id=attempt_id)
                batch_sr = (batch_successes / max(int(batch_info["num_envs"]), 1)) * 100.0
                overall_sr = (stats.successful / max(stats.attempted, 1)) * 100.0
                saved_rate = (stats.saved / max(stats.attempted, 1)) * 100.0
                log(
                    f"[batch {batch}] steps={batch_info['steps_taken']}/{horizon} | "
                    f"batch_SR={batch_sr:.0f}% ({batch_successes}/{int(batch_info['num_envs'])}) | "
                    f"overall_SR={overall_sr:.1f}% | saved_rate={saved_rate:.1f}% | "
                    f"saved_task={saved_for_task}/{int(per_task_success)} | saved_all={stats.saved}/{target_total}"
                )
    else:
        while stats.saved < int(n_success_episodes):
            batch += 1
            if tasks_to_run is None:
                tasks = None
            elif len(tasks_list) == 1:
                tasks = [tasks_list[0]] * int(env.num_envs)
            else:
                tasks = [str(np.random.choice(tasks_list)) for _ in range(int(env.num_envs))]
            episodes, batch_info = run_batch(
                env,
                horizon=horizon,
                planner_params=planner_params,
                tasks=tasks,
                fps=fps,
            )
            stats.attempted += int(batch_info["num_envs"])
            base_attempt_id = stats.attempted - int(batch_info["num_envs"])
            batch_successes = int(batch_info["batch_successes"])
            for i, ep in enumerate(episodes):
                if ep is None:
                    continue
                attempt_id = base_attempt_id + i + 1
                demo_dir = demos_dir / f"demo_{stats.saved:06d}"
                if stats.saved < int(n_success_episodes):
                    _handle_episode(ep, demo_dir=demo_dir, attempt_id=attempt_id)
            batch_sr = (batch_successes / max(int(batch_info["num_envs"]), 1)) * 100.0
            overall_sr = (stats.successful / max(stats.attempted, 1)) * 100.0
            saved_rate = (stats.saved / max(stats.attempted, 1)) * 100.0
            log(
                f"[batch {batch}] steps={batch_info['steps_taken']}/{horizon} | "
                f"batch_SR={batch_sr:.0f}% ({batch_successes}/{int(batch_info['num_envs'])}) | "
                f"overall_SR={overall_sr:.1f}% | saved_rate={saved_rate:.1f}% | "
                f"saved={stats.saved}/{int(n_success_episodes)}"
            )

    if pbar is not None:
        pbar.close()
    elapsed = time.time() - start
    return {
        "n_target_success_episodes": int(target_total),
        "n_attempted_episodes": int(stats.attempted),
        "n_successful_episodes": int(stats.successful),
        "n_saved_episodes": int(stats.saved),
        "rollout_success_rate": stats.successful / max(stats.attempted, 1) * 100.0,
        "saved_success_rate": stats.saved / max(stats.attempted, 1) * 100.0,
        "elapsed_s": elapsed,
        "demos_dir": str(demos_dir),
        "n_videos_saved": int(stats.videos),
        "per_task_success": int(per_task_success),
        "per_task": {
            task: {
                "attempted": int(stats.task_attempts[task]),
                "successful": int(stats.task_successes[task]),
                "success_rate": (stats.task_successes[task] / max(stats.task_attempts[task], 1)) * 100.0,
            }
            for task in sorted(stats.task_attempts.keys())
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate CG_L5 demos with waypoint pick-place and button press planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-success", type=int, default=200, help="Successful episodes to collect")
    parser.add_argument("--per-task-success", type=int, default=0, help="If >0, save this many successful demos per task")
    parser.add_argument("--task", type=str, default="", help='Optional exact task, e.g. "place the cube into the bin then press the red button"')
    parser.add_argument("--num-task-shards", "--num-shards", dest="num_task_shards", type=int, default=1, help="Split the 64 tasks into this many shards")
    parser.add_argument("--task-shard-index", "--shard-index", dest="task_shard_index", type=int, default=0, help="Task shard index to run, in [0, num_task_shards)")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments")
    parser.add_argument("--horizon", type=int, default=320, help="Max steps per episode")
    parser.add_argument("--out-dir", default="results/cg_l5_motion_gen")
    parser.add_argument("--run-id", default="", help="Optional run id. Shards with the same run id write to the same gen_<run-id> folder")
    parser.add_argument("--max-videos", type=int, default=20, help="Debug videos to save (0 = none)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--use-orientation-control", action="store_true", help="Enable grasp yaw/orientation control")
    parser.add_argument("--no-rotate-for-grasp", action="store_true", help="Disable rotate_for_grasp yaw alignment")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(args.seed)

    num_task_shards = int(args.num_task_shards)
    task_shard_index = int(args.task_shard_index)
    if num_task_shards < 1:
        raise ValueError(f"--num-task-shards must be >= 1, got {num_task_shards}")
    if not 0 <= task_shard_index < num_task_shards:
        raise ValueError(
            f"--task-shard-index must be in [0, {num_task_shards}), got {task_shard_index}"
        )

    task_arg = str(args.task).strip().lower()
    tasks_to_run = None
    if task_arg:
        if num_task_shards > 1:
            raise ValueError("Use either --task or task sharding, not both")
        if task_arg not in [t.lower() for t in ALL_TASKS]:
            raise ValueError(f"--task not recognized: {args.task!r}\nExpected one of: {ALL_TASKS}")
        tasks_to_run = [next(t for t in ALL_TASKS if t.lower() == task_arg)]
    elif num_task_shards > 1:
        tasks_to_run = _select_task_shard(
            list(ALL_TASKS),
            num_shards=num_task_shards,
            shard_index=task_shard_index,
        )

    env = L5CGWrapper(
        make_env_fn=lambda: make_env(horizon=args.horizon),
        num_envs=args.num_envs,
        use_relative_coordinates=True,
    )

    run_id = str(args.run_id).strip() or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"gen_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    demos_dir = out_dir / ("task" if int(args.per_task_success) > 0 else "demos")
    videos_dir = (out_dir / "videos") if args.max_videos > 0 else None
    logger = _setup_logger(out_dir, args.log_level)
    metadata_suffix = (
        f"_shard_{task_shard_index:02d}_of_{num_task_shards:02d}"
        if num_task_shards > 1
        else ""
    )
    (out_dir / f"args{metadata_suffix}.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    info = generate_dataset(
        env,
        n_success_episodes=args.n_success,
        per_task_success=int(args.per_task_success),
        tasks_to_run=tasks_to_run,
        horizon=args.horizon,
        demos_dir=demos_dir,
        planner_params=PickPlaceParams(
            use_orientation_control=bool(args.use_orientation_control),
            rotate_for_grasp_enable=not bool(args.no_rotate_for_grasp),
            stage_offset_rules=[
                StageOffsetRule(stage="place", container="mug", delta_xyz=(0.00, 0.00, 0.05)),
                StageOffsetRule(stage="place", container="mug_no_handle", delta_xyz=(0.00, 0.00, 0.05)),
                StageOffsetRule(stage="place", container="bin", delta_xyz=(0.00, 0.00, 0.02)),
            ],
        ),
        videos_dir=videos_dir,
        max_videos=int(args.max_videos),
        fps=int(args.fps),
        logger=logger,
    )
    info["num_total_tasks"] = len(ALL_TASKS)
    info["num_task_shards"] = num_task_shards
    info["task_shard_index"] = task_shard_index
    info["selected_tasks"] = tasks_to_run if tasks_to_run is not None else list(ALL_TASKS)
    selected_tasks = list(tasks_to_run) if tasks_to_run is not None else list(ALL_TASKS)

    env.close()
    logger.info("=" * 60)
    logger.info("CG_L5 generation complete")
    for key in [
        "n_target_success_episodes",
        "n_attempted_episodes",
        "n_successful_episodes",
        "n_saved_episodes",
        "rollout_success_rate",
        "saved_success_rate",
        "elapsed_s",
        "demos_dir",
        "n_videos_saved",
    ]:
        logger.info(f"{key}: {info[key]}")
    logger.info("=" * 60)
    (out_dir / f"gen_info{metadata_suffix}.json").write_text(json.dumps(info, indent=2) + "\n")
    _write_dataset_manifests(
        out_dir=out_dir,
        demos_dir=demos_dir,
        selected_tasks=selected_tasks,
        info=info,
        num_task_shards=num_task_shards,
        task_shard_index=task_shard_index,
        metadata_suffix=metadata_suffix,
    )


if __name__ == "__main__":
    main()
