#!/usr/bin/env python
"""
Generate one-stage CG_L4 pick-and-place data using a waypoint motion planner.

This script does not load a learned policy. Instead, it:
1. samples a fixed one-stage CG_L4 task,
2. runs a scripted motion planner with OSC delta-position actions,
3. saves successful episodes to HDF5,
4. optionally writes debug videos.

Output HDF5 layout:
  demo_{N}/
    obs/
      robot0_eef_pos           (T, 3)
      robot0_eef_quat          (T, 4)
      robot0_gripper_qpos      (T, 2)
      object                   (T, 70)  world-frame poses for 10 objects / containers
      environment_state        (T, 73)  optional EEF-relative object state + gripper_to_target
      agentview_image          (T, H, W, 3) uint8
      robot0_eye_in_hand_image (T, H, W, 3) uint8
    actions                    (T, 7)
    attrs:
      task
      planner = "waypoint_pick_place"
      success = True

Usage:
  python generate_cg_l4_motion_planning.py \
      --n-success 200 \
      --num-envs 4 \
      --horizon 220 \
      --out-dir results/cg_l4_motion_gen
"""

from __future__ import annotations

import argparse
import logging
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CG_ROOT = REPO_ROOT / "CG"
ROBOSUITE_ROOT = CG_ROOT / "robosuite"

sys.path.insert(0, str(CG_ROOT))
sys.path.insert(0, str(ROBOSUITE_ROOT))

from robosuite.controllers import load_composite_controller_config
from robosuite.utils.errors import RandomizationError
from robosuite.utils.placement_samplers import UniformApartRandomSampler

# Reuse the simple controller implementation from our test script.
# This keeps "controller math" in one place and lets generation stay minimal.
from test_motion_planner import ControllerParams as SimpleControllerParams
from test_motion_planner import SimplePoseController as SimplePoseController


OBJECT_NAMES = ["cross", "cube", "cylinder", "milk"]
CONTAINER_NAMES = ["bin", "mug", "plate", "mug_no_handle"]
ALL_TASKS = [f"place the {obj} into the {cont}" for obj in OBJECT_NAMES for cont in CONTAINER_NAMES]
POSED_ENTITY_NAMES = OBJECT_NAMES + CONTAINER_NAMES


def _setup_logger(out_dir: Path, level: str) -> logging.Logger:
    """
    Clean, single logger that writes to both stdout and a file.
    """
    logger = logging.getLogger("cg_l4_motion_gen")
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if re-imported / re-run in an interactive session.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname).1s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logger.level)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(out_dir / "gen.log")
    fh.setLevel(logger.level)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=-1)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    q_inv = q.copy()
    q_inv[..., :3] *= -1.0
    return q_inv


def canonicalize_quaternion(q: np.ndarray) -> np.ndarray:
    q_canon = q.copy()
    if q_canon[..., 3] < 0:
        q_canon = -q_canon
    return q_canon


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def quat_to_euler_xyz(q_xyzw: np.ndarray) -> tuple[float, float, float]:
    """
    Returns roll, pitch, yaw (XYZ intrinsic) from a quaternion in xyzw.
    """
    x, y, z, w = [float(v) for v in q_xyzw]
    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(t0, t1))
    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = float(np.clip(t2, -1.0, 1.0))
    pitch = float(np.arcsin(t2))
    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(t3, t4))
    return roll, pitch, yaw


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Returns quaternion (xyzw) from roll, pitch, yaw (XYZ intrinsic).
    """
    cr = float(np.cos(roll * 0.5))
    sr = float(np.sin(roll * 0.5))
    cp = float(np.cos(pitch * 0.5))
    sp = float(np.sin(pitch * 0.5))
    cy = float(np.cos(yaw * 0.5))
    sy = float(np.sin(yaw * 0.5))

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return canonicalize_quaternion(np.array([x, y, z, w], dtype=np.float32))


def quat_to_rotvec(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Quaternion (xyzw) -> rotation vector (axis * angle), radians.
    """
    q = canonicalize_quaternion(q_xyzw.astype(np.float32))
    x, y, z, w = [float(v) for v in q]
    v = np.array([x, y, z], dtype=np.float32)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arctan2(v_norm, w))
    axis = v / v_norm
    return axis * angle


def _wrap_to_pi(a: float) -> float:
    a = float(a)
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _nearest_equiv_yaw(desired_yaw: float, current_yaw: float, *, symmetry: int) -> float:
    """
    Choose the yaw equivalent (mod 2pi/symmetry) closest to current_yaw.
    symmetry=4 means 90-degree rotational symmetry.
    """
    base = float(desired_yaw)
    cur = float(current_yaw)
    step = (2.0 * np.pi) / max(int(symmetry), 1)
    candidates = [base + k * step for k in range(int(symmetry))]
    best = candidates[0]
    best_abs = 1e9
    for c in candidates:
        d = abs(_wrap_to_pi(c - cur))
        if d < best_abs:
            best_abs = d
            best = c
    return float(cur + _wrap_to_pi(best - cur))


def _aligned_grasp_quat_xyzw(env, *, eef_quat_xyzw: np.ndarray, obj_quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Keep roll/pitch stable, but align yaw so the gripper closes along the object's main axes.

    This helps cross / cube grasps present a flatter contact surface (less edge / corner contact).
    """
    eef_r, eef_p, eef_y = quat_to_euler_xyz(np.asarray(eef_quat_xyzw, dtype=np.float32))
    _obj_r, _obj_p, obj_y = quat_to_euler_xyz(np.asarray(obj_quat_xyzw, dtype=np.float32))

    obj_idx = int(getattr(env, "object_A_index", 0))
    # cross (0) and cube (1): 90-degree symmetry around vertical axis.
    if obj_idx in (0, 1):
        yaw = _nearest_equiv_yaw(obj_y, eef_y, symmetry=4)
        return euler_xyz_to_quat(eef_r, eef_p, yaw)
    # cylinder (2): yaw doesn't matter much; keep current.
    return canonicalize_quaternion(np.asarray(eef_quat_xyzw, dtype=np.float32))


def _yaw_only_aligned_grasp_quat_xyzw(env, *, eef_quat_xyzw: np.ndarray, obj_quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Like `_aligned_grasp_quat_xyzw`, but enforces a *pure yaw* change about the vertical axis.

    - Keeps the gripper "vertical" (no roll/pitch adjustment)
    - Chooses the nearest equivalent yaw under 90-degree symmetry (cross/cube), so the required
      rotation magnitude is minimal (<= 45 degrees).
    """
    q_cur = canonicalize_quaternion(np.asarray(eef_quat_xyzw, dtype=np.float32))
    _eef_r, _eef_p, eef_y = quat_to_euler_xyz(q_cur)
    _obj_r, _obj_p, obj_y = quat_to_euler_xyz(np.asarray(obj_quat_xyzw, dtype=np.float32))

    obj_idx = int(getattr(env, "object_A_index", 0))
    if obj_idx not in (0, 1):
        return q_cur

    desired_yaw = _nearest_equiv_yaw(obj_y, eef_y, symmetry=4)
    delta_yaw = _wrap_to_pi(desired_yaw - float(eef_y))
    q_delta = euler_xyz_to_quat(0.0, 0.0, float(delta_yaw))
    return canonicalize_quaternion(quaternion_multiply(q_delta, q_cur))


def transform_to_relative_coordinates(eef_pos: np.ndarray, eef_quat: np.ndarray, obj_data: np.ndarray) -> np.ndarray:
    obj_pos = obj_data[:3]
    obj_quat = obj_data[3:7]
    pos_rel = obj_pos - eef_pos
    quat_rel = quaternion_multiply(quaternion_inverse(eef_quat), obj_quat)
    quat_rel = canonicalize_quaternion(quat_rel)
    return np.concatenate([pos_rel, quat_rel]).astype(np.float32)


def _write_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    import imageio.v2 as iio  # type: ignore

    writer = iio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        output_params=["-pix_fmt:v", "yuv420p", "-crf", "18"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _make_video_frames(agentview: np.ndarray, eye_in_hand: np.ndarray) -> list[np.ndarray]:
    divider = np.zeros((agentview.shape[1], 2, 3), dtype=np.uint8)
    divider[:, :, 0] = 255
    return [np.concatenate([agentview[t], divider, eye_in_hand[t]], axis=1) for t in range(len(agentview))]


def _task_short(task: str) -> str:
    words = task.lower().split()
    obj = next((w for w in words if w in OBJECT_NAMES), "obj")
    cont = next((w for w in words if w in CONTAINER_NAMES), "cont")
    return f"{obj}_{cont}"


def fix_env_task_pointers(env) -> None:
    """Update live object references after task / index changes."""
    env.object_A = env.object_A_list[env.object_A_index]
    env.object_B = env.object_B_list[env.object_B_index]
    env.object_A_body_id = env.sim.model.body_name2id(env.object_A.root_body)
    env.object_B_body_id = env.sim.model.body_name2id(env.object_B.root_body)


def load_cg_l4_env_class():
    """
    Load the environment class from CG_L4.py.

    The current repo may still export the class as CG_L2 inside CG_L4.py, so
    this helper accepts either symbol.
    """
    module = importlib.import_module("robosuite.environments.manipulation.CG_L4")
    env_cls = getattr(module, "CG_L4", None)
    if env_cls is None:
        env_cls = getattr(module, "CG_L2", None)
    if env_cls is None:
        raise AttributeError("Could not find CG_L4 or CG_L2 class in CG_L4.py")
    return env_cls


def make_env(*, horizon: int):
    env_cls = load_cg_l4_env_class()
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
    return env_cls(
        robots="Panda",
        controller_configs=controller_config,
        gripper_types="PandaGripper",
        strategy="fixed",
        task="place the cross into the bin",  # overridden on every reset
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


class SingleStageCGWrapper:
    """
    Minimal vectorized wrapper around CG_L4 for planner-based generation.

    Observations returned:
      observation.state              (9,)
      observation.environment_state  (73,)   10 x 7D EEF-relative poses + gripper_to_target
      observation.object_world       (70,)   10 x 7D world-frame poses
      observation.images.agentview
      observation.images.robot0_eye_in_hand
    """

    def __init__(self, make_env_fn, num_envs: int = 1, use_relative_coordinates: bool = True):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.use_relative_coordinates = use_relative_coordinates
        self.obs: list[dict | None] = [None] * num_envs
        self.tasks: list[str] = [""] * num_envs
        self.obj_indices: list[int] = [0] * num_envs
        self.cont_indices: list[int] = [0] * num_envs
        self.is_success: list[bool] = [False] * num_envs

    def _safe_reset_env(self, env, obj_idx: int, cont_idx: int, max_attempts: int = 10, **kwargs):
        last_error = None
        for _ in range(max_attempts):
            try:
                obs = env.reset(**kwargs)
                fix_env_task_pointers(env)
                obs = env._get_observations()
                return obs
            except RandomizationError as exc:
                last_error = exc
        raise last_error if last_error is not None else RuntimeError("env.reset() failed without exception")

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        self.obs = []
        for i, env in enumerate(self.envs):
            task = str(np.random.choice(ALL_TASKS))
            # task = "place the cross into the mug"
            self.tasks[i] = task
            self.obj_indices[i] = next(idx for idx, name in enumerate(OBJECT_NAMES) if name in task)
            self.cont_indices[i] = next(idx for idx, name in enumerate(CONTAINER_NAMES) if name in task)
            env.strategy = "fixed"
            env.task = task
            obs_i = self._safe_reset_env(
                env,
                obj_idx=self.obj_indices[i],
                cont_idx=self.cont_indices[i],
                **kwargs,
            )
            self.obs.append(obs_i)
        self.is_success = [False] * self.num_envs
        obs_dicts = [self._compute_observation(i) for i in range(self.num_envs)]
        batched = {k: np.stack([obs[k] for obs in obs_dicts], axis=0) for k in obs_dicts[0]}
        return batched, [{} for _ in range(self.num_envs)]

    def step(self, actions: np.ndarray):
        next_obs, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            if self.is_success[i]:
                obs_i, reward, terminated, info = env.step(np.zeros_like(actions[i]))
            else:
                obs_i, reward, terminated, info = env.step(actions[i].copy())

            self.obs[i] = obs_i
            info["is_success"] = bool(env._check_success())
            if not self.is_success[i]:
                self.is_success[i] = info["is_success"]

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
            pose = np.concatenate([raw[f"{name}_pos"], raw[f"{name}_quat"]]).astype(np.float32)
            object_poses_world.append(pose)
            if self.use_relative_coordinates:
                object_poses_rel.append(transform_to_relative_coordinates(eef_pos, eef_quat, pose))
            else:
                object_poses_rel.append(pose)

        object_world = np.concatenate(object_poses_world).astype(np.float32)
        target_pos = env.sim.data.body_xpos[env.object_A_body_id].astype(np.float32)
        gripper_to_target = (target_pos - eef_pos).astype(np.float32)
        environment_state = np.concatenate(
            object_poses_rel + [gripper_to_target]
        ).astype(np.float32)

        out = {
            "observation.state": np.concatenate([eef_pos, eef_quat, gripper_q]).astype(np.float32),
            "observation.environment_state": environment_state,
            "observation.object_world": object_world,
        }

        if "agentview_image" in raw:
            out["observation.images.agentview"] = np.flipud(raw["agentview_image"].astype(np.uint8))
        if "robot0_eye_in_hand_image" in raw:
            out["observation.images.robot0_eye_in_hand"] = np.flipud(raw["robot0_eye_in_hand_image"].astype(np.uint8))

        return out

    def close(self):
        for env in self.envs:
            env.close()


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray
    quat: np.ndarray | None


@dataclass(frozen=True)
class TargetPoses:
    object_pose: Pose
    container_pose: Pose


@dataclass(frozen=True)
class GraspReleasePoses:
    pregrasp: Pose
    grasp: Pose
    preplace: Pose
    place: Pose


@dataclass(frozen=True)
class StageOffsetRule:
    """
    Optional, manually-tuned XYZ offsets (world frame, meters) applied to stage target points.

    Matching rules:
    - `stage` must match exactly (e.g. "pregrasp", "grasp", "preplace", "place")
    - `object` / `container` are optional; when provided they must match names in
      `OBJECT_NAMES` / `CONTAINER_NAMES`.
    - Offsets are additive if multiple rules match.
    """

    stage: str
    delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object: str | None = None
    container: str | None = None


@dataclass
class PickPlaceParams:
    # If True, command OSC orientation deltas (axis-angle) to track planned quaternions.
    # If False, keep translation-only OSC (legacy / more stable on some setups).
    use_orientation_control: bool = False
    # If False, never do grasp yaw alignment / rotation stages (even for cross/cube).
    rotate_for_grasp_enable: bool = True
    # Optional manual offsets for different object/container combinations.
    # Applied to the generated stage targets in `generate_grasp_release_poses()`.
    stage_offset_rules: list[StageOffsetRule] = field(default_factory=list)
    # Legacy: full lift after grasp (unused by current planner; kept for config compatibility).
    lift_z: float = 0.26
    # Small vertical clearance after grasp before moving toward the container (m).
    post_grasp_lift_z: float = 0.07
    # Release pose: Z offset above container body center (m). XY matches container center.
    # Bin uses the base value; plate lower, mug higher (see _release_z_above_container_for_env).
    release_z_above_container_center: float = 0.10
    release_z_above_container_center_plate: float = 0.055
    release_z_above_container_center_mug: float = 0.15
    # above object center
    pregrasp_h: float = 0.08
    # above placement center (used only by generate_grasp_release_poses for non-planner tooling)
    preplace_h: float = 0.09
    # tolerance for object<->container XY alignment (meters)
    xy_tol: float = 0.006
    # smooth trajectory discretization (meters per waypoint segment)
    step_size: float = 0.060
    # minimum number of points per segment (including endpoint samples)
    min_segment_points: int = 3
    # Mug only: lift this much (m) along Z after lowering, before opening — clears the rim / avoids scraping.
    mug_release_z_lift: float = 0.03
    # Smooth path tracking (fractional index along `_poses`, advance per env step — avoids stop-go at each sample).
    path_speed_m_per_step: float = 0.009 
    path_u_max_step: float = 0.82
    path_end_pos_tol_m: float = 0.022
    # --- Pose noise for diversity (meters) ---
    # Noise is applied in world frame as independent Gaussian noise:
    #   dx,dy ~ N(0, noise_xy_*), dz ~ N(0, noise_z_*)
    # Recommended: z std > xy std.
    pose_noise_enable: bool = True
    # Pregrasp: larger noise
    noise_pregrasp_xy: float = 0.010
    noise_pregrasp_z: float = 0.020
    # Grasp: smaller noise (more precise)
    noise_grasp_xy: float = 0.003
    noise_grasp_z: float = 0.001
    # Preplace: larger noise (approach)
    noise_preplace_xy: float = 0.012
    noise_preplace_z: float = 0.01
    # Place: smaller noise (precision at release)
    noise_place_xy: float = 0.004
    noise_place_z: float = 0.001


@dataclass(frozen=True)
class Waypoint:
    pos: np.ndarray
    quat: np.ndarray | None
    gripper: float

    @staticmethod
    def from_pose(p: Pose, gripper: float) -> "Waypoint":
        return Waypoint(
            pos=np.asarray(p.pos, dtype=np.float32),
            quat=(p.quat.copy() if p.quat is not None else None),
            gripper=float(gripper),
        )


def get_target_poses(env) -> TargetPoses:
    """
    Get target object + container center pose (world frame).
    Quaternion returned in xyzw to match robosuite observations.
    """
    obj_pos = env.sim.data.body_xpos[env.object_A_body_id].copy().astype(np.float32)
    cont_pos = env.sim.data.body_xpos[env.object_B_body_id].copy().astype(np.float32)
    obj_quat = quat_wxyz_to_xyzw(env.sim.data.body_xquat[env.object_A_body_id].copy().astype(np.float32))
    cont_quat = quat_wxyz_to_xyzw(env.sim.data.body_xquat[env.object_B_body_id].copy().astype(np.float32))
    return TargetPoses(object_pose=Pose(obj_pos, obj_quat), container_pose=Pose(cont_pos, cont_quat))


def _container_place_z_offset(env, *, obj_half_h: float) -> float:
    # Heuristic offsets to place object into different container types.
    if env.object_B_index == 0:  # bin
        return obj_half_h + 0.006
    if env.object_B_index == 1:  # mug
        return max(obj_half_h + 0.010, 0.022)
    if env.object_B_index == 2:  # plate
        return obj_half_h + 0.004
    return obj_half_h + 0.008


def _release_z_above_container_for_env(env, params: PickPlaceParams) -> float:
    """Z above container body center for open-gripper release; plate lower, mug higher."""
    idx = int(env.object_B_index)
    if idx == 1:  # mug
        return float(params.release_z_above_container_center_mug)
    if idx == 2:  # plate
        return float(params.release_z_above_container_center_plate)
    return float(params.release_z_above_container_center)


def generate_grasp_release_poses(
    env,
    targets: TargetPoses,
    *,
    eef_quat_xyzw: np.ndarray | None,
    params: PickPlaceParams,
) -> GraspReleasePoses:
    """
    Generate grasp + release point position and orientation.
    Orientation is kept as current EEF orientation (stable). Swap to a fixed top-down quat if needed.
    """
    obj_half = float(env.object_A.top_offset[-1])

    def _sample_delta(*, xy_std: float, z_std: float) -> np.ndarray:
        dxy = np.random.normal(loc=0.0, scale=float(xy_std), size=(2,)).astype(np.float32)
        dz = np.random.normal(loc=0.0, scale=float(z_std), size=(1,)).astype(np.float32)
        return np.array([dxy[0], dxy[1], dz[0]], dtype=np.float32)

    def _extra_std(large: float, small: float) -> float:
        # if we want pre* to be "larger noise" but correlated with the smaller-noise pose
        return float(np.sqrt(max(float(large) ** 2 - float(small) ** 2, 0.0)))

    pregrasp0 = targets.object_pose.pos.copy().astype(np.float32)
    pregrasp0[2] += float(params.pregrasp_h)

    grasp0 = targets.object_pose.pos.copy().astype(np.float32)
    grasp0[2] += float(np.clip(0.08 * obj_half, 0.001, 0.002))

    place0 = targets.container_pose.pos.copy().astype(np.float32)
    place0[2] += float(_container_place_z_offset(env, obj_half_h=obj_half))

    preplace0 = place0.copy()
    preplace0[2] += float(params.preplace_h)

    # Apply stage-dependent noise (z larger than xy).
    if bool(params.pose_noise_enable):
        d_place = _sample_delta(xy_std=params.noise_place_xy, z_std=params.noise_place_z)
        d_preplace = _sample_delta(
            xy_std=_extra_std(params.noise_preplace_xy, params.noise_place_xy),
            z_std=_extra_std(params.noise_preplace_z, params.noise_place_z),
        )
        d_preplace[2] = abs(d_preplace[2])
        d_preplace += d_place

        d_grasp = _sample_delta(xy_std=params.noise_grasp_xy, z_std=params.noise_grasp_z)
        d_pregrasp = _sample_delta(
            xy_std=_extra_std(params.noise_pregrasp_xy, params.noise_grasp_xy),
            z_std=_extra_std(params.noise_pregrasp_z, params.noise_grasp_z),
        )
        d_pregrasp[2] = abs(d_pregrasp[2])
        d_pregrasp += d_grasp

        place = place0 + d_place
        preplace = preplace0 + d_preplace
        grasp = grasp0 + d_grasp
        pregrasp = pregrasp0 + d_pregrasp
    else:
        place = place0
        preplace = preplace0
        grasp = grasp0
        pregrasp = pregrasp0

    def _name_or(names: list[str], idx: int, default: str) -> str:
        return names[idx] if 0 <= int(idx) < len(names) else default

    def _stage_offset(stage: str) -> np.ndarray:
        obj = _name_or(OBJECT_NAMES, int(getattr(env, "object_A_index", -1)), "obj")
        cont = _name_or(CONTAINER_NAMES, int(getattr(env, "object_B_index", -1)), "cont")
        d = np.zeros(3, dtype=np.float32)
        for rule in params.stage_offset_rules:
            if str(rule.stage) != str(stage):
                continue
            if rule.object is not None and str(rule.object) != obj:
                continue
            if rule.container is not None and str(rule.container) != cont:
                continue
            d += np.asarray(rule.delta_xyz, dtype=np.float32)
        return d

    # Manual, object/container-specific offsets (world frame).
    pregrasp = (pregrasp + _stage_offset("pregrasp")).astype(np.float32)
    grasp = (grasp + _stage_offset("grasp")).astype(np.float32)
    preplace = (preplace + _stage_offset("preplace")).astype(np.float32)
    place = (place + _stage_offset("place")).astype(np.float32)

    # Safety: keep all Z above the table.
    table_z = float(env.model.mujoco_arena.table_offset[2])
    z_floor = table_z + 0.01
    for p in (pregrasp, grasp, preplace, place):
        p[2] = max(float(p[2]), z_floor)

    q = (eef_quat_xyzw.copy() if eef_quat_xyzw is not None else None)
    return GraspReleasePoses(
        pregrasp=Pose(pregrasp.astype(np.float32), q),
        grasp=Pose(grasp.astype(np.float32), q),
        preplace=Pose(preplace.astype(np.float32), q),
        place=Pose(place.astype(np.float32), q),
    )


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def quat_slerp(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, t: float) -> np.ndarray:
    q0 = canonicalize_quaternion(np.asarray(q0_xyzw, dtype=np.float32))
    q1 = canonicalize_quaternion(np.asarray(q1_xyzw, dtype=np.float32))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + float(t) * (q1 - q0)
        out /= max(float(np.linalg.norm(out)), 1e-8)
        return canonicalize_quaternion(out.astype(np.float32))
    theta0 = float(np.arccos(dot))
    sin0 = float(np.sin(theta0))
    theta = theta0 * float(t)
    s0 = float(np.sin(theta0 - theta)) / sin0
    s1 = float(np.sin(theta)) / sin0
    out = s0 * q0 + s1 * q1
    return canonicalize_quaternion(out.astype(np.float32))


# Matches default composite BASIC + OSC_POSE for Panda (see robosuite basic.json).
_OSC_POS_MAX_M = 0.05
_OSC_ROT_MAX_RAD = 0.5


def _robot_base_rotmat(env) -> np.ndarray:
    return env.sim.data.get_body_xmat(env.robots[0].robot_model.root_body).reshape(3, 3).astype(np.float32)


def _eef_site_pos(env) -> np.ndarray:
    return env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].astype(np.float32)


def _sample_polyline(
    points: list[np.ndarray],
    *,
    step_size: float,
    min_segment_points: int,
    end_quat_xyzw: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Dense samples along a polyline (piecewise linear). Quaternions are constant (top-down grasp).
    Uses smoothstep along each chord so points bunch slightly near segment ends (gentler motion).
    """
    if len(points) < 2:
        q = end_quat_xyzw.astype(np.float32)
        return [points[0].copy()], [q.copy()]

    pos_out: list[np.ndarray] = []
    quat_out: list[np.ndarray] = []
    q = end_quat_xyzw.astype(np.float32)

    for seg in range(len(points) - 1):
        p0 = points[seg].astype(np.float32)
        p1 = points[seg + 1].astype(np.float32)
        dist = float(np.linalg.norm(p1 - p0))
        if dist < 1e-6:
            continue
        n = max(min_segment_points, int(np.ceil(dist / max(step_size, 1e-6))) + 1)
        for j in range(n):
            t = j / (n - 1) if n > 1 else 1.0
            u = _smoothstep01(float(t))
            pos_out.append((p0 + (p1 - p0) * u).astype(np.float32))
            quat_out.append(q.copy())

    return pos_out, quat_out


@dataclass
class _TimedPose:
    pos: np.ndarray
    quat: np.ndarray
    grip: float
    phase: str
    repeat: int = 1


class WaypointPickPlacePlanner:
    """
    Scripted pick-place: dense world-frame waypoints + OSC delta actions toward each target.

    Motion uses a fractional index along the polyline (linear interpolation between consecutive samples)
    with a bounded advance per step, so the end-effector does not stop at every discrete waypoint.
    Grasp/release still use timed dwells at the planned poses. Translation-only OSC preserves orientation.

    Phases used by run_batch: mostly 'approach' / 'place', then 'retreat' after releasing, then 'done'.
    """

    GRIP_CLOSE = 1.0
    GRIP_OPEN = -1.0

    def __init__(self, params: PickPlaceParams | None = None):
        self.params = params or PickPlaceParams()
        self.phase: str = "init"
        self.done: bool = False
        self._poses: list[_TimedPose] = []
        self._path_u: float = 0.0
        self._dwell_remaining: int | None = None
        self._dwell_index: int | None = None
        self._built: bool = False
        self._use_orientation_control: bool = bool(self.params.use_orientation_control)

    def reset(self) -> None:
        self.phase = "init"
        self.done = False
        self._poses = []
        self._path_u = 0.0
        self._dwell_remaining = None
        self._dwell_index = None
        self._built = False
        self._use_orientation_control = bool(self.params.use_orientation_control)

    def _is_grasped(self, env) -> bool:
        return bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.object_A))

    def _append_hold(self, pos: np.ndarray, quat: np.ndarray, grip: float, phase: str, n: int) -> None:
        self._poses.append(_TimedPose(pos=pos.copy(), quat=quat.copy(), grip=float(grip), phase=phase, repeat=max(1, int(n))))

    def _build(self, env, eef_quat_xyzw: np.ndarray) -> None:
        p = self.params
        targets = get_target_poses(env)
        if bool(p.use_orientation_control) and bool(p.rotate_for_grasp_enable):
            q_path = _yaw_only_aligned_grasp_quat_xyzw(
                env,
                eef_quat_xyzw=eef_quat_xyzw,
                obj_quat_xyzw=targets.object_pose.quat,  # type: ignore[arg-type]
            )
        else:
            q_path = canonicalize_quaternion(np.asarray(eef_quat_xyzw, dtype=np.float32))
        gr = generate_grasp_release_poses(env, targets, eef_quat_xyzw=q_path, params=p)

        eef0 = _eef_site_pos(env)
        q_eef = canonicalize_quaternion(np.asarray(eef_quat_xyzw, dtype=np.float32))



        pre, grsp = gr.pregrasp, gr.grasp
        table_z = float(env.model.mujoco_arena.table_offset[2])

        # --- approach: start -> pregrasp -> grasp ---
        line1 = [eef0, pre.pos, grsp.pos]
        pos1, quat1 = _sample_polyline(line1, step_size=p.step_size, min_segment_points=p.min_segment_points, end_quat_xyzw=q_path)
        for i in range(len(pos1)):
            ph = "approach"
            self._poses.append(_TimedPose(pos=pos1[i], quat=quat1[i], grip=self.GRIP_OPEN, phase=ph, repeat=1))

        # Close gripper at grasp pose (hold to allow contacts).
        self._append_hold(grsp.pos, q_path, self.GRIP_CLOSE, "approach", n=18)

        # --- after grasp: small lift, then move to release (above container center) ---
        lift_slight = grsp.pos.copy()
        lift_slight[2] = max(float(grsp.pos[2] + p.post_grasp_lift_z), table_z + 0.12)

        cont = targets.container_pose.pos.copy()
        rz = _release_z_above_container_for_env(env, p)
        release_pos = np.array(
            [float(cont[0]), float(cont[1]), float(cont[2]) + rz],
            dtype=np.float32,
        )

        pos2, quat2 = _sample_polyline(
            [grsp.pos, lift_slight],
            step_size=p.step_size,
            min_segment_points=p.min_segment_points,
            end_quat_xyzw=q_path,
        )
        for i in range(len(pos2)):
            self._poses.append(_TimedPose(pos=pos2[i], quat=quat2[i], grip=self.GRIP_CLOSE, phase="transfer", repeat=1))

        pos3, quat3 = _sample_polyline(
            [lift_slight, release_pos],
            step_size=p.step_size,
            min_segment_points=p.min_segment_points,
            end_quat_xyzw=q_path,
        )
        for i in range(len(pos3)):
            self._poses.append(_TimedPose(pos=pos3[i], quat=quat3[i], grip=self.GRIP_CLOSE, phase="transfer", repeat=1))

        # Open above container; object drops from height (XY over container center).
        self._append_hold(release_pos, q_path, self.GRIP_OPEN, "place", n=22)

        # --- retreat: CG_L4._check_success() needs gripper_z - table_z > 0.20; release is low, so ensure
        # the commanded retreat height clears that (not only +0.10 above a rim-level pose).
        retreat = release_pos.copy()
        retreat[2] += 0.10
        retreat[1] += 0.04
        retreat[2] = max(float(retreat[2]), table_z + 0.21)
        pos5, quat5 = _sample_polyline(
            [release_pos, retreat],
            step_size=p.step_size,
            min_segment_points=p.min_segment_points,
            end_quat_xyzw=q_path,
        )
        for i in range(len(pos5)):
            self._poses.append(_TimedPose(pos=pos5[i], quat=quat5[i], grip=self.GRIP_OPEN, phase="retreat", repeat=1))

    def _apply_xy_feedback(self, env, pos: np.ndarray, phase: str) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32).copy()
        # During transfer / place, snap target XY to live container center.
        if phase in ("transfer", "place"):
            pos[:2] = env.sim.data.body_xpos[env.object_B_body_id][:2].astype(np.float32)
        # Final approach: snap target XY to live object center when close.
        elif phase == "approach":
            eef_xy = _eef_site_pos(env)[:2]
            live_o = env.sim.data.body_xpos[env.object_A_body_id][:2].astype(np.float32)
            if float(np.linalg.norm(eef_xy - live_o)) < 0.14:
                pos[:2] = live_o
        return pos

    def _interp_state(self, env, u: float) -> tuple[np.ndarray, np.ndarray, float, str]:
        """
        Linear interpolation between consecutive poses in index space. Start (u=0) and end (u=n-1)
        match the first and last planned positions; orientation follows the waypoint quaternions (slerp).
        """
        n = len(self._poses)
        u = float(np.clip(u, 0.0, float(n - 1)))
        i = int(np.floor(u))
        if i >= n - 1:
            p = self._poses[n - 1]
            return (
                self._apply_xy_feedback(env, p.pos, p.phase),
                p.quat,
                float(p.grip),
                p.phase,
            )
        t = float(u - float(i))
        p0, p1 = self._poses[i], self._poses[i + 1]
        pos_raw = ((1.0 - t) * p0.pos.astype(np.float64) + t * p1.pos.astype(np.float64)).astype(np.float32)
        phase = p0.phase if t < 0.5 else p1.phase
        pos = self._apply_xy_feedback(env, pos_raw, phase)
        quat = quat_slerp(p0.quat, p1.quat, t)
        # Do not apply next pose's grip until a dwell (e.g. close at grasp) — avoids closing early.
        if int(p1.repeat) > 1:
            grip = float(p0.grip)
        else:
            grip = float(p0.grip) if t < 1.0 - 1e-6 else float(p1.grip)
        ph = p0.phase if t < 0.5 else p1.phase
        return pos, quat, grip, ph

    def _advance_path_u(self) -> None:
        n = len(self._poses)
        u = float(self._path_u)
        if u >= float(n - 1) - 1e-12:
            return

        i = int(min(np.floor(u + 1e-9), n - 2))
        j = min(i + 1, n - 1)
        p0, p1 = self._poses[i].pos, self._poses[j].pos
        seg_len = float(np.linalg.norm(p1 - p0))
        p = self.params
        if seg_len < 1e-8:
            du = p.path_u_max_step
        else:
            du = min(p.path_speed_m_per_step / seg_len, p.path_u_max_step)
        u_next = min(u + du, float(n - 1))

        k_enter = int(math.floor(u_next + 1e-9))
        u_floor = int(math.floor(u + 1e-9))
        for k in range(u_floor + 1, k_enter + 1):
            if k < n and int(self._poses[k].repeat) > 1:
                u_next = float(k)
                self._dwell_index = k
                self._dwell_remaining = int(self._poses[k].repeat)
                break

        self._path_u = u_next

    def _pose_to_action(
        self,
        env,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        cur_quat_xyzw: np.ndarray,
        grip_cmd: float,
    ) -> np.ndarray:
        eef_pos = _eef_site_pos(env)
        Rb = _robot_base_rotmat(env)
        err_w = (target_pos - eef_pos).astype(np.float32)
        err_b = (Rb.T @ err_w).astype(np.float32)

        # Position: normalized OSC delta (one-step proportional toward goal).
        act = np.zeros(7, dtype=np.float32)
        act[:3] = np.clip(err_b / _OSC_POS_MAX_M, -1.0, 1.0)

        # Orientation: axis-angle delta in base frame (OSC_POSE expects delta, ref frame = base).
        # Keep EEF vertical (aligned with target_quat_xyzw, which is a downward-pointing pose)
        # throughout all phases to prevent roll/pitch drift during transfer and retreat.
        if self._use_orientation_control:
            q_tgt = canonicalize_quaternion(np.asarray(target_quat_xyzw, dtype=np.float32))
            q_cur = canonicalize_quaternion(np.asarray(cur_quat_xyzw, dtype=np.float32))
            q_err = quaternion_multiply(q_tgt, quaternion_inverse(q_cur))
            rotvec_w = quat_to_rotvec(q_err)
            rotvec_b = (Rb.T @ rotvec_w.astype(np.float32)).astype(np.float32)
            act[3:6] = np.clip(rotvec_b / _OSC_ROT_MAX_RAD, -1.0, 1.0)
        else:
            act[3:6] = 0.0

        act[6] = float(grip_cmd)
        return act

    def get_action(self, env, eef_quat_xyzw: np.ndarray) -> np.ndarray:
        if not self._built:
            self._build(env, eef_quat_xyzw)
            self._built = True

        if self.done or not self._poses:
            self.done = True
            self.phase = "done"
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.GRIP_OPEN], dtype=np.float32)

        n = len(self._poses)

        # Timed holds (grasp close, gripper open settle) — exact pose from plan.
        if self._dwell_remaining is not None and self._dwell_index is not None and self._dwell_remaining > 0:
            cur = self._poses[self._dwell_index]
            self.phase = cur.phase
            tgt_pos = self._apply_xy_feedback(env, cur.pos, cur.phase)
            a = self._pose_to_action(env, tgt_pos, cur.quat, eef_quat_xyzw, cur.grip)
            self._dwell_remaining -= 1
            if self._dwell_remaining <= 0:
                self._path_u = float(self._dwell_index) + 1.0
                self._dwell_remaining = None
                self._dwell_index = None
            return a

        # Terminal: last planned pose (retreat end) until convergence, then zero action.
        if self._path_u >= float(n - 1) - 1e-9:
            last = self._poses[n - 1]
            self.phase = last.phase
            tgt_pos = self._apply_xy_feedback(env, last.pos, last.phase)
            a = self._pose_to_action(env, tgt_pos, last.quat, eef_quat_xyzw, last.grip)
            eef = _eef_site_pos(env)
            if float(np.linalg.norm(tgt_pos - eef)) < float(self.params.path_end_pos_tol_m):
                self.done = True
                self.phase = "done"
            return a

        pos, quat, grip, ph = self._interp_state(env, self._path_u)
        self.phase = ph
        a = self._pose_to_action(env, pos, quat, eef_quat_xyzw, grip)
        self._advance_path_u()
        return a


@dataclass(frozen=True)
class SimpleWaypoint:
    pos: np.ndarray
    quat_xyzw: np.ndarray | None
    gripper_move: float
    gripper_hold: float
    name: str
    repeat: int = 1


class PickPlaceWaypointGenerator:
    """
    Single-responsibility: compute a minimal waypoint list.
    Current stage: pregrasp -> grasp -> prerelease -> release -> lift.
    """

    def __init__(self, *, params: PickPlaceParams | None = None, grasp_hold_steps: int = 18, release_hold_steps: int = 22):
        self.params = params or PickPlaceParams()
        self.grasp_hold_steps = int(grasp_hold_steps)
        self.release_hold_steps = int(release_hold_steps)

    def generate(self, env, *, eef_quat_xyzw: np.ndarray) -> list[SimpleWaypoint]:
        def _scene_clearance_z(*, margin: float = 0.10, min_above_table: float = 0.18) -> float:
            """
            Compute a conservative Z height above all objects / containers.

            We scan known CG_L4 body ids when present (cross/cube/cylinder/milk/bin/mug/plate/mug_no_handle).
            """
            table_z = float(env.model.mujoco_arena.table_offset[2])
            z_vals: list[float] = []
            for attr in (
                "cross_body_id",
                "cube_body_id",
                "cylinder_body_id",
                "milk_body_id",
                "bin_body_id",
                "mug_body_id",
                "plate_body_id",
                "mug_no_handle_body_id",
                "object_A_body_id",
                "object_B_body_id",
            ):
                bid = getattr(env, attr, None)
                if bid is None:
                    continue
                try:
                    z_vals.append(float(env.sim.data.body_xpos[int(bid)][2]))
                except Exception:
                    continue
            z_max = max(z_vals) if z_vals else table_z
            return max(z_max + float(margin), table_z + float(min_above_table))

        targets = get_target_poses(env)
        # Do not align / control orientation for this simplified generator (translation-only).
        q = canonicalize_quaternion(np.asarray(eef_quat_xyzw, dtype=np.float32))
        gr = generate_grasp_release_poses(env, targets, eef_quat_xyzw=q, params=self.params)

        # After grasp: lift straight up to clear all objects / containers to avoid collisions.
        post_grasp_lift = gr.grasp.pos.astype(np.float32).copy()
        post_grasp_lift[2] = float(_scene_clearance_z())

        prerelease = gr.preplace.pos.astype(np.float32)  # higher than release; same (x,y)
        release = gr.place.pos.astype(np.float32)

        # Lift after release so CG_L4 lift_check can pass (and to clear rims).
        table_z = float(env.model.mujoco_arena.table_offset[2])
        lift = release.copy()
        lift[2] = max(float(lift[2] + 0.12), table_z + 0.22)
        lift[1] = float(lift[1] + 0.04)

        quat_xyzw = None

        waypoints: list[SimpleWaypoint] = [
            SimpleWaypoint(
                pos=gr.pregrasp.pos.astype(np.float32),
                quat_xyzw=quat_xyzw,
                gripper_move=-1.0,
                gripper_hold=-1.0,
                name="pregrasp",
            ),
        ]

        obj_idx = int(getattr(env, "object_A_index", 0))
        # Only rotate-for-grasp for objects that benefit from yaw alignment.
        # Cylinder (2) does not need rotation.
        if bool(self.params.use_orientation_control) and bool(self.params.rotate_for_grasp_enable) and (obj_idx in (0, 1)):
            waypoints.append(
                SimpleWaypoint(
                    pos=gr.pregrasp.pos.astype(np.float32),
                    # Target orientation will be computed at runtime to only rotate yaw
                    # relative to the current EEF orientation at pregrasp.
                    quat_xyzw=None,
                    gripper_move=-1.0,
                    gripper_hold=-1.0,
                    name="rotate_for_grasp",
                    repeat=6,
                )
            )

        waypoints += [
            SimpleWaypoint(
                pos=gr.grasp.pos.astype(np.float32),
                # Same yaw-aligned orientation target as rotate_for_grasp (computed at runtime)
                quat_xyzw=None,
                # Only close after reaching grasp.
                gripper_move=-1.0,
                gripper_hold=1.0,
                name="grasp",
                repeat=max(1, self.grasp_hold_steps),
            ),
            # Collision-avoid lift: go up before translating toward container.
            SimpleWaypoint(
                pos=post_grasp_lift,
                quat_xyzw=quat_xyzw,
                gripper_move=1.0,
                gripper_hold=1.0,
                name="post_grasp_lift",
            ),
            SimpleWaypoint(
                pos=prerelease,
                quat_xyzw=quat_xyzw,
                gripper_move=1.0,
                gripper_hold=1.0,
                name="prerelease",
            ),
            SimpleWaypoint(
                pos=release,
                quat_xyzw=quat_xyzw,
                # Only open after reaching release.
                gripper_move=1.0,
                gripper_hold=-1.0,
                name="release",
                repeat=max(1, self.release_hold_steps),
            ),
            SimpleWaypoint(
                pos=lift.astype(np.float32),
                quat_xyzw=quat_xyzw,
                gripper_move=-1.0,
                gripper_hold=-1.0,
                name="lift",
            ),
        ]

        return waypoints


class WaypointSequencer:
    """
    Single-responsibility: hold the active waypoint and advance when reached.
    """

    def __init__(self, *, pos_tol: float):
        self.pos_tol = float(pos_tol)
        self._wps: list[SimpleWaypoint] = []
        self._idx: int = 0
        self._holding: bool = False
        self._dwell_remaining: int = 0
        self.done: bool = False

    def reset(self, waypoints: list[SimpleWaypoint]) -> None:
        self._wps = list(waypoints)
        self._idx = 0
        self._holding = False
        self._dwell_remaining = 0
        # "done" means we've reached the end of a *non-empty* sequence.
        # An empty list is treated as "uninitialized" so the planner can populate it lazily.
        self.done = False

    def is_empty(self) -> bool:
        return len(self._wps) == 0

    def current(self) -> SimpleWaypoint | None:
        if self.done or self._idx >= len(self._wps):
            return None
        return self._wps[self._idx]

    def is_holding(self) -> bool:
        return bool(self._holding) and (self._dwell_remaining > 0)

    def advance(self, *, reached: bool) -> None:
        if self.done:
            return
        cur = self.current()
        if cur is None:
            self.done = True
            return
        if self._holding:
            self._dwell_remaining -= 1
            if self._dwell_remaining <= 0:
                self._holding = False
                self._idx += 1
                if self._idx >= len(self._wps):
                    self.done = True
            return

        if not bool(reached):
            return

        if int(cur.repeat) > 1:
            self._holding = True
            self._dwell_remaining = int(cur.repeat)
            return

        self._idx += 1
        if self._idx >= len(self._wps):
            self.done = True


class SimplePickPlacePlanner:
    """
    Minimal pipeline:
      - waypoint generator (pregrasp only)
      - waypoint sequencer (advance on tolerance)
      - controller (SimplePoseController)
    """

    def __init__(
        self,
        *,
        waypoint_params: PickPlaceParams | None = None,
        max_step_size: float = 0.02,
        pos_tol: float = 0.01,
        osc_pos_limit: float = 0.05,
        grasp_hold_steps: int = 20,
        release_hold_steps: int = 22,
    ):
        self.generator = PickPlaceWaypointGenerator(
            params=waypoint_params,
            grasp_hold_steps=grasp_hold_steps,
            release_hold_steps=release_hold_steps,
        )
        self.seq = WaypointSequencer(pos_tol=pos_tol)
        self.ctrl = SimplePoseController(
            SimpleControllerParams(
                max_step_size=float(max_step_size),
                pos_tolerance=float(pos_tol),
                osc_pos_limit=float(osc_pos_limit),
                # Orientation control is only used when a waypoint provides a target quaternion.
                use_orientation=True,
                rot_tolerance_rad=0.15,
                osc_rot_limit_rad=0.5,
                max_rot_step_rad=0.08,
                ignore_position=False,
            )
        )
        self.phase: str = "init"
        self.done: bool = False
        self.pregrasp_pos: np.ndarray | None = None
        self.last_eef_pos: np.ndarray | None = None
        self.last_waypoint: str | None = None
        self._cached_yaw_target_quat_xyzw: np.ndarray | None = None

    def reset(self) -> None:
        self.phase = "init"
        self.done = False
        self.pregrasp_pos = None
        self.last_eef_pos = None
        self.last_waypoint = None
        self._cached_yaw_target_quat_xyzw = None
        self.seq.reset([])

    def get_action(self, env, *, eef_quat_xyzw: np.ndarray) -> np.ndarray:
        self.last_eef_pos = _eef_site_pos(env).copy()
        if self.done:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        if self.seq.is_empty():
            wps = self.generator.generate(env, eef_quat_xyzw=eef_quat_xyzw)
            self.seq.reset(wps)
            if wps:
                self.pregrasp_pos = wps[0].pos.copy()

        cur = self.seq.current()
        if cur is None:
            self.done = True
            self.phase = "done"
            self.last_waypoint = "done"
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        self.phase = cur.name
        # Rotation is optional. If disabled, we always run position-only OSC.
        target_quat = cur.quat_xyzw
        if cur.name not in ("rotate_for_grasp", "grasp"):
            self._cached_yaw_target_quat_xyzw = None

        if (
            cur.name in ("rotate_for_grasp", "grasp")
            and bool(self.generator.params.use_orientation_control)
            and bool(self.generator.params.rotate_for_grasp_enable)
        ):
            if self._cached_yaw_target_quat_xyzw is None:
                targets = get_target_poses(env)
                self._cached_yaw_target_quat_xyzw = _yaw_only_aligned_grasp_quat_xyzw(
                    env,
                    eef_quat_xyzw=canonicalize_quaternion(np.asarray(eef_quat_xyzw, dtype=np.float32)),
                    obj_quat_xyzw=targets.object_pose.quat,  # type: ignore[arg-type]
                ).astype(np.float32)
            target_quat = self._cached_yaw_target_quat_xyzw
        else:
            target_quat = None

        act, pos_dist, rot_dist, _err = self.ctrl.compute_action(env, target_pos=cur.pos, target_quat_xyzw=target_quat)
        act = act.astype(np.float32)
        eef = _eef_site_pos(env)
        reached_pos = float(pos_dist) <= float(self.seq.pos_tol)
        reached_rot = (target_quat is None) or (float(rot_dist) <= float(self.ctrl.params.rot_tolerance_rad))
        reached = bool(reached_pos and reached_rot)
        # While moving: gripper_move. When holding (dwell) at waypoint: gripper_hold.
        if self.seq.is_holding() or (reached and int(cur.repeat) > 1):
            act[6] = float(cur.gripper_hold)
        else:
            act[6] = float(cur.gripper_move)

        self.seq.advance(reached=reached)
        if self.seq.done:
            self.done = True
            self.phase = "done"
            self.last_waypoint = cur.name
        return act


def save_episode_to_hdf5(hdf5_path: Path, ep: dict, ep_id: int) -> None:
    with h5py.File(hdf5_path, "a") as f:
        grp = f.require_group(f"demo_{ep_id}")
        obs_grp = grp.require_group("obs")

        obs_grp.create_dataset("robot0_eef_pos", data=ep["eef_pos"], compression="gzip")
        obs_grp.create_dataset("robot0_eef_quat", data=ep["eef_quat"], compression="gzip")
        obs_grp.create_dataset("robot0_gripper_qpos", data=ep["gripper_q"], compression="gzip")
        obs_grp.create_dataset("object", data=ep["obj_world"], compression="gzip")
        if "env_state" in ep:
            obs_grp.create_dataset("environment_state", data=ep["env_state"], compression="gzip")

        if "agentview" in ep:
            obs_grp.create_dataset(
                "agentview_image",
                data=ep["agentview"],
                compression="gzip",
                chunks=(1, *ep["agentview"].shape[1:]),
            )
        if "eye_in_hand" in ep:
            obs_grp.create_dataset(
                "robot0_eye_in_hand_image",
                data=ep["eye_in_hand"],
                compression="gzip",
                chunks=(1, *ep["eye_in_hand"].shape[1:]),
            )

        grp.create_dataset("actions", data=ep["actions"], compression="gzip")
        grp.attrs["task"] = str(ep["task"])
        grp.attrs["planner"] = str(ep.get("planner", "pregrasp_simple"))
        grp.attrs["success"] = bool(ep.get("success", False))


def run_batch(
    env: SingleStageCGWrapper,
    *,
    horizon: int,
    planner_params: PickPlaceParams | None = None,
) -> tuple[list[dict | None], dict]:
    obs, _ = env.reset()
    planners = [
        SimplePickPlacePlanner(
            waypoint_params=planner_params,
            max_step_size=0.05,
            pos_tol=0.005,
            osc_pos_limit=_OSC_POS_MAX_M,
        )
        for _ in range(env.num_envs)
    ]
    for planner in planners:
        planner.reset()

    done = [False] * env.num_envs
    ep_success = [False] * env.num_envs

    buf_eef_pos = [[] for _ in range(env.num_envs)]
    buf_eef_quat = [[] for _ in range(env.num_envs)]
    buf_gripper_q = [[] for _ in range(env.num_envs)]
    buf_obj_world = [[] for _ in range(env.num_envs)]
    buf_env_state = [[] for _ in range(env.num_envs)]
    buf_agentview = [[] for _ in range(env.num_envs)]
    buf_eye = [[] for _ in range(env.num_envs)]
    buf_actions = [[] for _ in range(env.num_envs)]

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
            eef_quat_xyzw = obs["observation.state"][i][3:7].astype(np.float32)
            action_mat[i] = planners[i].get_action(env.envs[i], eef_quat_xyzw=eef_quat_xyzw)
            buf_actions[i].append(action_mat[i].copy())

        # print(eef_quat_xyzw)
        # print(action_mat)

        obs, _reward, terminated, truncated, info = env.step(action_mat)

        for i in range(env.num_envs):
            if done[i]:
                continue
            # Full task success is provided by the environment.
            if bool(info["is_success"][i]):
                ep_success[i] = True
                done[i] = True
            elif bool(terminated[i]) or bool(truncated[i]):
                done[i] = True
            if done[i]:
                # Print pregrasp target vs final EEF for debugging (even for full task).
                tgt = getattr(planners[i], "pregrasp_pos", None)
                cur = getattr(planners[i], "last_eef_pos", None)
                last_wp = getattr(planners[i], "last_waypoint", None)
                if tgt is not None and cur is not None:
                    dist = float(np.linalg.norm(np.asarray(tgt) - np.asarray(cur)))
                    tag = "success" if ep_success[i] else "done"
                    print(
                        f"[env {i}] {tag} | last_wp={last_wp} | "
                        f"pregrasp_target={np.array(tgt)} | final_eef={np.array(cur)} | dist_to_pregrasp={dist:.4f} m"
                    )

    batch_successes = int(sum(ep_success))

    results: list[dict | None] = []
    for i in range(env.num_envs):
        if not buf_actions[i]:
            results.append(None)
            continue

        T = len(buf_actions[i])
        ep = {
            "eef_pos": np.stack(buf_eef_pos[i][:T]),
            "eef_quat": np.stack(buf_eef_quat[i][:T]),
            "gripper_q": np.stack(buf_gripper_q[i][:T]),
            "obj_world": np.stack(buf_obj_world[i][:T]),
            "env_state": np.stack(buf_env_state[i][:T]),
            "actions": np.stack(buf_actions[i]),
            "task": env.tasks[i],
            "success": bool(ep_success[i]),
            "planner": "simple_pick_place",
        }
        # Add debug scalars (not used by training; useful for checking convergence).
        tgt = getattr(planners[i], "pregrasp_pos", None)
        cur = getattr(planners[i], "last_eef_pos", None)
        if tgt is not None and cur is not None:
            ep["pregrasp_target_pos"] = np.asarray(tgt, dtype=np.float32)
            ep["final_eef_pos"] = np.asarray(cur, dtype=np.float32)
            ep["final_eef_dist_to_pregrasp"] = float(np.linalg.norm(ep["pregrasp_target_pos"] - ep["final_eef_pos"]))
        if buf_agentview[i]:
            ep["agentview"] = np.stack(buf_agentview[i][:T])
        if buf_eye[i]:
            ep["eye_in_hand"] = np.stack(buf_eye[i][:T])
        results.append(ep)

    return results, {"steps_taken": steps_taken, "batch_successes": batch_successes, "num_envs": int(env.num_envs)}


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


def generate_dataset(
    env: SingleStageCGWrapper,
    *,
    n_success_episodes: int,
    horizon: int,
    hdf5_path: Path,
    planner_params: PickPlaceParams | None = None,
    videos_dir: Path | None = None,
    max_videos: int = 0,
    fps: int = 20,
    logger: logging.Logger | None = None,
) -> dict:
    if videos_dir is not None and max_videos > 0:
        videos_dir.mkdir(parents=True, exist_ok=True)

    stats = GenStats()
    batch = 0
    start = time.time()

    log = (logger.info if logger is not None else print)
    log(f"Target: {n_success_episodes} successful one-stage episodes")
    log(f"HDF5 : {hdf5_path}")
    if max_videos > 0 and videos_dir is not None:
        log(f"Videos: up to {max_videos} -> {videos_dir}")

    while stats.saved < n_success_episodes:
        batch += 1
        log(f"[batch {batch}] start | saved={stats.saved}/{n_success_episodes}")

        episodes, batch_info = run_batch(env, horizon=horizon, planner_params=planner_params)
        stats.attempted += int(batch_info["num_envs"])
        base_attempt_id = stats.attempted - int(batch_info["num_envs"])
        for task in env.tasks:
            stats.task_attempts[str(task)] += 1

        for i, ep in enumerate(episodes):
            if ep is None:
                continue

            if bool(ep.get("success", False)):
                stats.successful += 1
                stats.task_successes[str(ep["task"])] += 1

            if stats.saved >= n_success_episodes:
                continue

            # Only save successful rollouts into HDF5
            if bool(ep.get("success", False)):
                save_episode_to_hdf5(hdf5_path, ep, stats.saved)

            # Save videos for both success and failure by default (bounded by max_videos)
            if videos_dir is not None and stats.videos < max_videos and "agentview" in ep and "eye_in_hand" in ep:
                tag = "success" if bool(ep.get("success", False)) else "fail"
                attempt_id = base_attempt_id + i + 1
                name = f"ep{attempt_id:05d}_{_task_short(ep['task'])}_{tag}.mp4"
                frames = _make_video_frames(ep["agentview"], ep["eye_in_hand"])
                try:
                    _write_video(frames, videos_dir / name, fps=fps)
                    stats.videos += 1
                    log(f"  video -> {name}")
                except Exception as exc:
                    if logger is not None:
                        logger.warning(f"  video failed ({name}): {exc}")

            # Print per-episode line; save counter increments only on success
            tag = "success" if bool(ep.get("success", False)) else "fail"
            log(f"  ep | {tag:7s} | {ep['task']} | T={len(ep['actions'])}")
            if bool(ep.get("success", False)):
                stats.saved += 1

        batch_sr = (int(batch_info["batch_successes"]) / max(int(batch_info["num_envs"]), 1)) * 100.0
        overall_sr = (stats.successful / max(stats.attempted, 1)) * 100.0
        saved_rate = (stats.saved / max(stats.attempted, 1)) * 100.0
        log(
            f"[batch {batch}] done | steps={batch_info['steps_taken']}/{horizon} | "
            f"batch_SR={batch_sr:.0f}% | overall_SR={overall_sr:.1f}% | saved_rate={saved_rate:.1f}% | "
            f"attempted={stats.attempted} success={stats.successful} saved={stats.saved}"
        )

    elapsed = time.time() - start
    return {
        "n_target_success_episodes": int(n_success_episodes),
        "n_attempted_episodes": int(stats.attempted),
        "n_successful_episodes": int(stats.successful),
        "n_saved_episodes": int(stats.saved),
        "rollout_success_rate": stats.successful / max(stats.attempted, 1) * 100.0,
        "saved_success_rate": stats.saved / max(stats.attempted, 1) * 100.0,
        "elapsed_s": elapsed,
        "hdf5_path": str(hdf5_path),
        "n_videos_saved": int(stats.videos),
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
        description="Generate CG_L4 one-stage demos with waypoint motion planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-success", type=int, default=200, help="Successful episodes to collect")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments")
    parser.add_argument("--horizon", type=int, default=220, help="Max steps per episode")
    parser.add_argument("--out-dir", default="results/cg_l4_motion_gen")
    parser.add_argument("--max-videos", type=int, default=20, help="Debug videos to save (0 = none)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)")
    parser.add_argument(
        "--use-orientation-control",
        action="store_true",
        help="Enable OSC orientation deltas. (Grasp yaw alignment can be disabled separately.)",
    )
    parser.add_argument(
        "--no-rotate-for-grasp",
        action="store_true",
        help="Disable rotate_for_grasp + yaw alignment entirely (position-only grasp).",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(args.seed)

    env = SingleStageCGWrapper(
        make_env_fn=lambda: make_env(horizon=args.horizon),
        num_envs=args.num_envs,
        use_relative_coordinates=True,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"gen_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / "demos.hdf5"
    videos_dir = (out_dir / "videos") if args.max_videos > 0 else None

    logger = _setup_logger(out_dir, args.log_level)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    info = generate_dataset(
        env,
        n_success_episodes=args.n_success,
        horizon=args.horizon,
        hdf5_path=hdf5_path,
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
        max_videos=args.max_videos,
        fps=args.fps,
        logger=logger,
    )

    env.close()

    logger.info("=" * 60)
    logger.info("Generation complete")
    logger.info("=" * 60)
    # Keep stdout concise; detailed stats live in gen_info.json (including per-task breakdown)
    for k in [
        "n_target_success_episodes",
        "n_attempted_episodes",
        "n_successful_episodes",
        "n_saved_episodes",
        "rollout_success_rate",
        "saved_success_rate",
        "elapsed_s",
        "hdf5_path",
        "n_videos_saved",
    ]:
        logger.info(f"{k}: {info[k]}")
    logger.info("=" * 60)

    (out_dir / "gen_info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
