#!/usr/bin/env python
"""
Minimal motion-planner smoke test for CG_L4.

Goal:
  Given a target end-effector position, test whether a simple proportional
  controller with a capped step size can move the robot to that target.

This intentionally keeps the logic small and readable:
  1. measure current EEF position
  2. compute target - current
  3. cap that delta by `max_step_size`
  4. convert to the controller action space
  5. repeat until within tolerance or horizon is reached

Examples:
  python repos/robot_envs/tests/test_motion_planner.py
  python repos/robot_envs/tests/test_motion_planner.py --target-mode absolute --target-x 0.10 --target-y 0.00 --target-z 1.05
  python repos/robot_envs/tests/test_motion_planner.py --target-mode object
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CG_ROOT = REPO_ROOT / "CG"
ROBOSUITE_ROOT = CG_ROOT / "robosuite"

sys.path.insert(0, str(CG_ROOT))
sys.path.insert(0, str(ROBOSUITE_ROOT))

from robosuite.controllers import load_composite_controller_config
from robosuite.utils.errors import RandomizationError
from robosuite.utils.placement_samplers import UniformApartRandomSampler


def load_cg_l4_env_class():
    """
    Load the environment class from CG_L4.py.

    Some local versions still export the class as `CG_L2`, so accept either.
    """
    module = importlib.import_module("robosuite.environments.manipulation.CG_L4")
    env_cls = getattr(module, "CG_L4", None)
    if env_cls is None:
        env_cls = getattr(module, "CG_L2", None)
    if env_cls is None:
        raise AttributeError("Could not find CG_L4 or CG_L2 class in CG_L4.py")
    return env_cls


def fix_env_task_pointers(env) -> None:
    """Refresh live object references after reset / task changes."""
    env.object_A = env.object_A_list[env.object_A_index]
    env.object_B = env.object_B_list[env.object_B_index]
    env.object_A_body_id = env.sim.model.body_name2id(env.object_A.root_body)
    env.object_B_body_id = env.sim.model.body_name2id(env.object_B.root_body)


def safe_reset_env(env, *, max_attempts: int = 10):
    last_error = None
    for _ in range(max_attempts):
        try:
            obs = env.reset()
            fix_env_task_pointers(env)
            return obs
        except RandomizationError as exc:
            last_error = exc
    raise last_error if last_error is not None else RuntimeError("env.reset() failed without exception")


def make_env(*, horizon: int, task: str, record_video: bool, video_size: int):
    env_cls = load_cg_l4_env_class()
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

    sampler_a = UniformApartRandomSampler(
        name="FullDesk_Object_A_Sampler",
        x_range=[-0.26, 0.26],
        y_range=[-0.28, -0.05],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        min_distance=0.01,
    )
    sampler_b = UniformApartRandomSampler(
        name="FullDesk_Object_B_Sampler",
        x_range=[-0.26, 0.26],
        y_range=[0.05, 0.28],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0.0, 0.0, 0.8),
        z_offset=0.01,
        min_distance=0.005,
    )

    cam_kwargs = {}
    if record_video:
        cam_kwargs = dict(
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=video_size,
            camera_widths=video_size,
            render_camera=["agentview", "robot0_eye_in_hand"],
        )

    return env_cls(
        robots="Panda",
        controller_configs=controller_config,
        gripper_types="PandaGripper",
        task=task,
        horizon=horizon,
        hard_reset=False,
        placement_initializer_A=sampler_a,
        placement_initializer_B=sampler_b,
        has_renderer=False,
        has_offscreen_renderer=bool(record_video),
        use_camera_obs=bool(record_video),
        **cam_kwargs,
    )


def _make_video_writer(path: Path, fps: int):
    """
    Returns (append_frame_fn, close_fn).
    Writes H.264 MP4 via imageio-ffmpeg so the output plays in VSCode.
    """
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
        state["obj"].append_data(rgb_u8.astype(np.uint8))

    def close():
        if state["obj"] is not None:
            state["obj"].close()

    return append_frame, close


def _extract_video_frame(obs: dict, *, mode: str, flip_ud: bool) -> np.ndarray | None:
    """
    mode: agentview | eye | both
    Returns RGB uint8, or None if images missing.
    """
    av = obs.get("agentview_image", None)
    eye = obs.get("robot0_eye_in_hand_image", None)
    if av is not None:
        av = av.astype(np.uint8)
        if flip_ud:
            av = np.flipud(av)
    if eye is not None:
        eye = eye.astype(np.uint8)
        if flip_ud:
            eye = np.flipud(eye)

    if mode == "agentview":
        return av
    if mode == "eye":
        return eye
    if mode == "both":
        if av is None and eye is None:
            return None
        if av is None:
            return eye
        if eye is None:
            return av
        h = min(av.shape[0], eye.shape[0])
        av = av[:h]
        eye = eye[:h]
        divider = np.zeros((h, 2, 3), dtype=np.uint8)
        divider[:, :, 0] = 255
        return np.concatenate([av, divider, eye], axis=1)
    raise ValueError(f"Unknown video mode: {mode}")


def eef_pos(env) -> np.ndarray:
    return env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].astype(np.float32)


def robot_base_rotmat(env) -> np.ndarray:
    root_body = env.robots[0].robot_model.root_body
    return env.sim.data.get_body_xmat(root_body).reshape(3, 3).astype(np.float32)


def eef_quat_xyzw(env) -> np.ndarray:
    """
    End-effector site quaternion in xyzw.
    """
    from robosuite.utils.transform_utils import mat2quat  # type: ignore

    site_id = env.robots[0].eef_site_id["right"]
    mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
    # robosuite mat2quat returns (w, x, y, z)
    q_wxyz = mat2quat(mat)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def eef_rotmat_world(env) -> np.ndarray:
    site_id = env.robots[0].eef_site_id["right"]
    return env.sim.data.site_xmat[site_id].reshape(3, 3).astype(np.float32)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    from robosuite.utils import transform_utils as T  # type: ignore

    return T.quat2mat(quat_xyzw_to_wxyz(q_xyzw)).astype(np.float32)


def canonicalize_quaternion(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32).copy()
    if float(q[3]) < 0.0:
        q = -q
    return q


def quaternion_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = [float(v) for v in q1_xyzw]
    x2, y2, z2, w2 = [float(v) for v in q2_xyzw]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def quaternion_inverse(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32).copy()
    q[:3] *= -1.0
    return q


def quat_to_rotvec(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Quaternion (xyzw) -> rotation vector (axis * angle), radians.
    """
    q = canonicalize_quaternion(np.asarray(q_xyzw, dtype=np.float32))
    x, y, z, w = [float(v) for v in q]
    v = np.array([x, y, z], dtype=np.float32)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arctan2(v_norm, w))
    axis = v / v_norm
    return (axis * angle).astype(np.float32)


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """
    Rotation vector (axis*angle, radians) -> quaternion (xyzw).
    """
    v = np.asarray(rotvec, dtype=np.float32)
    angle = float(np.linalg.norm(v))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = v / angle
    s = float(np.sin(angle * 0.5))
    c = float(np.cos(angle * 0.5))
    return canonicalize_quaternion(np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32))


@dataclass
class ControllerParams:
    max_step_size: float = 0.02
    pos_tolerance: float = 0.01
    osc_pos_limit: float = 0.05
    use_orientation: bool = False
    rot_tolerance_rad: float = 0.10
    osc_rot_limit_rad: float = 0.50
    max_rot_step_rad: float = 0.15
    ignore_position: bool = False
    # Acceleration limiting: max change in action per step (normalized action space).
    # The raw action from position error is treated as "target velocity",
    # and we accelerate/decelerate toward it smoothly. Set to 0 to disable.
    max_acceleration: float = 0.0
    # When close to target, apply early deceleration to stop smoothly.
    # If enabled, we compute stopping distance and start decelerating early.
    use_smooth_stop: bool = True


class SimplePoseController:
    """
    Translation-only controller:
    move straight toward the target, but cap the commanded delta magnitude.
    With acceleration limiting for smooth start/stop.
    """

    def __init__(self, params: ControllerParams):
        self.params = params
        self._current_velocity: np.ndarray | None = None  # Current action velocity (6-dim: pos+rot)

    def reset(self) -> None:
        """Reset internal state (e.g., when starting a new episode)."""
        self._current_velocity = None

    def compute_action(
        self,
        env,
        *,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        current_pos = eef_pos(env)
        error_world = np.asarray(target_pos, dtype=np.float32) - current_pos
        pos_dist = float(np.linalg.norm(error_world))

        action = np.zeros(7, dtype=np.float32)
        rot_dist = 0.0
        rotvec_b = np.zeros(3, dtype=np.float32)
        if self.params.use_orientation and (target_quat_xyzw is not None):
            from robosuite.utils import transform_utils as T  # type: ignore

            Rb = robot_base_rotmat(env)
            R_cur_w = eef_rotmat_world(env)
            R_tgt_w = quat_xyzw_to_rotmat(canonicalize_quaternion(target_quat_xyzw))
            R_cur_b = (Rb.T @ R_cur_w).astype(np.float32)
            R_tgt_b = (Rb.T @ R_tgt_w).astype(np.float32)
            R_err_b = (R_tgt_b @ R_cur_b.T).astype(np.float32)
            rotvec_b = T.quat2axisangle(T.mat2quat(R_err_b)).astype(np.float32)
            rot_dist = float(np.linalg.norm(rotvec_b))

        eff_pos_dist = 0.0 if bool(self.params.ignore_position) else pos_dist
        if (eff_pos_dist <= self.params.pos_tolerance) and (rot_dist <= self.params.rot_tolerance_rad):
            return action, pos_dist, rot_dist, error_world

        if bool(self.params.ignore_position):
            commanded_world = np.zeros(3, dtype=np.float32)
        else:
            if pos_dist > self.params.max_step_size:
                commanded_world = error_world * (self.params.max_step_size / max(pos_dist, 1e-8))
            else:
                commanded_world = error_world

        commanded_base = robot_base_rotmat(env).T @ commanded_world
        target_action = np.zeros(7, dtype=np.float32)
        target_action[:3] = np.clip(commanded_base / self.params.osc_pos_limit, -1.0, 1.0)
        if self.params.use_orientation and (target_quat_xyzw is not None):
            # Take a bounded step in rotation-vector space to match delta-controller semantics.
            rv = rotvec_b.astype(np.float32)
            rv_norm = float(np.linalg.norm(rv))
            if rv_norm > float(self.params.max_rot_step_rad):
                rv = rv * (float(self.params.max_rot_step_rad) / max(rv_norm, 1e-8))
            target_action[3:6] = np.clip(rv / self.params.osc_rot_limit_rad, -1.0, 1.0)

        # Acceleration limiting: treat target_action as desired velocity,
        # accelerate/decelerate toward it with max_acceleration.
        if self.params.max_acceleration > 0.0:
            if self._current_velocity is None:
                self._current_velocity = np.zeros(6, dtype=np.float32)
            
            target_vel = target_action[:6]
            
            # Smooth stop: when approaching target, compute stopping distance and decelerate early.
            if self.params.use_smooth_stop:
                # Estimate how many steps needed to stop from current velocity.
                current_speed = float(np.linalg.norm(self._current_velocity[:3]))
                if current_speed > 1e-6:
                    # Stopping distance: v^2 / (2*a), in action space
                    stopping_steps = current_speed / (2.0 * self.params.max_acceleration)
                    # Distance to target in normalized action space (approx).
                    # pos_dist is in meters, we normalize by osc_pos_limit to get action-space distance.
                    action_space_dist = pos_dist / self.params.osc_pos_limit
                    
                    # If we're within stopping distance, reduce target velocity proportionally.
                    if action_space_dist < stopping_steps * current_speed:
                        # Deceleration phase: scale down target velocity.
                        scale = max(0.0, action_space_dist / max(stopping_steps * current_speed, 1e-8))
                        target_vel[:3] = target_vel[:3] * scale
            
            # Apply acceleration limit: clamp velocity change per step.
            delta_v = target_vel - self._current_velocity
            delta_v_norm = float(np.linalg.norm(delta_v))
            if delta_v_norm > self.params.max_acceleration:
                delta_v = delta_v * (self.params.max_acceleration / max(delta_v_norm, 1e-8))
            
            self._current_velocity = self._current_velocity + delta_v
            action[:6] = self._current_velocity.copy()
        else:
            # No acceleration limiting: use target action directly.
            action[:6] = target_action[:6]
            if self._current_velocity is not None:
                self._current_velocity = action[:6].copy()
        
        action[6] = target_action[6]  # Gripper command (no smoothing)
        return action, pos_dist, rot_dist, error_world


def resolve_target_position(env, args) -> np.ndarray:
    current = eef_pos(env)

    if args.target_mode == "offset":
        return current + np.array([args.offset_x, args.offset_y, args.offset_z], dtype=np.float32)

    if args.target_mode == "absolute":
        if args.target_x is None or args.target_y is None or args.target_z is None:
            raise ValueError("--target-x/--target-y/--target-z are required for --target-mode absolute")
        return np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)

    if args.target_mode == "object":
        return env.sim.data.body_xpos[env.object_A_body_id].astype(np.float32).copy()

    if args.target_mode == "container":
        return env.sim.data.body_xpos[env.object_B_body_id].astype(np.float32).copy()

    raise ValueError(f"Unsupported target mode: {args.target_mode}")


def resolve_target_quaternion_xyzw(env, args) -> np.ndarray | None:
    if not bool(args.use_orientation):
        return None

    if args.target_quat_mode == "current":
        return eef_quat_xyzw(env).copy()

    if args.target_quat_mode == "absolute":
        if (
            args.target_qx is None
            or args.target_qy is None
            or args.target_qz is None
            or args.target_qw is None
        ):
            raise ValueError("--target-qx/--target-qy/--target-qz/--target-qw are required for --target-quat-mode absolute")
        q = np.array([args.target_qx, args.target_qy, args.target_qz, args.target_qw], dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-8)
        return canonicalize_quaternion(q)

    if args.target_quat_mode == "object":
        q_wxyz = env.sim.data.body_xquat[env.object_A_body_id].astype(np.float32).copy()
        return canonicalize_quaternion(np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32))

    if args.target_quat_mode == "container":
        q_wxyz = env.sim.data.body_xquat[env.object_B_body_id].astype(np.float32).copy()
        return canonicalize_quaternion(np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32))

    raise ValueError(f"Unsupported target quat mode: {args.target_quat_mode}")


def run_motion_test(
    args,
    *,
    out_dir: Path,
    trial_name: str = "trial",
    explicit_target_pos: np.ndarray | None = None,
    explicit_target_quat_xyzw: np.ndarray | None = None,
    target_pos_offset_world: np.ndarray | None = None,
    target_quat_delta_rotvec_local: np.ndarray | None = None,
) -> dict:
    video_path: Path | None = None
    append_frame = None
    close_video = None

    if bool(args.record_video):
        video_path = out_dir / f"{trial_name}.mp4"
        append_frame, close_video = _make_video_writer(video_path, fps=int(args.fps))

    env = make_env(
        horizon=args.horizon,
        task=args.task,
        record_video=bool(args.record_video),
        video_size=int(args.video_size),
    )
    try:
        obs = safe_reset_env(env)
        try:
            obs = env._get_observations()
        except Exception:
            pass

        if append_frame is not None and isinstance(obs, dict):
            frame0 = _extract_video_frame(obs, mode=args.video_mode, flip_ud=bool(args.video_flip_ud))
            if frame0 is not None:
                append_frame(frame0)

        cur_pos0 = eef_pos(env).copy()
        cur_quat0 = canonicalize_quaternion(eef_quat_xyzw(env).copy())

        if explicit_target_pos is not None:
            target_pos = np.asarray(explicit_target_pos, dtype=np.float32)
        elif target_pos_offset_world is not None:
            target_pos = cur_pos0 + np.asarray(target_pos_offset_world, dtype=np.float32)
        else:
            target_pos = resolve_target_position(env, args)

        if explicit_target_quat_xyzw is not None:
            target_quat = canonicalize_quaternion(np.asarray(explicit_target_quat_xyzw, dtype=np.float32))
        elif target_quat_delta_rotvec_local is not None:
            dq = rotvec_to_quat_xyzw(np.asarray(target_quat_delta_rotvec_local, dtype=np.float32))
            target_quat = canonicalize_quaternion(quaternion_multiply(cur_quat0, dq))
        else:
            target_quat = resolve_target_quaternion_xyzw(env, args)
        controller = SimplePoseController(
            ControllerParams(
                max_step_size=args.max_step_size,
                pos_tolerance=args.pos_tolerance,
                osc_pos_limit=args.osc_pos_limit,
                use_orientation=bool(args.use_orientation),
                rot_tolerance_rad=args.rot_tolerance,
                osc_rot_limit_rad=args.osc_rot_limit,
                max_rot_step_rad=args.max_rot_step_rad,
                ignore_position=bool(getattr(args, "ignore_position", False)),
            )
        )

        positions = [eef_pos(env).copy()]
        quats = [eef_quat_xyzw(env).copy()]
        actions = []
        pos_distances = []
        rot_distances = []
        rewards = []
        reached = False

        for step_idx in range(args.horizon):
            action, pos_dist, rot_dist, error_world = controller.compute_action(
                env, target_pos=target_pos, target_quat_xyzw=target_quat
            )
            pos_distances.append(pos_dist)
            rot_distances.append(rot_dist)

            if (pos_dist <= args.pos_tolerance) and (rot_dist <= args.rot_tolerance):
                reached = True
                print(f"Reached target at step {step_idx} | pos={pos_dist:.4f} m | rot={rot_dist:.4f} rad")
                break

            obs, reward, terminated, info = env.step(action)
            _ = info
            actions.append(action.copy())
            rewards.append(float(reward))
            positions.append(eef_pos(env).copy())
            quats.append(eef_quat_xyzw(env).copy())

            if append_frame is not None and isinstance(obs, dict):
                frame = _extract_video_frame(obs, mode=args.video_mode, flip_ud=bool(args.video_flip_ud))
                if frame is not None:
                    append_frame(frame)

            print(
                f"step {step_idx + 1:03d} | "
                f"pos={pos_dist:.4f} m | rot={rot_dist:.4f} rad | "
                f"error_world={np.array2string(error_world, precision=4)} | "
                f"action_xyz={np.array2string(action[:3], precision=4)} | "
                f"action_rot={np.array2string(action[3:6], precision=4)}"
            )

            if bool(terminated):
                print("Environment terminated before reaching the target.")
                break

        final_pos = eef_pos(env).copy()
        final_distance = float(np.linalg.norm(target_pos - final_pos))
        final_rot_dist = 0.0
        final_quat = eef_quat_xyzw(env).copy()
        if bool(args.use_orientation) and (target_quat is not None):
            from robosuite.utils import transform_utils as T  # type: ignore

            Rb = robot_base_rotmat(env)
            R_cur_w = eef_rotmat_world(env)
            R_tgt_w = quat_xyzw_to_rotmat(canonicalize_quaternion(target_quat))
            R_cur_b = (Rb.T @ R_cur_w).astype(np.float32)
            R_tgt_b = (Rb.T @ R_tgt_w).astype(np.float32)
            R_err_b = (R_tgt_b @ R_cur_b.T).astype(np.float32)
            final_rot_dist = float(np.linalg.norm(T.quat2axisangle(T.mat2quat(R_err_b))))

        reached = reached or ((final_distance <= args.pos_tolerance) and (final_rot_dist <= args.rot_tolerance))

        result = {
            "task": args.task,
            "target_mode": args.target_mode,
            "target_pos": target_pos.tolist(),
            "use_orientation": bool(args.use_orientation),
            "target_quat_mode": args.target_quat_mode if bool(args.use_orientation) else None,
            "target_quat_xyzw": (target_quat.tolist() if target_quat is not None else None),
            "target_pos_offset_world": (np.asarray(target_pos_offset_world, dtype=np.float32).tolist() if target_pos_offset_world is not None else None),
            "target_quat_delta_rotvec_local": (np.asarray(target_quat_delta_rotvec_local, dtype=np.float32).tolist() if target_quat_delta_rotvec_local is not None else None),
            "initial_pos": positions[0].tolist(),
            "final_pos": final_pos.tolist(),
            "initial_quat_xyzw": quats[0].tolist(),
            "final_quat_xyzw": final_quat.tolist(),
            "reached_target": bool(reached),
            "final_distance": final_distance,
            "final_rot_distance_rad": final_rot_dist,
            "steps_executed": len(actions),
            "controller": asdict(
                ControllerParams(
                    max_step_size=args.max_step_size,
                    pos_tolerance=args.pos_tolerance,
                    osc_pos_limit=args.osc_pos_limit,
                    use_orientation=bool(args.use_orientation),
                    rot_tolerance_rad=args.rot_tolerance,
                    osc_rot_limit_rad=args.osc_rot_limit,
                )
            ),
            "path_positions": np.asarray(positions, dtype=np.float32).tolist(),
            "path_quat_xyzw": np.asarray(quats, dtype=np.float32).tolist(),
            "actions": np.asarray(actions, dtype=np.float32).tolist(),
            "pos_distances": [float(d) for d in pos_distances],
            "rot_distances_rad": [float(d) for d in rot_distances],
            "rewards": rewards,
            "video_path": (str(video_path) if video_path is not None else None),
        }
        return result
    finally:
        if close_video is not None:
            close_video()
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Simple CG_L4 target-pose motion test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default="place the cross into the bin")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--target-mode", choices=["offset", "absolute", "object", "container"], default="offset")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=None)
    parser.add_argument("--offset-x", type=float, default=0.08)
    parser.add_argument("--offset-y", type=float, default=0.00)
    parser.add_argument("--offset-z", type=float, default=0.00)
    parser.add_argument("--max-step-size", type=float, default=0.02, help="Max commanded translation per control step (m)")
    parser.add_argument("--pos-tolerance", type=float, default=0.01, help="Success threshold on EEF-to-target distance (m)")
    parser.add_argument("--osc-pos-limit", type=float, default=0.05, help="Controller translation scale for BASIC / OSC action normalization")
    parser.add_argument("--use-orientation", action="store_true", help="Also track a target orientation (quat) via OSC rotation deltas")
    parser.add_argument("--target-quat-mode", choices=["current", "absolute", "object", "container"], default="current")
    parser.add_argument("--target-qx", type=float, default=None)
    parser.add_argument("--target-qy", type=float, default=None)
    parser.add_argument("--target-qz", type=float, default=None)
    parser.add_argument("--target-qw", type=float, default=None)
    parser.add_argument("--rot-tolerance", type=float, default=0.10, help="Success threshold on orientation error (radians)")
    parser.add_argument("--osc-rot-limit", type=float, default=0.50, help="Controller rotation scale for BASIC / OSC normalization (radians)")
    parser.add_argument("--max-rot-step-rad", type=float, default=0.15, help="Max commanded rotation per control step (radians)")
    parser.add_argument("--record-video", action="store_true", help="Record an MP4 (requires offscreen rendering)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-size", type=int, default=256)
    parser.add_argument("--video-mode", choices=["agentview", "eye", "both"], default="both")
    parser.add_argument("--video-flip-ud", action="store_true", help="Flip images vertically (MuJoCo camera convention)")
    parser.add_argument("--ignore-position", action="store_true", help="For orientation tests: command rotation only and ignore position in success check")
    parser.add_argument("--batch-random", action="store_true", help="Run multiple random trials (pos / rot / pos+rot)")
    parser.add_argument("--n-random-pos", type=int, default=5)
    parser.add_argument("--n-random-rot", type=int, default=5)
    parser.add_argument("--n-random-both", type=int, default=5)
    parser.add_argument("--random-pos-max", type=float, default=0.18, help="Max abs offset (m) for random position targets (x,y)")
    parser.add_argument("--random-pos-z-max", type=float, default=0.10, help="Max abs offset (m) for random position targets (z)")
    parser.add_argument("--random-rot-max-rad", type=float, default=0.8, help="Max rotation angle (rad) for random orientation targets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(args.seed)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"motion_test_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bool(args.batch_random):
        result = run_motion_test(args, out_dir=out_dir, trial_name="rollout")
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2) + "\n")

        print("=" * 60)
        print(f"Reached target : {result['reached_target']}")
        print(f"Initial pos    : {np.array(result['initial_pos'])}")
        print(f"Target pos     : {np.array(result['target_pos'])}")
        print(f"Final pos      : {np.array(result['final_pos'])}")
        print(f"Final distance : {result['final_distance']:.4f} m")
        print(f"Steps executed : {result['steps_executed']}")
        print(f"Summary        : {summary_path}")
        print("=" * 60)
        return

    rng = np.random.RandomState(int(args.seed))

    def _sample_pos_offset() -> np.ndarray:
        dx = rng.uniform(-float(args.random_pos_max), float(args.random_pos_max))
        dy = rng.uniform(-float(args.random_pos_max), float(args.random_pos_max))
        dz = rng.uniform(-float(args.random_pos_z_max), float(args.random_pos_z_max))
        return np.array([dx, dy, dz], dtype=np.float32)

    def _sample_rotvec_delta() -> np.ndarray:
        # axis uniformly random, angle bounded (local frame)
        axis = rng.normal(size=(3,)).astype(np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        angle = rng.uniform(-float(args.random_rot_max_rad), float(args.random_rot_max_rad))
        return (axis * float(angle)).astype(np.float32)

    all_results: list[dict] = []

    # --- Position-only trials ---
    for i in range(int(args.n_random_pos)):
        # disable orientation for pos-only
        args.use_orientation = False
        trial_dir = out_dir / f"pos_{i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        pos_off = _sample_pos_offset()

        res = run_motion_test(
            args,
            out_dir=trial_dir,
            trial_name="rollout",
            explicit_target_pos=None,
            explicit_target_quat_xyzw=None,
            target_pos_offset_world=pos_off,
            target_quat_delta_rotvec_local=None,
        )
        (trial_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        all_results.append(res)

    # --- Orientation-only trials ---
    for i in range(int(args.n_random_rot)):
        args.use_orientation = True
        args.ignore_position = True
        trial_dir = out_dir / f"rot_{i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        rot_d = _sample_rotvec_delta()

        res = run_motion_test(
            args,
            out_dir=trial_dir,
            trial_name="rollout",
            explicit_target_pos=None,
            explicit_target_quat_xyzw=None,
            target_pos_offset_world=np.zeros(3, dtype=np.float32),
            target_quat_delta_rotvec_local=rot_d,
        )
        (trial_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        all_results.append(res)

    # --- Position + orientation trials ---
    for i in range(int(args.n_random_both)):
        args.use_orientation = True
        args.ignore_position = False
        trial_dir = out_dir / f"both_{i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        pos_off = _sample_pos_offset()
        rot_d = _sample_rotvec_delta()

        res = run_motion_test(
            args,
            out_dir=trial_dir,
            trial_name="rollout",
            explicit_target_pos=None,
            explicit_target_quat_xyzw=None,
            target_pos_offset_world=pos_off,
            target_quat_delta_rotvec_local=rot_d,
        )
        (trial_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
        all_results.append(res)

    batch_summary = {
        "seed": int(args.seed),
        "n_pos": int(args.n_random_pos),
        "n_rot": int(args.n_random_rot),
        "n_both": int(args.n_random_both),
        "success_rate": float(np.mean([1.0 if r.get("reached_target", False) else 0.0 for r in all_results])) if all_results else 0.0,
        "results": all_results,
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2) + "\n")
    print(f"Wrote batch summary: {out_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
