import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from robosuite.environments.manipulation.CG_L2 import CG_L2
from CG_L2_image_wrapper import ImageBasedCGWrapper


def make_env():
    return CG_L2(
        robots="Panda",
        gripper_types="PandaGripper",
        strategy="fixed",
        task="place the cross into the bin",
        horizon=200,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        render_camera="agentview",
    )


def _make_video_writer(path: Path, fps: int):
    """
    Returns (append_frame_fn, close_fn).
    Writes H.264 MP4 via ffmpeg (imageio backend) so the output plays in VSCode
    (Chromium/Electron requires H.264 + yuv420p).
    """
    import imageio.v2 as iio  # type: ignore

    writer_state: dict = {"obj": None}

    def append_frame(rgb_u8: np.ndarray):
        if writer_state["obj"] is None:
            writer_state["obj"] = iio.get_writer(
                str(path),
                fps=fps,
                codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-crf", "18"],
            )
        writer_state["obj"].append_data(rgb_u8)  # imageio expects RGB

    def close():
        if writer_state["obj"] is not None:
            writer_state["obj"].close()

    return append_frame, close


def main(steps: int, out_dir: Path, fps: int, seed: int):
    # Wrapper constraints:
    # - num_envs must be even
    # - env.train_task must be set before reset()
    np.random.seed(seed)

    env = ImageBasedCGWrapper(make_env_fn=make_env, num_envs=2, use_relative_coordinates=False)
    env.train_task = "all"

    obs, _info = env.reset()

    required = {
        "observation.state",
        "observation.environment_state",
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    }
    missing = required.difference(obs.keys())
    if missing:
        raise RuntimeError(f"Missing obs keys: {sorted(missing)}")

    print("obs keys:", sorted(obs.keys()))
    print("state shape:", obs["observation.state"].shape)
    print("env_state shape:", obs["observation.environment_state"].shape)
    print("agentview shape:", obs["observation.images.agentview"].shape)
    print("eye_in_hand shape:", obs["observation.images.robot0_eye_in_hand"].shape)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out_dir / f"cg_l2_image_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / "agentview.mp4"
    append_frame, close_video = _make_video_writer(video_path, fps=fps)

    actions_log = []
    rewards_log = []
    success_log = []
    eef_pos_log = []

    prev_eef = np.stack([env.obs[i]["robot0_eef_pos"] for i in range(env.num_envs)], axis=0)
    prev_img = obs["observation.images.agentview"].copy()
    prev_grip = np.stack([env.obs[i]["robot0_gripper_qpos"] for i in range(env.num_envs)], axis=0)

    eef_delta_norms = []
    img_delta_means = []
    gripper_delta_means = []

    # Record initial frame (env 0)
    append_frame(obs["observation.images.agentview"][0].astype(np.uint8))
    eef_pos_log.append(prev_eef.copy())

    def record_step(step_actions: np.ndarray, step_obs, step_rew, step_info, step_idx: int):
        nonlocal prev_eef, prev_img, prev_grip

        actions_log.append(step_actions.copy())
        rewards_log.append(np.array(step_rew).copy())
        success_log.append(np.array(step_info.get("is_success", [False] * env.num_envs)).copy())

        cur_eef = np.stack([env.obs[i]["robot0_eef_pos"] for i in range(env.num_envs)], axis=0)
        cur_grip = np.stack([env.obs[i]["robot0_gripper_qpos"] for i in range(env.num_envs)], axis=0)
        eef_pos_log.append(cur_eef.copy())

        eef_delta = np.linalg.norm(cur_eef - prev_eef, axis=1)
        eef_delta_norms.append(float(np.mean(eef_delta)))
        prev_eef = cur_eef

        cur_img = step_obs["observation.images.agentview"]
        img_delta = float(np.mean(np.abs(cur_img.astype(np.float32) - prev_img.astype(np.float32))))
        img_delta_means.append(img_delta)
        prev_img = cur_img.copy()

        grip_delta = float(np.mean(np.abs(cur_grip.astype(np.float32) - prev_grip.astype(np.float32))))
        gripper_delta_means.append(grip_delta)
        prev_grip = cur_grip

        append_frame(cur_img[0].astype(np.uint8))

        if (step_idx + 1) % 20 == 0:
            print(
                f"step {step_idx+1:04d} | mean_eef_delta={eef_delta_norms[-1]:.6f} | "
                f"mean_img_delta={img_delta_means[-1]:.3f} | mean_gripper_delta={gripper_delta_means[-1]:.6f}"
            )

    # Scripted sequence: +x 1s, +y 1s, +z 1s, close 1s, open 1s, rotate 1s
    control_hz = int(getattr(env.envs[0], "control_freq", 20))
    seg_steps = control_hz  # 1 second
    step_idx = 0

    def run_segment(name: str, action_7: np.ndarray, n_steps: int):
        nonlocal step_idx
        print(f"segment: {name} ({n_steps} steps @ {control_hz} Hz)")
        for _ in range(n_steps):
            actions = np.repeat(action_7[None, :], env.num_envs, axis=0).astype(np.float32)
            obs_seg, rew_seg, _term, _trunc, info_seg = env.step(actions)
            record_step(actions, obs_seg, rew_seg, info_seg, step_idx=step_idx)
            step_idx += 1

    dx = 0.05
    drot = 0.3

    run_segment("move +x", np.array([dx, 0, 0, 0, 0, 0, 0], dtype=np.float32), seg_steps)
    run_segment("move +y", np.array([0, dx, 0, 0, 0, 0, 0], dtype=np.float32), seg_steps)
    run_segment("move +z", np.array([0, 0, dx, 0, 0, 0, 0], dtype=np.float32), seg_steps)
    run_segment("close gripper", np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32), seg_steps)
    run_segment("open gripper", np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), seg_steps)
    run_segment("rotate (z axis)", np.array([0, 0, 0, 0, 0, drot, 0], dtype=np.float32), seg_steps)

    # Optional random phase (keeps old --steps behavior as "extra" steps)
    if steps > 0:
        print(f"segment: random ({steps} steps)")
        for _ in range(steps):
            actions = np.zeros((env.num_envs, 7), dtype=np.float32)
            actions[:, :3] = np.random.uniform(-0.05, 0.05, size=(env.num_envs, 3)).astype(np.float32)
            actions[:, 3:6] = np.random.uniform(-0.3, 0.3, size=(env.num_envs, 3)).astype(np.float32)
            actions[:, 6] = np.random.uniform(-1.0, 1.0, size=(env.num_envs,)).astype(np.float32)
            obs_r, rew_r, _term, _trunc, info_r = env.step(actions)
            record_step(actions, obs_r, rew_r, info_r, step_idx=step_idx)
            step_idx += 1

    print("done.")
    print(f"avg mean_eef_delta: {float(np.mean(eef_delta_norms)):.6f}")
    print(f"avg mean_img_delta: {float(np.mean(img_delta_means)):.3f}")
    print(f"avg mean_gripper_delta: {float(np.mean(gripper_delta_means)):.6f}")

    close_video()

    actions_arr = np.stack(actions_log, axis=0) if actions_log else np.zeros((0, env.num_envs, 7), dtype=np.float32)
    rewards_arr = np.stack(rewards_log, axis=0) if rewards_log else np.zeros((0, env.num_envs), dtype=np.float32)
    success_arr = np.stack(success_log, axis=0) if success_log else np.zeros((0, env.num_envs), dtype=bool)
    eef_pos_arr = np.stack(eef_pos_log, axis=0) if eef_pos_log else np.zeros((0, env.num_envs, 3), dtype=np.float32)

    np.savez_compressed(
        out_dir / "rollout.npz",
        actions=actions_arr,
        rewards=rewards_arr,
        is_success=success_arr,
        eef_pos=eef_pos_arr,
    )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "video": str(video_path),
                "npz": str(out_dir / "rollout.npz"),
                "num_envs": env.num_envs,
                "steps_requested": int(steps),
                "steps_recorded": int(actions_arr.shape[0]),
                "fps": int(fps),
                "seed": int(seed),
                "avg_mean_eef_delta": float(np.mean(eef_delta_norms)) if eef_delta_norms else 0.0,
                "avg_mean_img_delta": float(np.mean(img_delta_means)) if img_delta_means else 0.0,
                "avg_mean_gripper_delta": float(np.mean(gripper_delta_means)) if gripper_delta_means else 0.0,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"wrote video: {video_path}")
    print(f"wrote rollout: {out_dir / 'rollout.npz'}")
    print(f"wrote summary: {out_dir / 'summary.json'}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Common default for headless rendering; override as needed.
    os.environ.setdefault("MUJOCO_GL", "egl")

    main(args.steps, out_dir=Path(args.out_dir), fps=args.fps, seed=args.seed)

