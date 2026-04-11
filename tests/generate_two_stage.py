#!/usr/bin/env python
"""
Two-stage dataset generation for CG_L2.

Episode structure (three phases, recorded flat):
  Phase 0  STAGE1 : policy runs task (obj_A → cont_B)
  Phase 1  HOMING : EEF returns to resting position (optional plug-in)
  Phase 2  STAGE2 : policy runs task (obj_C → cont_D)

Recording is completely agnostic to phases — it just appends (obs, action)
every step.  Phase boundaries are stored as step indices in the HDF5 attrs.

HDF5 output layout:
  demo_{N}/
    obs/
      robot0_eef_pos           (T, 3)
      robot0_eef_quat          (T, 4)
      robot0_gripper_qpos      (T, 2)
      object                   (T, 42)  world-frame poses (6×7D)
      agentview_image          (T, H, W, 3)  uint8
      robot0_eye_in_hand_image (T, H, W, 3)  uint8
    actions                    (T, 7)
    attrs:
      stage1_task, stage2_task
      stage1_end   : step index where phase 0 ends (= first homing step)
      homing_end   : step index where phase 1 ends (= first stage-2 step)
                     equals stage1_end when homing is disabled
      success = True

Video output:
  videos/ep{N:03d}_{s1}__then__{s2}.mp4
  Side-by-side agentview | eye-in-hand.  Colour bars mark phase boundaries:
    green → start of homing (or stage-2 if homing disabled)
    blue  → start of stage-2 policy (only shown when homing > 0 steps)

Usage:
  python generate_two_stage.py \\
      --checkpoint /path/to/checkpoint.pt \\
      --n-success 500 \\
      --num-envs 4 \\
      --horizon 400 \\
      --max-videos 20 \\
      --out-dir results/two_stage_gen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CG"))

from CG_L2_two_stage_wrapper import TwoStageCGWrapper


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

STAGE1 = "stage1"
HOMING = "homing"
STAGE2 = "stage2"


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _write_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    """Write RGB uint8 (H, W, 3) frames to H.264 MP4."""
    import imageio.v2 as iio  # type: ignore
    writer = iio.get_writer(
        str(path), fps=fps, codec="libx264",
        output_params=["-pix_fmt:v", "yuv420p", "-crf", "18"],
    )
    for f in frames:
        writer.append_data(f)
    writer.close()


def _make_video_frames(
    agentview: np.ndarray,      # (T, H, W, 3) uint8 – already correctly oriented
    eye_in_hand: np.ndarray,    # (T, H, W, 3) uint8 – already correctly oriented
    stage1_end: int,            # step index where stage-1 ends
    homing_end: int,            # step index where homing ends (= stage1_end if disabled)
) -> list[np.ndarray]:
    """
    Side-by-side compositing with phase-boundary annotations.
    Colour bars at top of frame (4 px):
      green → stage1_end  (start of homing, or stage-2 if homing disabled)
      blue  → homing_end  (start of stage-2 policy; only drawn if homing_end > stage1_end)
    """
    divider = np.zeros((agentview.shape[1], 2, 3), dtype=np.uint8)
    divider[:, :, 0] = 255  # red vertical line

    out = []
    for t in range(len(agentview)):
        frame = np.concatenate([agentview[t], divider, eye_in_hand[t]], axis=1)
        if t == stage1_end:
            frame[:4, :, :] = 0;  frame[:4, :, 1] = 255   # green
        if homing_end > stage1_end and t == homing_end:
            frame[:4, :, :] = 0;  frame[:4, :, 2] = 255   # blue
        out.append(frame)
    return out


def _task_short(task: str) -> str:
    """'place the cross into the bin' → 'cross_bin'"""
    words = task.lower().split()
    obj  = next((w for w in words if w in ("cross", "cube", "cylinder")), "obj")
    cont = next((w for w in words if w in ("bin", "cup", "plate")),       "cont")
    return f"{obj}_{cont}"


# ---------------------------------------------------------------------------
# Environment and policy factories
# ---------------------------------------------------------------------------

def make_env(*, horizon: int):
    """CG_L2 with both cameras enabled."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CG" / "robosuite"))
    from robosuite.environments.manipulation.CG_L2 import CG_L2
    return CG_L2(
        robots="Panda",
        gripper_types="PandaGripper",
        strategy="fixed",
        task="place the cross into the bin",  # overridden at every reset
        horizon=horizon,
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
        raise FileNotFoundError(checkpoint)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "models" / "src"))
    from policies.training.policy_loading import load_robot_flow_policy  # type: ignore
    policy = load_robot_flow_policy(checkpoint, device=device)
    policy.model.eval()
    return policy


# ---------------------------------------------------------------------------
# HDF5 saving
# ---------------------------------------------------------------------------

def save_episode_to_hdf5(hdf5_path: Path, ep: dict, ep_id: int) -> None:
    with h5py.File(hdf5_path, "a") as f:
        grp     = f.require_group(f"demo_{ep_id}")
        obs_grp = grp.require_group("obs")

        obs_grp.create_dataset("robot0_eef_pos",      data=ep["eef_pos"],   compression="gzip")
        obs_grp.create_dataset("robot0_eef_quat",     data=ep["eef_quat"],  compression="gzip")
        obs_grp.create_dataset("robot0_gripper_qpos", data=ep["gripper_q"], compression="gzip")
        obs_grp.create_dataset("object",              data=ep["obj_world"], compression="gzip")

        if "agentview" in ep:
            obs_grp.create_dataset("agentview_image", data=ep["agentview"],
                                   compression="gzip", chunks=(1, *ep["agentview"].shape[1:]))
        if "eye_in_hand" in ep:
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=ep["eye_in_hand"],
                                   compression="gzip", chunks=(1, *ep["eye_in_hand"].shape[1:]))

        grp.create_dataset("actions", data=ep["actions"], compression="gzip")

        grp.attrs["stage1_task"] = ep["stage1_task"]
        grp.attrs["stage2_task"] = ep["stage2_task"]
        grp.attrs["stage1_end"]  = ep["stage1_end"]   # first homing step
        grp.attrs["homing_end"]  = ep["homing_end"]   # first stage-2 step
        grp.attrs["success"]     = True


# ---------------------------------------------------------------------------
# Homing controller (plug-in between stages)
# ---------------------------------------------------------------------------

class HomeController:
    """
    Proportional controller that drives the EEF back to the resting home
    position between stage 1 and stage 2.

    Action convention (OSC_POSE, 7-D):
      [0:3]  delta EEF position  — sent at full scale so the OSC controller
                                   clips them to its internal limits.  This
                                   gives maximum homing speed.
      [3:6]  delta orientation   — zero (don't perturb orientation)
      [6]    gripper             — open (-1)

    Parameters
    ----------
    enabled    : plug-in switch; False → completely skipped (no-op)
    home_pos   : target EEF position in world frame
    pos_gain   : proportional gain; set high enough to always saturate max_delta
    max_delta  : per-axis action cap; should be close to 1.0 so the OSC
                 controller runs at maximum speed (it clips internally)
    threshold  : distance (m) at which homing is considered complete
    max_steps  : hard timeout (steps at control_freq Hz)
    """

    _DEFAULT_HOME = np.array([-0.11821743, -0.00377651, 0.9975709], dtype=np.float32)

    def __init__(
        self,
        enabled:    bool             = True,
        home_pos:   np.ndarray|None  = None,
        pos_gain:   float            = 20.0,   # always saturates max_delta
        max_delta:  float            = 0.9,    # ~max OSC speed (clipped internally)
        threshold:  float            = 0.05,
        max_steps:  int              = 60,
    ):
        self.enabled   = enabled
        self.home_pos  = (
            np.array(home_pos, dtype=np.float32)
            if home_pos is not None else self._DEFAULT_HOME.copy()
        )
        self.pos_gain  = pos_gain
        self.max_delta = max_delta
        self.threshold = threshold
        self.max_steps = max_steps
        self._steps: dict[int, int] = {}

    def trigger(self, env_idx: int) -> None:
        """Activate homing for env_idx (called on stage-1 completion)."""
        if self.enabled:
            self._steps[env_idx] = 0

    def is_active(self, env_idx: int) -> bool:
        return env_idx in self._steps

    def get_action(self, env_idx: int, eef_pos: np.ndarray) -> np.ndarray | None:
        """
        Return a 7-D homing action, or None when done.
        None removes env_idx from the active set.
        """
        delta = self.home_pos - np.asarray(eef_pos, dtype=np.float32)
        dist  = float(np.linalg.norm(delta))
        self._steps[env_idx] += 1

        if dist < self.threshold or self._steps[env_idx] >= self.max_steps:
            del self._steps[env_idx]
            return None

        action        = np.zeros(7, dtype=np.float32)
        action[:3]    = np.clip(delta * self.pos_gain, -self.max_delta, self.max_delta)
        action[6]     = -1.0   # open gripper
        return action

    def reset(self) -> None:
        self._steps.clear()


# ---------------------------------------------------------------------------
# Policy query helper
# ---------------------------------------------------------------------------

def _query_policy(policy, state_hists, env_state_hists, task_idx_tensors,
                  need_query, n_execute, device, use_env_state):
    """Batch-query the policy for `need_query` envs and fill their action queues."""
    import torch

    B = len(need_query)
    state_batch = torch.from_numpy(
        np.stack([np.stack(list(state_hists[i])) for i in need_query])
    ).to(torch.float32).to(device)

    raw = {"state": state_batch}
    if use_env_state:
        raw["env_state"] = torch.from_numpy(
            np.stack([np.stack(list(env_state_hists[i])) for i in need_query])
        ).to(torch.float32).to(device)

    task_batch = torch.cat([task_idx_tensors[i] for i in need_query], dim=0)
    norm       = policy.processor(raw)

    kwargs = {
        "robot_state":  norm["state"],
        "task_indices": task_batch,
        "flow_algo":    policy.model.flow_algo,
        "batch_size":   B,
    }
    if use_env_state:
        kwargs["env_state"] = norm["env_state"]

    with torch.no_grad():
        acts = policy.denormalize_actions(policy.model.generate_actions(**kwargs))

    acts_np = acts[:, :n_execute].detach().cpu().numpy()   # (B, n_execute, 7)
    return {i: [acts_np[qi, t, :7].astype(np.float32) for t in range(acts_np.shape[1])]
            for qi, i in enumerate(need_query)}


# ---------------------------------------------------------------------------
# Single batched rollout
# ---------------------------------------------------------------------------

def run_batch(
    env: TwoStageCGWrapper,
    policy,
    *,
    horizon:    int,
    n_execute:  int,
    device:     str,
    home_ctrl:  HomeController | None = None,
) -> list[dict | None]:
    """
    Run one batched rollout covering both stages (+ optional homing).

    Per-env phase machine:
      STAGE1 → (stage-1 success) → HOMING → (home reached) → STAGE2 → done

    Recording is flat and phase-agnostic: every active step appends to the
    obs/action buffers.  Phase boundaries are recorded as step indices.

    Returns: list of length num_envs — dict (success) or None (failed/timeout).
    """
    import torch

    num_envs      = env.num_envs
    obs, _        = env.reset()
    n_obs_steps   = int(policy.state_feature.window_size)
    use_env_state = policy.env_state_feature is not None

    if home_ctrl is not None:
        home_ctrl.reset()

    # ---- per-env state ----
    phase         = [STAGE1] * num_envs
    action_queues: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    done          = [False]  * num_envs
    ep_success    = [False]  * num_envs
    stage1_end    = [-1]     * num_envs   # step index: last STAGE1 step + 1
    homing_end    = [-1]     * num_envs   # step index: last HOMING step + 1

    # ---- task-index tensors (stage1 first, updated to stage2 on transition) ----
    def _task_idx(env_i: int, stg: str) -> torch.Tensor:
        if stg == STAGE1:
            oi, ci = env.stage1_obj_idx[env_i], env.stage1_cont_idx[env_i]
        else:
            oi, ci = env.stage2_obj_idx[env_i], env.stage2_cont_idx[env_i]
        return torch.tensor([[oi, ci]], dtype=torch.long, device=device)

    task_idx = [_task_idx(i, STAGE1) for i in range(num_envs)]

    # ---- obs-history deques (fed to policy) ----
    state_hists     = [deque(maxlen=n_obs_steps) for _ in range(num_envs)]
    env_state_hists = [deque(maxlen=n_obs_steps) for _ in range(num_envs)]

    def _seed(i: int):
        state_hists[i].append(obs["observation.state"][i].astype(np.float32))
        if use_env_state:
            env_state_hists[i].append(obs["observation.environment_state"][i].astype(np.float32))
        while len(state_hists[i]) < n_obs_steps:
            state_hists[i].append(state_hists[i][-1].copy())
            if use_env_state:
                env_state_hists[i].append(env_state_hists[i][-1].copy())

    for i in range(num_envs):
        _seed(i)

    # ---- flat trajectory buffers (recording, phase-agnostic) ----
    buf_eef_pos   = [[] for _ in range(num_envs)]
    buf_eef_quat  = [[] for _ in range(num_envs)]
    buf_gripper_q = [[] for _ in range(num_envs)]
    buf_obj_world = [[] for _ in range(num_envs)]
    buf_agentview = [[] for _ in range(num_envs)]
    buf_eye       = [[] for _ in range(num_envs)]
    buf_actions   = [[] for _ in range(num_envs)]

    # ====================================================================
    # Main loop
    # ====================================================================
    for step in range(horizon):
        if all(done):
            break

        # ----------------------------------------------------------------
        # 1. RECORD current observation (flat, no phase awareness)
        # ----------------------------------------------------------------
        for i in range(num_envs):
            if done[i]:
                continue
            st = obs["observation.state"][i]
            buf_eef_pos[i].append(st[:3].copy())
            buf_eef_quat[i].append(st[3:7].copy())
            buf_gripper_q[i].append(st[7:9].copy())
            buf_obj_world[i].append(obs["observation.object_world"][i].copy())
            if "observation.images.agentview" in obs:
                buf_agentview[i].append(obs["observation.images.agentview"][i].copy())
            if "observation.images.robot0_eye_in_hand" in obs:
                buf_eye[i].append(obs["observation.images.robot0_eye_in_hand"][i].copy())

        # ----------------------------------------------------------------
        # 2. UPDATE obs histories for policy
        # ----------------------------------------------------------------
        for i in range(num_envs):
            if done[i]:
                continue
            state_hists[i].append(obs["observation.state"][i].astype(np.float32))
            if use_env_state:
                env_state_hists[i].append(obs["observation.environment_state"][i].astype(np.float32))

        # ----------------------------------------------------------------
        # 3. QUERY POLICY for STAGE1/STAGE2 envs that need new actions
        # ----------------------------------------------------------------
        need_query = [
            i for i in range(num_envs)
            if not done[i] and phase[i] != HOMING and not action_queues[i]
        ]
        if need_query:
            new_acts = _query_policy(
                policy, state_hists, env_state_hists, task_idx,
                need_query, n_execute, device, use_env_state,
            )
            for i, acts in new_acts.items():
                action_queues[i].extend(acts)

        # ----------------------------------------------------------------
        # 4. BUILD action matrix
        #    STAGE1/STAGE2 → pop from policy queue
        #    HOMING        → P-controller toward home position
        # ----------------------------------------------------------------
        action_mat = np.zeros((num_envs, 7), dtype=np.float32)
        for i in range(num_envs):
            if done[i]:
                continue
            if phase[i] == HOMING:
                eef_pos  = obs["observation.state"][i, :3]
                h_action = home_ctrl.get_action(i, eef_pos)  # None when done
                if h_action is not None:
                    action_mat[i] = h_action
                # else: homing just finished this step; zero action is fine
            elif action_queues[i]:
                action_mat[i] = action_queues[i].pop(0)

        # ----------------------------------------------------------------
        # 5. RECORD action (flat)
        # ----------------------------------------------------------------
        for i in range(num_envs):
            if not done[i]:
                buf_actions[i].append(action_mat[i].copy())

        # ----------------------------------------------------------------
        # 6. STEP environment
        # ----------------------------------------------------------------
        obs, _rew, terminated, truncated, info = env.step(action_mat)

        # ----------------------------------------------------------------
        # 7. UPDATE phase state machine
        # ----------------------------------------------------------------
        for i in range(num_envs):
            if done[i]:
                continue

            if phase[i] == STAGE1 and env.current_stage[i] == 1:
                # Stage-1 just completed
                stage1_end[i] = step + 1           # exclusive end index
                action_queues[i].clear()
                task_idx[i] = _task_idx(i, STAGE2)
                if home_ctrl is not None and home_ctrl.enabled:
                    phase[i] = HOMING
                    home_ctrl.trigger(i)
                else:
                    phase[i]      = STAGE2
                    homing_end[i] = stage1_end[i]  # no homing → boundaries coincide

            elif phase[i] == HOMING and not home_ctrl.is_active(i):
                # Homing just finished (get_action returned None this step)
                phase[i]      = STAGE2
                homing_end[i] = step + 1

            # Check overall success (stage-2 done)
            if phase[i] == STAGE2 and info["is_success"][i]:
                ep_success[i] = True
                done[i]       = True

            if bool(terminated[i]) or bool(truncated[i]):
                done[i] = True

        # Progress print
        sr = sum(ep_success) / num_envs * 100
        print(f"\r  step {step+1:4d}/{horizon} | phases {[p[:2] for p in phase]} | SR={sr:.0f}%",
              end="", flush=True)

    print()

    # ----------------------------------------------------------------
    # 8. PACKAGE results
    # ----------------------------------------------------------------
    results: list[dict | None] = []
    for i in range(num_envs):
        if not ep_success[i] or not buf_actions[i]:
            results.append(None)
            continue
        T  = len(buf_actions[i])
        ep = {
            "eef_pos":    np.stack(buf_eef_pos[i][:T]),
            "eef_quat":   np.stack(buf_eef_quat[i][:T]),
            "gripper_q":  np.stack(buf_gripper_q[i][:T]),
            "obj_world":  np.stack(buf_obj_world[i][:T]),
            "actions":    np.stack(buf_actions[i]),
            "stage1_task": env.stage1_task[i],
            "stage2_task": env.stage2_task[i],
            "stage1_end":  stage1_end[i],
            "homing_end":  homing_end[i],
        }
        if buf_agentview[i]:
            ep["agentview"]   = np.stack(buf_agentview[i][:T])
        if buf_eye[i]:
            ep["eye_in_hand"] = np.stack(buf_eye[i][:T])
        results.append(ep)

    return results


# ---------------------------------------------------------------------------
# Dataset generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    env: TwoStageCGWrapper,
    policy,
    *,
    n_success_episodes: int,
    horizon:    int,
    n_execute:  int,
    device:     str,
    hdf5_path:  Path,
    videos_dir: Path | None = None,
    max_videos: int  = 0,
    fps:        int  = 20,
    use_homing: bool = True,
) -> dict:
    home_ctrl = HomeController(enabled=use_homing)

    if videos_dir is not None and max_videos > 0:
        videos_dir.mkdir(parents=True, exist_ok=True)

    n_saved, n_total, n_vids = 0, 0, 0
    start = time.time()
    batch = 0

    print(f"Target: {n_success_episodes} successful two-stage episodes")
    print(f"HDF5 : {hdf5_path}")
    if max_videos > 0:
        print(f"Videos: up to {max_videos} → {videos_dir}")

    while n_saved < n_success_episodes:
        print(f"\n[batch {batch+1}]  saved {n_saved}/{n_success_episodes}")
        episodes = run_batch(env, policy, horizon=horizon, n_execute=n_execute,
                             device=device, home_ctrl=home_ctrl)
        batch += 1

        for ep in episodes:
            n_total += 1
            if ep is None or n_saved >= n_success_episodes:
                continue

            save_episode_to_hdf5(hdf5_path, ep, n_saved)

            # Video for first max_videos successful episodes
            if videos_dir is not None and n_vids < max_videos \
                    and "agentview" in ep and "eye_in_hand" in ep:
                name = (f"ep{n_saved:03d}_{_task_short(ep['stage1_task'])}"
                        f"__then__{_task_short(ep['stage2_task'])}.mp4")
                frames = _make_video_frames(
                    ep["agentview"], ep["eye_in_hand"],
                    stage1_end=ep["stage1_end"],
                    homing_end=ep["homing_end"],
                )
                _write_video(frames, videos_dir / name, fps=fps)
                n_vids += 1
                print(f"  video  → {name}")

            T = len(ep["actions"])
            print(f"  demo_{n_saved}  {ep['stage1_task']}  →  {ep['stage2_task']}"
                  f"  T={T}  (s1={ep['stage1_end']}"
                  f"  hom={ep['homing_end'] - ep['stage1_end']}"
                  f"  s2={T - ep['homing_end']})")
            n_saved += 1

    elapsed = time.time() - start
    return {
        "n_success_episodes": n_saved,
        "n_total_episodes":   n_total,
        "success_rate":       n_saved / max(n_total, 1) * 100,
        "elapsed_s":          elapsed,
        "hdf5_path":          str(hdf5_path),
        "n_videos_saved":     n_vids,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate two-stage demos with a state-based policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--n-success",  type=int, default=500,
                        help="Successful two-stage episodes to collect")
    parser.add_argument("--num-envs",   type=int, default=4)
    parser.add_argument("--horizon",    type=int, default=400,
                        help="Max steps per episode (both stages combined)")
    parser.add_argument("--n-execute",  type=int, default=8,
                        help="Policy MPC chunk size")
    parser.add_argument("--out-dir",    default="results/two_stage_gen")
    parser.add_argument("--max-videos", type=int, default=20,
                        help="Debug videos to save (0 = none)")
    parser.add_argument("--fps",        type=int, default=20)
    parser.add_argument("--no-homing",  action="store_true",
                        help="Skip inter-stage EEF homing")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    np.random.seed(args.seed)

    policy = build_policy(checkpoint=Path(args.checkpoint), device=args.device)

    env = TwoStageCGWrapper(
        make_env_fn=lambda: make_env(horizon=args.horizon),
        num_envs=args.num_envs,
        use_relative_coordinates=(policy.env_state_feature is not None),
    )

    run_id    = time.strftime("%Y%m%d-%H%M%S")
    out_dir   = Path(args.out_dir) / f"gen_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / "demos.hdf5"
    videos_dir = (out_dir / "videos") if args.max_videos > 0 else None

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    import torch
    with torch.no_grad():
        info = generate_dataset(
            env, policy,
            n_success_episodes=args.n_success,
            horizon=args.horizon,
            n_execute=args.n_execute,
            device=args.device,
            hdf5_path=hdf5_path,
            videos_dir=videos_dir,
            max_videos=args.max_videos,
            fps=args.fps,
            use_homing=not args.no_homing,
        )

    env.close()

    print("\n" + "=" * 60)
    print("Generation complete!")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    (out_dir / "gen_info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
