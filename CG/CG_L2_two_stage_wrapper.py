"""
Two-stage task wrapper for CG_L2.

Stage 1: place obj_A into cont_B
Stage 2: place obj_C into cont_D  (obj_A != obj_C, cont_B != cont_D)

The wrapper manages the stage transition mid-episode by calling
env.update_task() and repointing the env's internal object/body references
so that env._check_success() evaluates the stage-2 condition correctly.

Camera images (agentview + robot0_eye_in_hand) are included in observations.
The env must be created with use_camera_obs=True and the two camera names.
"""

from __future__ import annotations

import numpy as np

_OBJ_NAMES  = ["cross", "cube", "cylinder"]
_CONT_NAMES = ["bin", "cup", "plate"]
_OBJ_IDX    = {n: i for i, n in enumerate(_OBJ_NAMES)}
_CONT_IDX   = {n: i for i, n in enumerate(_CONT_NAMES)}


def _parse_task_indices(task: str) -> tuple[int, int]:
    tl = task.lower()
    obj  = next((v for k, v in _OBJ_IDX.items()  if k in tl), 0)
    cont = next((v for k, v in _CONT_IDX.items() if k in tl), 0)
    return obj, cont


def _fix_env_task_pointers(env) -> None:
    """
    After env.update_task() only sets object_A_index / object_B_index,
    this function updates the live object references and MuJoCo body IDs
    so that _check_success() and the gripper_to_object_A observable work
    correctly for the new target.
    """
    obj_A_list = [env.cross, env.cube, env.cylinder]
    obj_B_list = [env.bin, env.cup, env.plate]
    env.object_A = obj_A_list[env.object_A_index]
    env.object_B = obj_B_list[env.object_B_index]
    env.object_A_body_id = env.sim.model.body_name2id(env.object_A.root_body)
    env.object_B_body_id = env.sim.model.body_name2id(env.object_B.root_body)


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.stack([x, y, z, w], axis=-1)


def quaternion_inverse(q):
    q_inv = q.copy()
    q_inv[..., :3] *= -1.0
    return q_inv


def canonicalize_quaternion(q):
    q_c = q.copy()
    if q_c[..., 3] < 0:
        q_c = -q_c
    return q_c


def transform_to_relative_coordinates(eef_pos, eef_quat, obj_data):
    """World-frame 7D pose → EEF-relative 7D pose."""
    obj_pos  = obj_data[:3]
    obj_quat = obj_data[3:7]
    pos_rel  = obj_pos - eef_pos
    eef_inv  = quaternion_inverse(eef_quat)
    quat_rel = quaternion_multiply(eef_inv, obj_quat)
    quat_rel = canonicalize_quaternion(quat_rel)
    return np.concatenate([pos_rel, quat_rel])


class TwoStageCGWrapper:
    """
    Vectorised wrapper around CG_L2 for two-stage pick-and-place generation.

    Observations returned:
      observation.state                 (9,)  eef_pos+quat+gripper
      observation.environment_state    (45,)  6×7D EEF-relative poses + 3D gripper_to_target
      observation.images.agentview     (H, W, 3) uint8
      observation.images.robot0_eye_in_hand (H, W, 3) uint8

    Episode is considered successful only when BOTH stages succeed.
    """

    def __init__(
        self,
        make_env_fn,
        num_envs: int = 1,
        use_relative_coordinates: bool = True,
    ):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.use_relative_coordinates = use_relative_coordinates

        self.obs = [None] * num_envs

        # Per-env task info
        self.stage1_task: list[str] = [""] * num_envs
        self.stage2_task: list[str] = [""] * num_envs
        self.stage1_obj_idx:  list[int] = [0] * num_envs
        self.stage1_cont_idx: list[int] = [0] * num_envs
        self.stage2_obj_idx:  list[int] = [0] * num_envs
        self.stage2_cont_idx: list[int] = [0] * num_envs

        # Stage tracking
        self.current_stage: list[int]  = [0] * num_envs
        self.is_success:    list[bool] = [False] * num_envs

    # ------------------------------------------------------------------
    # Task sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_two_stage_tasks() -> tuple[str, str, int, int, int, int]:
        """
        Sample (stage1_task, stage2_task) with:
          obj_A  != obj_C  and  cont_B != cont_D
        Returns task strings plus their integer indices.
        """
        obj_perm  = np.random.permutation(3)
        cont_perm = np.random.permutation(3)
        obj_A, obj_C   = obj_perm[0],  obj_perm[1]
        cont_B, cont_D = cont_perm[0], cont_perm[1]

        t1 = f"place the {_OBJ_NAMES[obj_A]} into the {_CONT_NAMES[cont_B]}"
        t2 = f"place the {_OBJ_NAMES[obj_C]} into the {_CONT_NAMES[cont_D]}"
        return t1, t2, int(obj_A), int(cont_B), int(obj_C), int(cont_D)

    # ------------------------------------------------------------------
    # Gymnasium-style interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        self.obs = []

        for i, env in enumerate(self.envs):
            t1, t2, oa, cb, oc, cd = self._sample_two_stage_tasks()
            self.stage1_task[i]     = t1
            self.stage2_task[i]     = t2
            self.stage1_obj_idx[i]  = oa
            self.stage1_cont_idx[i] = cb
            self.stage2_obj_idx[i]  = oc
            self.stage2_cont_idx[i] = cd
            self.current_stage[i]   = 0

            env.strategy = "fixed"
            env.task     = t1
            obs_i = env.reset(**kwargs)
            self.obs.append(obs_i)

        self.is_success = [False] * self.num_envs

        obs_dicts = [self._compute_observation(i) for i in range(self.num_envs)]
        batched   = {k: np.stack([o[k] for o in obs_dicts]) for k in obs_dicts[0]}
        return batched, [{} for _ in range(self.num_envs)]

    def step(self, actions):
        next_obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            if self.is_success[i]:
                obs_i, reward, terminated, info = env.step(np.zeros_like(actions[i]))
            else:
                obs_i, reward, terminated, info = env.step(actions[i].copy())

            self.obs[i] = obs_i

            # ---------- stage logic ----------
            stage_success = bool(env._check_success())

            if self.current_stage[i] == 0 and stage_success:
                # Stage 1 done → switch to stage 2
                self.current_stage[i] = 1
                env.update_task(self.stage2_task[i])
                _fix_env_task_pointers(env)
                # Refresh obs with new task context (next step will read updated observable)
                obs_i = env._get_observations()
                self.obs[i] = obs_i
                stage_success = False  # Stage 2 not yet done

            elif self.current_stage[i] == 1 and stage_success:
                # Stage 2 done → episode success
                self.is_success[i] = True

            info["is_success"] = self.is_success[i]
            info["current_stage"] = self.current_stage[i]

            next_obs_list.append(self._compute_observation(i))
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(False)
            infos.append(info)

        batched = {k: np.stack([o[k] for o in next_obs_list]) for k in next_obs_list[0]}
        final_info = [
            infos[i] if terminateds[i] or truncateds[i] else None
            for i in range(self.num_envs)
        ]
        return (
            batched,
            np.array(rewards, dtype=np.float32),
            np.array(terminateds),
            np.array(truncateds),
            {"is_success": list(self.is_success), "final_info": final_info},
        )

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _compute_observation(self, idx: int) -> dict:
        raw = self.obs[idx]

        eef_pos   = raw["robot0_eef_pos"].astype(np.float32)
        eef_quat  = raw["robot0_eef_quat"].astype(np.float32)
        gripper_q = raw["robot0_gripper_qpos"].astype(np.float32)

        cross    = np.concatenate([raw["cross_pos"],    raw["cross_quat"]]).astype(np.float32)
        cube     = np.concatenate([raw["cube_pos"],     raw["cube_quat"]]).astype(np.float32)
        cylinder = np.concatenate([raw["cylinder_pos"], raw["cylinder_quat"]]).astype(np.float32)
        bin_obj  = np.concatenate([raw["bin_pos"],      raw["bin_quat"]]).astype(np.float32)
        cup      = np.concatenate([raw["cup_pos"],      raw["cup_quat"]]).astype(np.float32)
        plate    = np.concatenate([raw["plate_pos"],    raw["plate_quat"]]).astype(np.float32)

        # Store world-frame poses for HDF5 saving (before relative transform)
        object_world = np.concatenate([cross, cube, cylinder, bin_obj, cup, plate])  # (42,)

        if self.use_relative_coordinates:
            cross    = transform_to_relative_coordinates(eef_pos, eef_quat, cross)
            cube     = transform_to_relative_coordinates(eef_pos, eef_quat, cube)
            cylinder = transform_to_relative_coordinates(eef_pos, eef_quat, cylinder)
            bin_obj  = transform_to_relative_coordinates(eef_pos, eef_quat, bin_obj)
            cup      = transform_to_relative_coordinates(eef_pos, eef_quat, cup)
            plate    = transform_to_relative_coordinates(eef_pos, eef_quat, plate)

        gripper_to_target = np.asarray(raw["gripper_to_object_A_pos"], dtype=np.float32)

        state           = np.concatenate([eef_pos, eef_quat, gripper_q])           # (9,)
        environment_state = np.concatenate(
            [cross, cube, cylinder, bin_obj, cup, plate, gripper_to_target]
        )  # (45,)

        out = {
            "observation.state":             state,
            "observation.environment_state": environment_state,
            "observation.object_world":      object_world,  # (42,) world-frame for HDF5
        }

        # Camera images (present when use_camera_obs=True in the CG_L2 env).
        # Robosuite renders offscreen bottom-to-top (MuJoCo convention), so
        # np.flipud is applied to match the orientation used by ImageBasedCGWrapper
        # and the image-based training pipeline.
        if "agentview_image" in raw:
            out["observation.images.agentview"] = np.flipud(
                raw["agentview_image"].astype(np.uint8)
            )
        if "robot0_eye_in_hand_image" in raw:
            out["observation.images.robot0_eye_in_hand"] = np.flipud(
                raw["robot0_eye_in_hand_image"].astype(np.uint8)
            )

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def render(self):
        frames = []
        for env in self.envs:
            try:
                frame = env.sim.render(camera_name="agentview", width=256, height=256)
                frames.append(np.flipud(frame))
            except Exception:
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
        return frames

    def close(self):
        for env in self.envs:
            env.close()

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            if name in {"_max_episode_steps", "horizon"}:
                results.append(env.horizon)
            else:
                attr = getattr(env, name)
                results.append(attr(*args, **kwargs) if callable(attr) else attr)
        return results

    def get_attr(self, name):
        return [getattr(env, name) for env in self.envs]

    def set_attr(self, name, values):
        for env, val in zip(self.envs, values):
            setattr(env, name, val)

    @property
    def unwrapped(self):
        return self

    @property
    def metadata(self):
        return {"render_fps": 20}

    def __getattr__(self, attr):
        if hasattr(self.envs[0], attr):
            return getattr(self.envs[0], attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
