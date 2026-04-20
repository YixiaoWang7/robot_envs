import numpy as np

from CG_L2_image_wrapper import (
    _all_tasks_for_level,
    _infer_task_level_from_env,
    _numpy_seed_scope,
    _transform_to_relative,
)


class StateBasedCGWrapper:
    """
    Batched gym-style wrapper for CG_L2 / CG_L4 state-conditioned evaluation.

    Usage:
        env = StateBasedCGWrapper(make_env_fn, num_envs=4)
        env.train_task = "all"          # or a list of task strings
        obs, _ = env.reset()
        obs, rew, term, trunc, info = env.step(actions)  # actions: (N, 7)
    """

    def __init__(self, make_env_fn, num_envs: int = 1,
                 use_relative_coordinates: bool = False,
                 gripper_types: str = "PandaGripper"):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.obs = [None] * num_envs
        self.step_counter = [0] * num_envs
        self.use_relative_coordinates = use_relative_coordinates
        self.gripper_types = gripper_types
        self.is_success = [False] * num_envs
        self.train_task = None  # str | list[str] | None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        tasks: list[str] | None = None,
        reset_seeds: list[int] | None = None,
        **kwargs,
    ):
        kwargs.pop("seed", None)
        self.obs = []
        if tasks is not None and len(tasks) != self.num_envs:
            raise ValueError(f"tasks must have length {self.num_envs}, got {len(tasks)}")
        if reset_seeds is not None and len(reset_seeds) != self.num_envs:
            raise ValueError(f"reset_seeds must have length {self.num_envs}, got {len(reset_seeds)}")

        for i, env in enumerate(self.envs):
            with _numpy_seed_scope(None if reset_seeds is None else reset_seeds[i]):
                level = _infer_task_level_from_env(env)

                if tasks is not None:
                    task_str = str(tasks[i])
                elif isinstance(self.train_task, (list, tuple)) and self.train_task:
                    task_str = str(np.random.choice(list(self.train_task)))
                elif isinstance(self.train_task, str) and self.train_task and self.train_task.lower() != "all":
                    task_str = str(self.train_task)
                else:
                    task_str = str(np.random.choice(_all_tasks_for_level(level)))

                if hasattr(env, "update_task") and callable(env.update_task):
                    env.update_task(task_str)
                else:
                    env.task = task_str
                if hasattr(env, "_refresh_task_pointers") and callable(env._refresh_task_pointers):
                    env._refresh_task_pointers()

                obs_i = env.reset(**kwargs)
            self.obs.append(obs_i)
            self.step_counter[i] = 0

        obs_dicts = [self._compute_observation(i) for i in range(self.num_envs)]
        batched_obs = {k: np.stack([o[k] for o in obs_dicts], axis=0) for k in obs_dicts[0]}
        self.is_success = [False] * self.num_envs
        return batched_obs, [{} for _ in range(self.num_envs)]

    def step(self, actions):
        next_obs, rewards, terminateds, infos = [], [], [], []

        for i, env in enumerate(self.envs):
            if self.is_success[i]:
                obs, reward, terminated, info = env.step(np.zeros_like(actions[i]))
            else:
                obs, reward, terminated, info = env.step(actions[i].copy())

            self.obs[i] = obs
            self.step_counter[i] += 1

            info["is_success"] = bool(env._check_success())
            if not self.is_success[i]:
                self.is_success[i] = info["is_success"]

            infos.append(info)
            rewards.append(reward)
            terminateds.append(terminated)
            next_obs.append(self._compute_observation(i))

        batched_obs = {k: np.stack([o[k] for o in next_obs], axis=0) for k in next_obs[0]}
        final_info = [infos[i] if terminateds[i] else None for i in range(self.num_envs)]
        return (
            batched_obs,
            np.array(rewards),
            np.array(terminateds),
            np.array([False] * self.num_envs),
            {"is_success": self.is_success, "final_info": final_info},
        )

    def _compute_observation(self, idx: int) -> dict:
        raw = self.obs[idx]
        eef_pos  = raw["robot0_eef_pos"]
        eef_quat = raw["robot0_eef_quat"]
        gripper  = raw["robot0_gripper_qpos"]

        if self.gripper_types == "RethinkGripper":
            panda_close   = np.array([ 0.00049761, -0.00049912])
            panda_open    = np.array([ 0.03948026, -0.03948177])
            rethink_close = np.array([-0.0118367,   0.01183658])
            rethink_open  = np.array([ 0.01106267, -0.01106308])
            gripper = (panda_close
                       + (gripper - rethink_close) / (rethink_open - rethink_close)
                       * (panda_open - panda_close))

        cross    = np.concatenate([raw["cross_pos"],    raw["cross_quat"]])
        cube     = np.concatenate([raw["cube_pos"],     raw["cube_quat"]])
        cylinder = np.concatenate([raw["cylinder_pos"], raw["cylinder_quat"]])
        bin_obj  = np.concatenate([raw["bin_pos"],      raw["bin_quat"]])
        cup      = np.concatenate([raw["cup_pos"],      raw["cup_quat"]])
        plate    = np.concatenate([raw["plate_pos"],    raw["plate_quat"]])

        if self.use_relative_coordinates:
            cross    = _transform_to_relative(eef_pos, eef_quat, cross)
            cube     = _transform_to_relative(eef_pos, eef_quat, cube)
            cylinder = _transform_to_relative(eef_pos, eef_quat, cylinder)
            bin_obj  = _transform_to_relative(eef_pos, eef_quat, bin_obj)
            cup      = _transform_to_relative(eef_pos, eef_quat, cup)
            plate    = _transform_to_relative(eef_pos, eef_quat, plate)

        gripper_to_object_A = np.asarray(raw["gripper_to_object_A_pos"], dtype=np.float32)

        return {
            "observation.state":             np.concatenate([eef_pos, eef_quat, gripper]),
            "observation.environment_state": np.concatenate([cross, cube, cylinder, bin_obj, cup, plate,
                                                             gripper_to_object_A]),
        }

    def render(self):
        frames = []
        for env in self.envs:
            try:
                frame = env.sim.render(camera_name="agentview", width=256, height=256)
                frames.append(np.flipud(frame))
            except Exception as e:
                print(f"Render failed: {e}")
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
        return frames

    def close(self):
        for env in self.envs:
            env.close()

    # ------------------------------------------------------------------
    # Gym / VecEnv compatibility helpers
    # ------------------------------------------------------------------

    def __getattr__(self, attr):
        if hasattr(self.envs[0], attr):
            return getattr(self.envs[0], attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

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
