import numpy as np
from contextlib import contextmanager


_L2_OBJECTS = ("cross", "cube", "cylinder")
_L2_CONTAINERS = ("bin", "cup", "plate")

_L4_OBJECTS = ("cross", "cube", "cylinder", "milk")
_L4_CONTAINERS = ("bin", "mug", "plate", "mug_no_handle")


def parse_task_indices(task: str | None, *, level: str = "auto") -> tuple[int, int]:
    """
    Parse a CG language task string into (object_idx, container_idx).

    Supports:
    - L2: cross/cube/cylinder into bin/cup/plate
    - L4: cross/cube/cylinder/milk into bin/mug/plate/mug_no_handle
    """
    tl = (task or "").lower()
    lvl = str(level).lower()
    if lvl not in {"l2", "l4", "auto"}:
        raise ValueError(f"level must be one of l2/l4/auto, got {level!r}")
    if lvl == "auto":
        # Disambiguate by container tokens first (cup vs mug/mug_no_handle),
        # then by the extra L4 object token (milk).
        if "cup" in tl:
            lvl = "l2"
        elif "mug_no_handle" in tl or "mug" in tl or "milk" in tl:
            lvl = "l4"
        else:
            lvl = "l2"

    obj_names  = _L4_OBJECTS if lvl == "l4" else _L2_OBJECTS
    cont_names = _L4_CONTAINERS if lvl == "l4" else _L2_CONTAINERS

    # Check longer tokens first so "mug_no_handle" doesn't match "mug".
    cont_candidates = sorted(cont_names, key=len, reverse=True)
    obj_candidates  = sorted(obj_names,  key=len, reverse=True)

    obj_name  = obj_candidates[next((i for i, n in enumerate(obj_candidates)  if n in tl), 0)]
    cont_name = cont_candidates[next((i for i, n in enumerate(cont_candidates) if n in tl), 0)]
    return int(obj_names.index(obj_name)), int(cont_names.index(cont_name))


def _infer_task_level_from_env(env) -> str:
    if hasattr(env, "mug_no_handle") or hasattr(env, "milk"):
        return "l4"
    return "l2"


def _all_tasks_for_level(level: str) -> list[str]:
    if str(level).lower() == "l4":
        objs, conts = _L4_OBJECTS, _L4_CONTAINERS
    else:
        objs, conts = _L2_OBJECTS, _L2_CONTAINERS
    return [f"place the {o} into the {c}" for o in objs for c in conts]


def all_tasks(*, level: str = "l2") -> list[str]:
    """Return the full task list for L2 or L4."""
    lvl = str(level).lower()
    if lvl not in {"l2", "l4"}:
        raise ValueError(f"level must be 'l2' or 'l4', got {level!r}")
    return _all_tasks_for_level(lvl)


@contextmanager
def _numpy_seed_scope(seed: int | None):
    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _transform_to_relative(eef_pos: np.ndarray, eef_quat: np.ndarray,
                            obj_data: np.ndarray) -> np.ndarray:
    """Transform object pose (7D: pos+quat) from world frame to EEF frame."""
    obj_pos, obj_quat = obj_data[:3], obj_data[3:7]

    # Relative position (world frame delta is sufficient for policy input).
    pos_rel = obj_pos - eef_pos

    # Relative orientation: q_rel = eef_quat_inv * obj_quat
    eef_inv = eef_quat.copy()
    eef_inv[..., :3] *= -1  # conjugate = inverse for unit quaternion

    x1, y1, z1, w1 = eef_inv[..., 0], eef_inv[..., 1], eef_inv[..., 2], eef_inv[..., 3]
    x2, y2, z2, w2 = obj_quat[..., 0], obj_quat[..., 1], obj_quat[..., 2], obj_quat[..., 3]
    quat_rel = np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)

    # Canonical form: enforce w >= 0.
    if quat_rel[..., 3] < 0:
        quat_rel = -quat_rel

    return np.concatenate([pos_rel, quat_rel])


class ImageBasedCGWrapper:
    """
    Batched gym-style wrapper for CG_L2 / CG_L4 image-conditioned evaluation.

    Usage:
        env = ImageBasedCGWrapper(make_env_fn, num_envs=4)
        env.train_task = ["place the cross into the bin", ...]  # task allow-list
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
                    if level == "l2" and not self.train_task:
                        # Legacy default: 3-task L2 distribution.
                        default_tasks = [
                            "place the cross into the bin",
                            "place the cube into the cup",
                            "place the cylinder into the plate",
                        ]
                        task_str = str(np.random.choice(default_tasks))
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
        plate    = np.concatenate([raw["plate_pos"],    raw["plate_quat"]])

        if "cup_pos" in raw:
            cup_like = np.concatenate([raw["cup_pos"], raw["cup_quat"]])
        elif "mug_pos" in raw:
            cup_like = np.concatenate([raw["mug_pos"], raw["mug_quat"]])
        else:
            cup_like = np.zeros(7, dtype=np.float32)

        if self.use_relative_coordinates:
            cross    = _transform_to_relative(eef_pos, eef_quat, cross)
            cube     = _transform_to_relative(eef_pos, eef_quat, cube)
            cylinder = _transform_to_relative(eef_pos, eef_quat, cylinder)
            bin_obj  = _transform_to_relative(eef_pos, eef_quat, bin_obj)
            cup_like = _transform_to_relative(eef_pos, eef_quat, cup_like)
            plate    = _transform_to_relative(eef_pos, eef_quat, plate)

        return {
            "observation.state":            np.concatenate([eef_pos, eef_quat, gripper]),
            "observation.environment_state": np.concatenate([cross, cube, cylinder, bin_obj, cup_like, plate]),
            "observation.images.agentview":          np.flipud(raw["agentview_image"]),
            "observation.images.robot0_eye_in_hand": np.flipud(raw["robot0_eye_in_hand_image"]),
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
