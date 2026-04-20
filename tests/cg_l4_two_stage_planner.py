"""
Two-stage CG_L4 planner utilities.

Implements a clean high-level planner that composes two pick→place goals:
  (obj0 -> cont0) then (obj1 -> cont1)

This module is intentionally standalone to avoid import cycles with
`generate_cg_l4_motion_planning.py`. The generator passes in a low-level planner
factory (e.g. SimplePickPlacePlanner) and uses the helpers here for sampling /
task-string/slug creation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class TwoStageTaskSpec:
    obj0: int
    cont0: int
    obj1: int
    cont1: int


def stage_task_string(*, obj_name: str, cont_name: str) -> str:
    # Keep CG_L4's expected instruction style for stage-level task resets.
    return f"place the {obj_name} into the {cont_name}"


def two_stage_task_string(
    *,
    obj0_name: str,
    cont0_name: str,
    obj1_name: str,
    cont1_name: str,
) -> str:
    # User requested single-string label: "pick a, place to b, pick c, place to d"
    return f"pick {obj0_name}, place to {cont0_name}, pick {obj1_name}, place to {cont1_name}"


def two_stage_task_slug(
    *,
    obj0_name: str,
    cont0_name: str,
    obj1_name: str,
    cont1_name: str,
) -> str:
    # Stable folder/task slug (used for per-combo generation).
    return f"{obj0_name}_into_{cont0_name}__{obj1_name}_into_{cont1_name}"


def _name_to_idx(name: str, names: list[str]) -> int:
    name = str(name).strip()
    if name in names:
        return int(names.index(name))
    lowered = name.lower()
    for i, n in enumerate(names):
        if str(n).lower() == lowered:
            return int(i)
    raise ValueError(f"Name {name!r} not found in names={names}")


def load_two_stage_specs_json(
    path: str | Path,
    *,
    object_names: list[str],
    container_names: list[str],
) -> list[TwoStageTaskSpec]:
    """
    Load two-stage combos from JSON.

    Supported formats:
      1) list of dicts with keys: obj0, cont0, obj1, cont1
         values can be ints (indices) or strings (names).
      2) dict with key "specs": [ ... same as above ... ]
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    specs_raw = data["specs"] if isinstance(data, dict) and "specs" in data else data
    if not isinstance(specs_raw, list):
        raise ValueError("Expected a list of specs or {'specs': [...]} in two-stage specs JSON.")

    out: list[TwoStageTaskSpec] = []
    for idx, s in enumerate(specs_raw):
        if not isinstance(s, dict):
            raise ValueError(f"Spec[{idx}] must be a dict, got {type(s)}")

        def _parse(v: object, *, names: list[str]) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, str):
                return _name_to_idx(v, names)
            raise ValueError(f"Spec[{idx}] field must be int or str, got {type(v)}")

        obj0 = _parse(s.get("obj0"), names=object_names)
        cont0 = _parse(s.get("cont0"), names=container_names)
        obj1 = _parse(s.get("obj1"), names=object_names)
        cont1 = _parse(s.get("cont1"), names=container_names)
        out.append(TwoStageTaskSpec(obj0=obj0, cont0=cont0, obj1=obj1, cont1=cont1))
    return out


def enumerate_all_two_stage_combos(
    *,
    num_objects: int,
    num_containers: int,
    distinct_objects: bool = True,
    distinct_containers: bool = True,
) -> list[TwoStageTaskSpec]:
    out: list[TwoStageTaskSpec] = []
    for o0 in range(int(num_objects)):
        for c0 in range(int(num_containers)):
            for o1 in range(int(num_objects)):
                if distinct_objects and int(o1) == int(o0):
                    continue
                for c1 in range(int(num_containers)):
                    if distinct_containers and int(c1) == int(c0):
                        continue
                    out.append(TwoStageTaskSpec(obj0=int(o0), cont0=int(c0), obj1=int(o1), cont1=int(c1)))
    return out


def set_env_target(env, *, obj_idx: int, cont_idx: int, task_str: Optional[str] = None) -> None:
    """
    Retarget CG_L4 to a new (object_A, object_B) pair *without resetting*.

    We mirror the logic of `fix_env_task_pointers()` locally to avoid import cycles.
    """
    env.object_A_index = int(obj_idx)
    env.object_B_index = int(cont_idx)
    if task_str is not None:
        env.task = str(task_str)
    # Update live object refs and body ids.
    env.object_A = env.object_A_list[env.object_A_index]
    env.object_B = env.object_B_list[env.object_B_index]
    env.object_A_body_id = env.sim.model.body_name2id(env.object_A.root_body)
    env.object_B_body_id = env.sim.model.body_name2id(env.object_B.root_body)


_PICK_WP = {"pregrasp", "rotate_for_grasp", "grasp"}
_PLACE_WP = {"post_grasp_lift", "prerelease", "release", "lift"}


class TwoStagePickPlacePlanner:
    """
    High-level planner that runs two sequential low-level pick-place planners.

    The low-level planner must provide:
      - reset()
      - get_action(env, eef_quat_xyzw=...) -> np.ndarray (7,)
      - done: bool
      - phase: str (waypoint/stage name)
    """

    def __init__(
        self,
        *,
        spec: TwoStageTaskSpec,
        object_names: list[str],
        container_names: list[str],
        low_level_planner_factory: Callable[[], object],
    ):
        self.spec = spec
        self.object_names = list(object_names)
        self.container_names = list(container_names)
        self._make_low = low_level_planner_factory

        self.stage_idx: int = 0
        self.low = self._make_low()
        self.done: bool = False
        self.phase: str = "init"
        self._stage_success = [False, False]
        self._just_switched: bool = False

    def reset(self) -> None:
        self.stage_idx = 0
        self.low = self._make_low()
        if hasattr(self.low, "reset"):
            self.low.reset()
        self.done = False
        self.phase = "init"
        self._stage_success = [False, False]
        self._just_switched = False

    def episode_success(self) -> bool:
        return bool(self._stage_success[0] and self._stage_success[1])

    def current_task_indices(self) -> np.ndarray:
        if int(self.stage_idx) == 0:
            return np.asarray([self.spec.obj0, self.spec.cont0], dtype=np.int64)
        return np.asarray([self.spec.obj1, self.spec.cont1], dtype=np.int64)

    def current_subtask_id(self) -> int:
        ph = str(getattr(self.low, "phase", self.phase) or self.phase)
        is_pick = ph in _PICK_WP
        is_place = ph in _PLACE_WP
        # Default unknown phases to "place" (more conservative).
        base = 0 if is_pick and not is_place else 1
        return int(base + 2 * int(self.stage_idx))

    def just_switched_stage(self) -> bool:
        return bool(self._just_switched)

    def _stage_task_str(self, *, stage: int) -> str:
        if int(stage) == 0:
            return stage_task_string(
                obj_name=self.object_names[int(self.spec.obj0)],
                cont_name=self.container_names[int(self.spec.cont0)],
            )
        return stage_task_string(
            obj_name=self.object_names[int(self.spec.obj1)],
            cont_name=self.container_names[int(self.spec.cont1)],
        )

    def episode_task_str(self) -> str:
        return two_stage_task_string(
            obj0_name=self.object_names[int(self.spec.obj0)],
            cont0_name=self.container_names[int(self.spec.cont0)],
            obj1_name=self.object_names[int(self.spec.obj1)],
            cont1_name=self.container_names[int(self.spec.cont1)],
        )

    def episode_task_slug(self) -> str:
        return two_stage_task_slug(
            obj0_name=self.object_names[int(self.spec.obj0)],
            cont0_name=self.container_names[int(self.spec.cont0)],
            obj1_name=self.object_names[int(self.spec.obj1)],
            cont1_name=self.container_names[int(self.spec.cont1)],
        )

    def _maybe_transition(self, env) -> None:
        # Called after env.step(), so success reflects the *post-action* state.
        if bool(env._check_success()):
            self._stage_success[int(self.stage_idx)] = True

        if int(self.stage_idx) == 0 and bool(self._stage_success[0]) and bool(getattr(self.low, "done", False)):
            # Switch to stage 1, retarget env in-place, and reset low-level planner.
            set_env_target(
                env,
                obj_idx=int(self.spec.obj1),
                cont_idx=int(self.spec.cont1),
                task_str=self._stage_task_str(stage=1),
            )
            if hasattr(self.low, "reset"):
                self.low.reset()
            self.stage_idx = 1
            self._just_switched = True

        # Mark overall done once stage 1 success has happened and the low-level planner finished.
        if int(self.stage_idx) == 1:
            if bool(env._check_success()):
                self._stage_success[1] = True
            if bool(self._stage_success[1]) and bool(getattr(self.low, "done", False)):
                self.done = True

    def post_step(self, env) -> None:
        """
        Observe the env *after* applying the previous action.

        This is where success is updated and stage transitions happen.
        """
        if self.done:
            return
        self._maybe_transition(env)

    def get_action(self, env, *, eef_quat_xyzw: np.ndarray) -> np.ndarray:
        self._just_switched = False
        if self.done:
            self.phase = "done"
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        # Ensure env pointers match the active stage (helps debugging and some env variants).
        cur_obj = int(self.current_task_indices()[0])
        cur_cont = int(self.current_task_indices()[1])
        if int(getattr(env, "object_A_index", -999)) != cur_obj or int(getattr(env, "object_B_index", -999)) != cur_cont:
            set_env_target(
                env,
                obj_idx=cur_obj,
                cont_idx=cur_cont,
                task_str=self._stage_task_str(stage=int(self.stage_idx)),
            )

        act = self.low.get_action(env, eef_quat_xyzw=eef_quat_xyzw)  # type: ignore[attr-defined]
        self.phase = f"stage{int(self.stage_idx)}::{getattr(self.low, 'phase', 'unknown')}"
        return np.asarray(act, dtype=np.float32)

