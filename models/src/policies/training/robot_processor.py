#!/usr/bin/env python
"""
Schema-driven data processor for robot manipulation datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from policies.datasets.schema import FeatureConfig, FeatureDef


class ProcessorConfig(BaseModel):
    """Configuration for ``RobotProcessor``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalization_type: Literal["zscore", "minmax"] = "zscore"
    normalize_actions: bool = True
    normalize_states: bool = True
    add_task_indices: bool = False
    obj_to_idx: Dict[str, int] = Field(default_factory=dict)
    container_to_idx: Dict[str, int] = Field(default_factory=dict)


class RobotProcessor:
    """
    Processor for robot manipulation data.

    Normalization statistics are computed for every HDF5-backed feature in the
    active ``FeatureConfig`` and stored by feature key. Video features are
    skipped automatically.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        *,
        config: ProcessorConfig | None = None,
        feature_config: FeatureConfig | None = None,
        preloaded_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        # Legacy positional args kept for backward compat with old scripts.
        task_files: Optional[Dict[str, str]] = None,
        normalization_type: str | None = None,
        normalize_actions: bool | None = None,
        normalize_states: bool | None = None,
        add_task_indices: bool | None = None,
        obj_to_idx: Mapping[str, int] | None = None,
        container_to_idx: Mapping[str, int] | None = None,
    ):
        if config is None:
            config = ProcessorConfig(
                normalization_type=normalization_type or "zscore",
                normalize_actions=normalize_actions if normalize_actions is not None else True,
                normalize_states=normalize_states if normalize_states is not None else True,
                add_task_indices=add_task_indices if add_task_indices is not None else False,
                obj_to_idx=dict(obj_to_idx) if obj_to_idx else {},
                container_to_idx=dict(container_to_idx) if container_to_idx else {},
            )
        self._config = config
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.task_files = task_files
        self.normalization_type = config.normalization_type
        self.normalize_actions = config.normalize_actions
        self.normalize_states = config.normalize_states
        self.add_task_indices = config.add_task_indices
        self.obj_to_idx = dict(config.obj_to_idx) if config.obj_to_idx else None
        self.container_to_idx = dict(config.container_to_idx) if config.container_to_idx else None
        if feature_config is None:
            raise ValueError("RobotProcessor requires a feature_config.")
        self.feature_config = feature_config
        self.stats_by_key: dict[str, dict[str, torch.Tensor]] = {}
        if preloaded_stats is not None:
            self.stats_by_key = preloaded_stats
        else:
            self._compute_stats()

    def infer_task_indices(self, tasks: list[str]) -> torch.Tensor:
        if self.obj_to_idx is None or self.container_to_idx is None:
            raise ValueError("infer_task_indices requires obj_to_idx and container_to_idx to be set on RobotProcessor")
        obj_idx = torch.zeros((len(tasks),), dtype=torch.long)
        cont_idx = torch.zeros((len(tasks),), dtype=torch.long)
        for i, t in enumerate(tasks):
            tl = (t or "").lower()
            for k, v in self.obj_to_idx.items():
                if k in tl:
                    obj_idx[i] = int(v)
                    break
            for k, v in self.container_to_idx.items():
                if k in tl:
                    cont_idx[i] = int(v)
                    break
        return torch.stack([obj_idx, cont_idx], dim=1)

    def _normalizable_features(self) -> tuple[FeatureDef, ...]:
        return tuple(fd for fd in self.feature_config.features if fd.is_hdf5)

    def _cache_path(self) -> Path:
        return self.data_dir / "normalization_stats.npz"

    def _legacy_role_to_key(self) -> dict[str, str]:
        out: dict[str, str] = {}
        out["action"] = "actions"
        if "state" in self.feature_config:
            out["state"] = "state"
        if "env_state" in self.feature_config:
            out["env_state"] = "env_state"
        return out

    def _try_load_stats_from_cache(self, cache_path: Path) -> bool:
        data = np.load(cache_path, allow_pickle=True)

        expected_shapes = {fd.key: tuple(fd.shape) for fd in self._normalizable_features()}

        if "__schema_version__" in data:
            version = int(np.asarray(data["__schema_version__"]).item())
            if version != 2:
                return False
            cache_keys = tuple(str(k) for k in data["feature_keys"].tolist())
            expected_keys = tuple(fd.key for fd in self._normalizable_features())
            if cache_keys != expected_keys:
                return False
            stats: dict[str, dict[str, torch.Tensor]] = {}
            for key in cache_keys:
                s = {
                    "min": torch.from_numpy(data[f"{key}__min"]).float(),
                    "max": torch.from_numpy(data[f"{key}__max"]).float(),
                    "mean": torch.from_numpy(data[f"{key}__mean"]).float(),
                    "std": torch.from_numpy(data[f"{key}__std"]).float(),
                }
                exp = expected_shapes.get(key)
                if exp is None:
                    return False
                if tuple(s["min"].shape) != exp or tuple(s["max"].shape) != exp or tuple(s["mean"].shape) != exp or tuple(s["std"].shape) != exp:
                    return False
                stats[key] = s
            self.stats_by_key = stats
            print("Loaded cached normalization statistics")
            return True

        role_to_key = self._legacy_role_to_key()
        legacy_map = {
            "action": ("action_min", "action_max", "action_mean", "action_std"),
            "state": ("robot_state_min", "robot_state_max", "robot_state_mean", "robot_state_std"),
            "env_state": ("env_state_min", "env_state_max", "env_state_mean", "env_state_std"),
        }
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for role, key in role_to_key.items():
            fields = legacy_map.get(role)
            if fields is None or any(field not in data for field in fields):
                return False
            s = {
                "min": torch.from_numpy(data[fields[0]]).float(),
                "max": torch.from_numpy(data[fields[1]]).float(),
                "mean": torch.from_numpy(data[fields[2]]).float(),
                "std": torch.from_numpy(data[fields[3]]).float(),
            }
            exp = expected_shapes.get(key)
            if exp is None:
                return False
            if tuple(s["min"].shape) != exp or tuple(s["max"].shape) != exp or tuple(s["mean"].shape) != exp or tuple(s["std"].shape) != exp:
                return False
            stats[key] = s
        expected_keys = {fd.key for fd in self._normalizable_features()}
        if set(stats.keys()) != expected_keys:
            return False
        self.stats_by_key = stats
        print("Loaded cached normalization statistics (legacy format)")
        return True

    def _load_feature_array(self, demo: h5py.Group, fd: FeatureDef) -> np.ndarray:
        if not fd.is_hdf5:
            raise TypeError(f"Feature '{fd.key}' is not HDF5-backed.")
        parts = [np.asarray(demo[path][()], dtype=np.float32) for path in fd.paths]
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _compute_stats(self) -> None:
        cache_path = self._cache_path()
        if cache_path.exists():
            print("Loading cached normalization statistics...")
            if self._try_load_stats_from_cache(cache_path):
                return
            print("Cached normalization statistics are incompatible with current schema; recomputing...")

        print("Computing normalization statistics from ALL demos (small-files)...")
        dataset_manifest_path = self.data_dir / "dataset_manifest.json"
        if not dataset_manifest_path.exists():
            raise FileNotFoundError(f"Expected small-files dataset_manifest.json at: {dataset_manifest_path}")

        with open(dataset_manifest_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        tasks = ds.get("tasks", [])
        if not tasks:
            raise ValueError(f"No tasks found in {dataset_manifest_path}")

        demo_index: list[dict[str, str]] = []
        for t in tasks:
            task_slug = str(t["task_slug"])
            task_manifest_path = self.data_dir / str(t["manifest"])
            with open(task_manifest_path, "r", encoding="utf-8") as f:
                tm = json.load(f)
            for sh in tm.get("shards", []):
                shard_hdf5_rel = f"{task_slug}/{sh['hdf5']}"
                for d in sh.get("demos", []):
                    demo_index.append({"file": shard_hdf5_rel, "demo_key": str(d["demo_key"])})

        arrays_by_key: dict[str, list[np.ndarray]] = {fd.key: [] for fd in self._normalizable_features()}
        demos_by_file: dict[str, list[dict[str, str]]] = {}
        for demo_info in demo_index:
            demos_by_file.setdefault(demo_info["file"], []).append(demo_info)

        for filename, demo_infos in demos_by_file.items():
            filepath = self.data_dir / filename
            with h5py.File(filepath, "r") as f:
                for demo_info in demo_infos:
                    demo = f[f"data/{demo_info['demo_key']}"]
                    for fd in self._normalizable_features():
                        arrays_by_key[fd.key].append(self._load_feature_array(demo, fd).copy())

        stats_by_key: dict[str, dict[str, torch.Tensor]] = {}
        for fd in self._normalizable_features():
            all_values = np.concatenate(arrays_by_key[fd.key], axis=0)
            stats_by_key[fd.key] = {
                "min": torch.from_numpy(all_values.min(axis=0)).float(),
                "max": torch.from_numpy(all_values.max(axis=0)).float(),
                "mean": torch.from_numpy(all_values.mean(axis=0)).float(),
                "std": torch.from_numpy(all_values.std(axis=0) + 1e-6).float(),
            }
        self.stats_by_key = stats_by_key

        payload: dict[str, np.ndarray] = {
            "__schema_version__": np.asarray(2, dtype=np.int64),
            "feature_keys": np.asarray([fd.key for fd in self._normalizable_features()], dtype=object),
        }
        for key, stats in self.stats_by_key.items():
            payload[f"{key}__min"] = stats["min"].numpy()
            payload[f"{key}__max"] = stats["max"].numpy()
            payload[f"{key}__mean"] = stats["mean"].numpy()
            payload[f"{key}__std"] = stats["std"].numpy()
        np.savez(cache_path, **payload)

    def _should_normalize_feature(self, fd: FeatureDef) -> bool:
        if fd.key == "actions":
            return bool(self.normalize_actions)
        return bool(self.normalize_states)

    def get_feature_stats(self, key: str) -> dict[str, torch.Tensor]:
        if key not in self.stats_by_key:
            raise KeyError(f"No normalization stats available for feature {key!r}.")
        return self.stats_by_key[key]

    def __call__(self, batch: Dict) -> Dict:
        if self.add_task_indices and ("task" in batch) and ("task_indices" not in batch):
            # Keep on CPU; training code will move to device with the rest of the batch.
            batch["task_indices"] = self.infer_task_indices(batch["task"])

        for fd in self._normalizable_features():
            if not self._should_normalize_feature(fd):
                continue
            if fd.key not in batch or not isinstance(batch[fd.key], torch.Tensor):
                continue
            stats = self.get_feature_stats(fd.key)
            value = batch[fd.key]
            if self.normalization_type == "zscore":
                value = (value - stats["mean"]) / stats["std"]
            elif self.normalization_type == "minmax":
                value = (value - stats["min"]) / (stats["max"] - stats["min"] + 1e-6)
            batch[fd.key] = value

        return batch

    def get_stats(self) -> Dict:
        return self.stats_by_key

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        stats = self.get_feature_stats("actions")
        device = actions.device
        dtype = actions.dtype
        if self.normalization_type == "zscore":
            std = stats["std"].to(device=device, dtype=dtype)
            mean = stats["mean"].to(device=device, dtype=dtype)
            return actions * std + mean
        if self.normalization_type == "minmax":
            max_v = stats["max"].to(device=device, dtype=dtype)
            min_v = stats["min"].to(device=device, dtype=dtype)
            return actions * (max_v - min_v + 1e-6) + min_v
        return actions

