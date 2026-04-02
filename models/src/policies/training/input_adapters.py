"""Batch -> model input adapters.

Adapters isolate modality-/architecture-specific packing logic so:
- `RobotFlowPolicyWrapper` stays network-agnostic
- trainer/evaluator can work with different policy networks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from policies.datasets.schema import FeatureConfig, FeatureDef


class BaseInputAdapter(Protocol):
    def pack_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]: ...

    def pack_infer(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]: ...


def _require(batch: dict[str, Any], key: str) -> torch.Tensor:
    if key not in batch or not torch.is_tensor(batch[key]):
        raise KeyError(f"Expected tensor batch[{key!r}]")
    return batch[key]


def _require_optional(batch: dict[str, Any], key: str) -> torch.Tensor | None:
    if key not in batch:
        return None
    if not torch.is_tensor(batch[key]):
        raise KeyError(f"Expected tensor batch[{key!r}]")
    return batch[key]


@dataclass(frozen=True)
class _CommonFeatures:
    actions: FeatureDef
    state: FeatureDef
    env_state: FeatureDef | None
    images: FeatureDef | None


def _resolve_features(feature_config: FeatureConfig) -> _CommonFeatures:
    actions = feature_config["actions"]
    state = feature_config["state"]  # robot state is required by convention
    env_state = feature_config["env_state"] if "env_state" in feature_config else None
    images = feature_config["images"] if "images" in feature_config else None
    return _CommonFeatures(actions=actions, state=state, env_state=env_state, images=images)


@dataclass(frozen=True)
class ImageInputAdapter:
    """Adapter for `FlowMatchingAttentionNetwork` (image-conditioned)."""

    feature_config: FeatureConfig
    cameras: tuple[str, ...]

    def __post_init__(self) -> None:
        f = _resolve_features(self.feature_config)
        if f.images is None:
            raise ValueError("ImageInputAdapter requires feature_config to include 'images'.")
        if not self.cameras:
            raise ValueError("ImageInputAdapter requires at least one camera.")

    def pack_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        f = _resolve_features(self.feature_config)
        assert f.images is not None
        actions = _require(batch, f.actions.key)
        robot_state = _require(batch, f.state.key)
        task_indices = _require(batch, "task_indices")
        image_payload = batch[f.images.key]
        if not isinstance(image_payload, dict):
            raise TypeError("Expected batch['images'] payload to be a dict keyed by camera name.")
        images = torch.stack([image_payload[c] for c in self.cameras], dim=2)
        return {
            "actions": actions,
            "robot_state": robot_state,
            "images": images,
            "task_indices": task_indices,
        }

    def pack_infer(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        f = _resolve_features(self.feature_config)
        assert f.images is not None
        robot_state = _require(batch, f.state.key)
        task_indices = _require(batch, "task_indices")
        image_payload = batch[f.images.key]
        if not isinstance(image_payload, dict):
            raise TypeError("Expected batch['images'] payload to be a dict keyed by camera name.")
        images = torch.stack([image_payload[c] for c in self.cameras], dim=2)
        return {
            "robot_state": robot_state,
            "images": images,
            "task_indices": task_indices,
        }


@dataclass(frozen=True)
class EnvStateTokenInputAdapter:
    """Adapter for `FlowMatchingEnvStateAttentionNetwork` (robot_state + env_state tokens)."""

    feature_config: FeatureConfig

    def __post_init__(self) -> None:
        f = _resolve_features(self.feature_config)
        if f.env_state is None:
            raise ValueError("EnvStateTokenInputAdapter requires feature_config to include 'env_state'.")
        if f.images is not None:
            raise ValueError("EnvStateTokenInputAdapter is for non-image policies; remove 'images' from feature_config.")

    def pack_train(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        f = _resolve_features(self.feature_config)
        assert f.env_state is not None
        actions = _require(batch, f.actions.key)
        robot_state = _require(batch, f.state.key)
        env_state = _require(batch, f.env_state.key)
        task_indices = _require(batch, "task_indices")
        return {
            "actions": actions,
            "robot_state": robot_state,
            "env_state": env_state,
            "task_indices": task_indices,
        }

    def pack_infer(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        f = _resolve_features(self.feature_config)
        assert f.env_state is not None
        robot_state = _require(batch, f.state.key)
        env_state = _require(batch, f.env_state.key)
        task_indices = _require(batch, "task_indices")
        return {
            "robot_state": robot_state,
            "env_state": env_state,
            "task_indices": task_indices,
        }

