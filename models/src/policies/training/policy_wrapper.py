"""Policy wrapper that binds model + processor for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from policies.datasets.schema import FeatureConfig
from policies.training.robot_processor import RobotProcessor
from policies.training.trainer import move_to_device


class RobotFlowPolicyWrapper(torch.nn.Module):
    """
    Binds a flow-matching policy network with its preprocessing pipeline.

    This keeps Trainer / Evaluator / inference code independent of how inputs
    are normalized and packed, and enables saving model + processor parameters
    together in a single checkpoint.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: RobotProcessor,
        cameras: list[str],
        feature_config: FeatureConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.cameras = [str(c) for c in cameras]
        self.feature_config = feature_config
        self.action_feature = self.feature_config["actions"]
        if "state" not in self.feature_config:
            raise KeyError("FeatureConfig must contain 'state' (robot state).")
        self.state_feature = self.feature_config["state"]
        self.env_state_feature = self.feature_config["env_state"] if "env_state" in self.feature_config else None
        self.image_feature = self.feature_config["images"] if "images" in self.feature_config else None

    # -- Inference helpers ---------------------------------------------------

    def process_raw_batch(self, raw_batch: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
        return move_to_device(self.processor(raw_batch), device)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.processor.denormalize_actions(actions)

    # -- Checkpoint save -----------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        run_config: Any,
        optimizer_state_dict: dict | None = None,
        scheduler_state_dict: dict | None = None,
        step: int | None = None,
        loss: float | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "run_config": run_config.to_json_dict(),
            "model_state_dict": self.model.state_dict(),
            "processor_stats": self.processor.get_stats(),
        }
        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict
        if scheduler_state_dict is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state_dict
        if step is not None:
            checkpoint["step"] = int(step)
        if loss is not None:
            checkpoint["loss"] = float(loss)

        torch.save(checkpoint, path)
