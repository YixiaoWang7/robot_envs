"""Checkpoint loading utilities for robot flow policies.

Keep `RobotFlowPolicyWrapper` agnostic to model architecture by centralizing
model reconstruction logic here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from policies.algorithms.flow_matching import FlowMatchingAlgorithm
from policies.config.run_config import RunConfig
from policies.policies.flow_policy import FlowMatchingAttentionNetwork
from policies.policies.flow_policy_env_state import FlowMatchingEnvStateAttentionNetwork
from policies.training.policy_wrapper import RobotFlowPolicyWrapper
from policies.training.robot_processor import RobotProcessor


def load_robot_flow_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> RobotFlowPolicyWrapper:
    """Reconstruct a full policy (model + processor) from a single checkpoint .pt."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = RunConfig.model_validate(checkpoint["run_config"])

    feature_config = cfg.feature
    action_feature = feature_config["actions"]
    state_feature = feature_config["state"]

    action_dim = int(np.prod(action_feature.shape))
    state_dim = int(np.prod(state_feature.shape))

    if "images" in feature_config:
        raw_hw = int(cfg.model.raw_image_size)
        model_input_shape: tuple[int, int, int] = (3, raw_hw, raw_hw)
        model: torch.nn.Module = FlowMatchingAttentionNetwork(
            action_dim=action_dim,
            horizon=action_feature.window_size,
            robot_state_dim=state_dim,
            n_obs_steps=state_feature.window_size,
            num_cameras=len(cfg.dataset.cameras),
            num_objects=len(cfg.processor.obj_to_idx),
            num_containers=len(cfg.processor.container_to_idx),
            input_shape=model_input_shape,
            config=cfg.model,
        )
    elif "env_state" in feature_config:
        env_state_feature = feature_config["env_state"]
        env_state_dim = int(np.prod(env_state_feature.shape))
        model = FlowMatchingEnvStateAttentionNetwork(
            action_dim=action_dim,
            horizon=action_feature.window_size,
            robot_state_dim=state_dim,
            env_state_dim=env_state_dim,
            n_obs_steps=state_feature.window_size,
            num_objects=len(cfg.processor.obj_to_idx),
            num_containers=len(cfg.processor.container_to_idx),
            config=cfg.model,
        )
    else:
        raise ValueError(
            "Unsupported checkpoint feature configuration: expected either image-conditioned "
            "features or env_state-conditioned features."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    flow_algo = FlowMatchingAlgorithm(config=cfg.algo)
    if hasattr(model, "set_flow_algo"):
        model.set_flow_algo(flow_algo)

    processor = RobotProcessor(
        config=cfg.processor,
        feature_config=feature_config,
        preloaded_stats=checkpoint.get("processor_stats", {}),
    )

    return RobotFlowPolicyWrapper(
        model=model,
        processor=processor,
        cameras=list(cfg.dataset.cameras),
        feature_config=feature_config,
    )

