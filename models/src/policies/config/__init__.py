"""
Unified configuration system for training and inference.

The goal is a single RunConfig (JSON) that fully specifies:
  - dataset + dataloader
  - feature schema
  - processor (normalization, task indices)
  - model architecture
  - flow-matching algorithm
  - training loop + logging

Component configs are owned by their respective modules:
  - FlowMatchingConfig   -> policies.algorithms.flow_matching
  - AttentionModelConfig  -> policies.policies.flow_policy
  - ProcessorConfig       -> policies.training.robot_processor
  - FeatureConfig         -> policies.datasets.schema
"""

from policies.config.run_config import (
    LoggingConfig,
    LRScheduleConfig,
    RunConfig,
    TrainConfig,
    WandBConfig,
    load_run_config,
)

__all__ = [
    "LoggingConfig",
    "LRScheduleConfig",
    "RunConfig",
    "TrainConfig",
    "WandBConfig",
    "load_run_config",
]
