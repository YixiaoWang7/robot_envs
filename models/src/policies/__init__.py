"""
Method-agnostic policy library.

This package is organized around:
- **datasets**: dataloaders + dataset schemas
- **modules**: reusable neural network components (vision encoders, heads, etc.)
- **algorithms**: training/inference algorithms (flow matching, diffusion, AR, regression)
- **policies**: end-to-end policy models combining modules + algorithms
"""

from policies.datasets import RobotDataset, create_robot_dataloader

__all__ = [
    "RobotDataset",
    "create_robot_dataloader",
]

