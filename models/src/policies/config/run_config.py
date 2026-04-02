"""
Unified run configuration for training and inference.

``RunConfig`` is a pure composition of the component config classes that live
next to their respective modules:

  - ``FlowMatchingConfig``  (``policies.algorithms.flow_matching``)
  - ``AttentionModelConfig`` (``policies.policies.flow_policy``)
  - ``ProcessorConfig``      (``policies.training.robot_processor``)
  - ``FeatureConfig``        (``policies.datasets.schema``)

Plus a few config sections that are specific to the training pipeline itself:

  - ``DatasetSection``  (data paths, cameras, loader, sampling)
  - ``TrainConfig``     (optimizer + training loop)
  - ``LoggingConfig``   (output dir, W&B)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from policies.algorithms.flow_matching import FlowMatchingConfig
from policies.datasets.schema import FeatureConfig
from policies.policies.flow_policy_config import AttentionModelConfig
from policies.training.robot_processor import ProcessorConfig


# ---------------------------------------------------------------------------
# Shared pydantic base
# ---------------------------------------------------------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Dataset section (pipeline-specific, no component owns this)
# ---------------------------------------------------------------------------

class DatasetSplitConfig(_Strict):
    val_fraction: float = 0.05
    split_seed: int = 0


class DatasetSamplingConfig(_Strict):
    k_passes: int = 1
    active_window_shards: int = 2
    locality_block_size: int = 8
    predecode_next_block: bool = True
    frame_cache_max_entries: int = 12_000
    profile_timing: bool = False
    profile_every_samples: int = 500
    log_stall_ms: float = 200.0


class DataLoaderConfig(_Strict):
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 4


class DatasetSection(_Strict):
    data_dir: str
    cameras: Tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    sampling: DatasetSamplingConfig = Field(default_factory=DatasetSamplingConfig)
    loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


# ---------------------------------------------------------------------------
# Training config (optimizer + loop)
# ---------------------------------------------------------------------------

class LRScheduleConfig(_Strict):
    """Learning-rate schedule applied on top of the base lr.

    type:
      "cosine"   – linear warmup then cosine decay to ``min_lr_ratio * lr``
      "constant" – linear warmup then constant lr (no decay)

    warmup_steps: number of steps to linearly ramp from 0 → lr.
    min_lr_ratio: floor for cosine decay expressed as a fraction of ``lr``
                  (ignored when type="constant").
    """
    type: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1


class TrainConfig(_Strict):
    """Optimizer + training-loop settings (single source of truth)."""

    device: str = ""
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = False
    grad_clip_norm: Optional[float] = 1.0
    num_steps: int = 10_000
    num_data_warmup: int = 10
    log_every: int = 50
    eval_every: int = 500
    eval_num_batches: int = 10
    eval_best_of_k: int = 4
    save_every: int = 0
    lr_schedule: LRScheduleConfig = Field(default_factory=LRScheduleConfig)


# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------

class WandBConfig(_Strict):
    enabled: bool = False
    project: str = "robot_flow_training"
    entity: str = ""
    run_name: str = ""


class LoggingConfig(_Strict):
    out_dir: str = "runs/robot_flow_train"
    wandb: WandBConfig = Field(default_factory=WandBConfig)


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------

class RunConfig(_Strict):
    """Single-source-of-truth configuration for training + inference.

    Each component config is imported from the module that owns it:
    - ``feature``    -> ``FeatureConfig``
    - ``algo``       -> ``FlowMatchingConfig``
    - ``model``      -> ``AttentionModelConfig``
    - ``processor``  -> ``ProcessorConfig``
    """

    version: int = 1
    feature: FeatureConfig
    dataset: DatasetSection
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    model: AttentionModelConfig = Field(default_factory=AttentionModelConfig)
    algo: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _validate_cross_section(self) -> "RunConfig":
        if "images" in self.feature:
            img = self.feature["images"]
            missing = [c for c in self.dataset.cameras if c not in img.camera_names]
            if missing:
                raise ValueError(f"dataset.cameras not in image feature camera_names: {missing}")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=False) + "\n"


def load_run_config(path: str | Path) -> RunConfig:
    """Loads and validates a single JSON run config."""
    p = Path(path)
    try:
        return RunConfig.model_validate_json(p.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"Invalid run config at {p}:\n{exc}") from exc
