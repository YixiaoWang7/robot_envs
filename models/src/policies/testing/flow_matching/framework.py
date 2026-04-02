#!/usr/bin/env python

"""A small, readable framework for flow-matching sanity checks.

Goals:
- Keep the runner minimal and easy to extend for new tasks (low-dim, conditional, image).
- Avoid coupling to robot datasets/training code.
- Provide a consistent interface for:
  - sampling batches (x_1, global_cond)
  - training a vector field model via FlowMatchingAlgorithm
  - periodic visualization hooks
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from policies.algorithms.flow_matching import FlowMatchingAlgorithm, FlowMatchingConfig


class FlowMatchingTask(Protocol):
    """A task provides data samples and (optionally) visualization."""

    name: str

    def sample_batch(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return (x_1, global_cond).

        - x_1: target data samples, arbitrary shape (B, ...)
        - global_cond: conditioning tensor, shape (B, Ccond) (use Ccond=0 for unconditional)
        """

    def on_batch(self, *, vector_field_net: nn.Module, x_1: Tensor, global_cond: Tensor) -> None:
        """Optional hook called after sampling each batch.

        Useful for tasks where the vector field net needs extra per-batch context
        that should not be packed into `global_cond` (e.g., images for an encoder).
        """

    @torch.no_grad()
    def visualize(
        self,
        *,
        out_dir: Path,
        step: int,
        algo: FlowMatchingAlgorithm,
        vector_field_net: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        n_samples: int,
        loss_history: list[float],
    ) -> None:
        """Optional visualization hook.

        Implementations should save figures/artifacts into out_dir.
        """


@dataclass(frozen=True)
class ExperimentConfig:
    # Output / reproducibility
    out_dir: Path
    seed: int = 0

    # Training
    steps: int = 2_000
    batch_size: int = 512
    lr: float = 1e-3

    # Logging / visualization
    log_every: int = 50
    viz_every: int = 200
    n_viz_samples: int = 4_096

    # Algo config (stored + used to construct FlowMatchingAlgorithm)
    flow: FlowMatchingConfig = field(
        default_factory=lambda: FlowMatchingConfig(
            path_type="linear",
            scheduler_type="uniform",
            solver_type="euler",
            num_inference_steps=50,
            time_eps=1e-3,
        )
    )


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlowMatchingExperiment:
    """Train and sample with FlowMatchingAlgorithm on a task."""

    def __init__(
        self,
        *,
        task: FlowMatchingTask,
        vector_field_net: nn.Module,
        config: ExperimentConfig,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.task = task
        self.vector_field_net = vector_field_net
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.out_dir = _ensure_dir(Path(config.out_dir))
        self.algo = FlowMatchingAlgorithm(config=config.flow)

        self.vector_field_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.vector_field_net.parameters(), lr=config.lr)

        self.loss_history: list[float] = []

        # Write config once for reproducibility
        _ensure_dir(self.out_dir)
        (self.out_dir / "task.txt").write_text(f"{self.task.name}\n")
        cfg = asdict(config, dict_factory=lambda pairs: {k: v for k, v in pairs})
        cfg["out_dir"] = str(cfg["out_dir"])
        cfg["flow"] = config.flow.model_dump()
        (self.out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    def _initial_noise_like(self, x_1: Tensor, n: int) -> Tensor:
        shape = (n, *x_1.shape[1:])
        return torch.randn(shape, device=self.device, dtype=self.dtype)

    def train(self) -> Path:
        set_seed(self.config.seed)
        self.vector_field_net.train()

        for step in range(1, self.config.steps + 1):
            x_1, global_cond = self.task.sample_batch(
                self.config.batch_size, device=self.device, dtype=self.dtype
            )
            # Optional per-batch hook (e.g., to set images on the vector field net).
            on_batch = getattr(self.task, "on_batch", None)
            if callable(on_batch):
                on_batch(vector_field_net=self.vector_field_net, x_1=x_1, global_cond=global_cond)

            loss = self.algo.compute_loss(
                vector_field_net=self.vector_field_net,
                x_1=x_1,
                global_cond=global_cond,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(float(loss.item()))

            if step % self.config.log_every == 0 or step == 1:
                recent = self.loss_history[-self.config.log_every :]
                mean_loss = sum(recent) / len(recent)
                print(f"[{self.task.name}] step={step:6d}  loss={mean_loss:.6f}")

            if self.config.viz_every > 0 and (step % self.config.viz_every == 0 or step == self.config.steps):
                try:
                    self.task.visualize(
                        out_dir=self.out_dir,
                        step=step,
                        algo=self.algo,
                        vector_field_net=self.vector_field_net,
                        device=self.device,
                        dtype=self.dtype,
                        n_samples=self.config.n_viz_samples,
                        loss_history=self.loss_history,
                    )
                except NotImplementedError:
                    pass

        ckpt_path = self.out_dir / "vector_field_net.pt"
        torch.save({"state_dict": self.vector_field_net.state_dict()}, ckpt_path)
        return ckpt_path

