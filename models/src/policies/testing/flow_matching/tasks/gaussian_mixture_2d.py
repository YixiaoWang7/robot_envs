#!/usr/bin/env python

"""A simple 2D Gaussian-mixture task for flow-matching sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor

from policies.algorithms.flow_matching import FlowMatchingAlgorithm


class MLPVectorField(nn.Module):
    """Tiny vector field network v_theta(x_t, t, cond) -> velocity with same shape as x_t.

    Works for x_t shaped (B, D) or (B, ..., D) as long as the last dim is D.
    """

    def __init__(self, data_dim: int, cond_dim: int = 0, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.data_dim = int(data_dim)
        self.cond_dim = int(cond_dim)
        in_dim = self.data_dim + 1 + self.cond_dim  # +1 for time

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        layers.append(nn.Linear(d, self.data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: Tensor, t: Tensor, *, global_cond: Tensor) -> Tensor:
        orig_shape = x_t.shape
        x = x_t.reshape(x_t.shape[0], -1, self.data_dim)  # (B, N, D)

        # Broadcast time and cond across token/sequence axis.
        t_in = t.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)  # (B, N, 1)
        if global_cond.ndim != 2:
            raise ValueError(f"global_cond must have shape (B, Ccond), got {tuple(global_cond.shape)}")
        c = global_cond[:, None, :].expand(x.shape[0], x.shape[1], global_cond.shape[1])  # (B, N, Ccond)

        inp = torch.cat([x, t_in, c], dim=-1)  # (B, N, D+1+Ccond)
        v = self.net(inp)  # (B, N, D)
        return v.reshape(orig_shape)


@dataclass(frozen=True)
class GaussianMixture2DTask:
    """Unconditional 2D Gaussian mixture (4 modes) for quick visualization."""

    name: str = "gmm2d"
    std: float = 0.20
    means: tuple[tuple[float, float], ...] = ((-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0))

    @property
    def data_dim(self) -> int:
        return 2

    @property
    def cond_dim(self) -> int:
        return 0

    def sample_batch(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        k = len(self.means)
        idx = torch.randint(0, k, (batch_size,), device=device)
        means = torch.tensor(self.means, device=device, dtype=dtype)  # (K, 2)
        x = means[idx] + self.std * torch.randn(batch_size, 2, device=device, dtype=dtype)
        global_cond = torch.zeros(batch_size, 0, device=device, dtype=dtype)
        return x, global_cond

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
        out_dir.mkdir(parents=True, exist_ok=True)

        # Real samples (for plotting).
        x_real, cond = self.sample_batch(n_samples, device=device, dtype=dtype)

        # Generated samples.
        x0 = torch.randn_like(x_real)
        x_gen = algo.generate_samples(vector_field_net=vector_field_net, initial_noise=x0, global_cond=cond)

        real = x_real.detach().cpu()
        gen = x_gen.detach().cpu()

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.scatter(real[:, 0], real[:, 1], s=5, alpha=0.5, label="real")
        ax1.scatter(gen[:, 0], gen[:, 1], s=5, alpha=0.5, label="generated")
        ax1.set_title(f"{self.name}: real vs generated (step {step})")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc="upper right")

        ax2.plot(loss_history, linewidth=1.0)
        ax2.set_title("training loss")
        ax2.set_xlabel("step")
        ax2.set_ylabel("mse")
        ax2.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(out_dir / f"viz_step_{step:06d}.png", dpi=160)
        plt.close(fig)

