#!/usr/bin/env python

"""Goal-conditioned low-dimensional trajectory task (no images).

Validates:
- `policies.modules.policy_head.FiLMConvPolicyHead` (global_cond path)
- Flow matching on moderately structured low-dim sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor

from policies.algorithms.flow_matching import FlowMatchingAlgorithm
from policies.modules.policy_head.unet import FiLMConvPolicyHead


@dataclass(frozen=True)
class GoalConditionedTrajectory2DTask:
    """Generate smooth 2D trajectories conditioned on (start, goal).

    Each sample is a curve from start -> goal with a random perpendicular "bump".
    Conditioning: global_cond = concat([start_xy, goal_xy]) with shape (B, 4).
    """

    name: str = "goal_traj2d"
    horizon: int = 32
    bump_scale: float = 0.35
    noise_std: float = 0.01

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def cond_dim(self) -> int:
        return 4

    def _sample_with_start_goal(
        self,
        start: Tensor,
        goal: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Sample trajectories for fixed (start, goal), varying bump/noise.

        Args:
            start: (B,2)
            goal:  (B,2)
        Returns:
            (traj, global_cond) where:
              - traj: (B,T,2)
              - global_cond: (B,4) = concat(start, goal)
        """
        if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
            raise ValueError(f"start/goal must both be (B,2), got {tuple(start.shape)} and {tuple(goal.shape)}")

        B, T = start.shape[0], self.horizon
        device = start.device

        s = torch.linspace(0, 1, T, device=device, dtype=dtype)[None, :, None].expand(B, T, 1)  # (B,T,1)
        base = (1 - s) * start[:, None, :] + s * goal[:, None, :]

        d = (goal - start)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        perp = torch.stack([-d[:, 1], d[:, 0]], dim=-1)  # (B,2)
        amp = (torch.rand(B, 1, device=device, dtype=dtype) * 2 - 1.0) * self.bump_scale
        bump = amp[:, None, :] * torch.sin(torch.pi * s) * perp[:, None, :]

        traj = base + bump
        if self.noise_std > 0:
            traj = traj + self.noise_std * torch.randn_like(traj)
        traj = traj.clamp(-1.0, 1.0)

        global_cond = torch.cat([start, goal], dim=-1)
        return traj, global_cond

    def sample_batch(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        B, T = batch_size, self.horizon
        start = (torch.rand(B, 2, device=device, dtype=dtype) * 1.6) - 0.8
        goal = (torch.rand(B, 2, device=device, dtype=dtype) * 1.6) - 0.8
        return self._sample_with_start_goal(start, goal, dtype=dtype)

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

        # Visualize a few fixed-condition slices (so we don't mix many different start/goal pairs).
        anchors = [
            (torch.tensor([-0.7, -0.5], device=device, dtype=dtype), torch.tensor([0.7, 0.5], device=device, dtype=dtype)),
            (torch.tensor([-0.7, 0.0], device=device, dtype=dtype), torch.tensor([0.7, 0.0], device=device, dtype=dtype)),
            (torch.tensor([-0.3, 0.7], device=device, dtype=dtype), torch.tensor([0.6, -0.6], device=device, dtype=dtype)),
        ]

        n_per_anchor = max(24, min(96, n_samples // max(1, len(anchors))))
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, len(anchors), height_ratios=[3.2, 1.0])

        for j, (s0, g0) in enumerate(anchors):
            start = s0.view(1, 2).expand(n_per_anchor, 2).contiguous()
            goal = g0.view(1, 2).expand(n_per_anchor, 2).contiguous()
            x_real, cond = self._sample_with_start_goal(start, goal, dtype=dtype)

            x0 = torch.randn(n_per_anchor, self.horizon, 2, device=device, dtype=dtype)
            x_gen = algo.generate_samples(vector_field_net=vector_field_net, initial_noise=x0, global_cond=cond)

            ax = fig.add_subplot(gs[0, j])
            for i in range(n_per_anchor):
                ax.plot(x_real[i, :, 0].cpu(), x_real[i, :, 1].cpu(), color="tab:blue", alpha=0.15, linewidth=1)
                ax.plot(x_gen[i, :, 0].cpu(), x_gen[i, :, 1].cpu(), color="tab:orange", alpha=0.15, linewidth=1)
            ax.scatter([float(s0[0].cpu())], [float(s0[1].cpu())], c="green", s=30, marker="o")
            ax.scatter([float(g0[0].cpu())], [float(g0[1].cpu())], c="red", s=30, marker="o")
            ax.set_title(f"anchor {j}: start→goal slice")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

        ax_loss = fig.add_subplot(gs[1, :])
        ax_loss.plot(loss_history, linewidth=1.0)
        ax_loss.set_title("training loss (MSE)")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("mse")
        ax_loss.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(out_dir / f"viz_step_{step:06d}.png", dpi=160)
        plt.close(fig)


def build_goal_conditioned_film_head(*, horizon: int) -> FiLMConvPolicyHead:
    return FiLMConvPolicyHead(
        action_dim=2,
        horizon=horizon,
        global_cond_dim=4,
        n_obs_steps=1,
        time_embed_dim=128,
        down_dims=(64, 128, 256),
        kernel_size=5,
        n_groups=8,
        use_film_scale_modulation=True,
    )

