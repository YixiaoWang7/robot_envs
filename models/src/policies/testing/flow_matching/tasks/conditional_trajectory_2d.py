#!/usr/bin/env python

"""Conditional low-dimensional trajectory task (no images).

Validates:
- `policies.modules.policy_head.MultiModalPolicyHead` conditioning path
- `policies.algorithms.flow_matching.FlowMatchingAlgorithm` on sequence-shaped data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor

from policies.algorithms.flow_matching import FlowMatchingAlgorithm
from policies.modules.policy_head.unet import ModalityConfig, PolicyHeadConfig, MultiModalPolicyHead


class MultiModalPolicyHeadAsVectorField(nn.Module):
    """Adapter: make MultiModalPolicyHead compatible with FlowMatchingAlgorithm.

    FlowMatchingAlgorithm calls: v = net(x_t, t, global_cond=...)
    MultiModalPolicyHead expects: v = head(x_t, t, conditions={...})
    """

    def __init__(self, head: MultiModalPolicyHead, *, condition_name: str):
        super().__init__()
        self.head = head
        self.condition_name = str(condition_name)

    def forward(self, x_t: Tensor, t: Tensor, *, global_cond: Tensor) -> Tensor:
        return self.head(x_t, t, conditions={self.condition_name: global_cond})


def _one_hot(labels: Tensor, num_classes: int) -> Tensor:
    return torch.nn.functional.one_hot(labels.to(torch.long), num_classes=num_classes).to(torch.float32)


@dataclass(frozen=True)
class ConditionalTrajectory2DTask:
    """Generate simple 2D trajectories conditioned on a discrete label.

    Labels map to different trajectory families:
    - 0: circle
    - 1: line
    - 2: spiral
    - 3: zigzag
    """

    name: str = "cond_traj2d"
    horizon: int = 32
    n_classes: int = 4
    noise_std: float = 0.01
    # Class-1 (line) distribution control:
    # Use a single canonical (start, end) pair, then jitter endpoints per-sample to form a *set* of similar lines.
    line_base_start: tuple[float, float] = (-0.7, 0.0)
    line_base_end: tuple[float, float] = (0.7, 0.0)
    line_endpoint_noise_std: float = 0.08

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def cond_dim(self) -> int:
        return self.n_classes

    def _sample_with_labels(
        self,
        labels: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Sample a batch with explicitly provided discrete labels.

        Args:
            labels: Tensor of shape (B,) with values in [0, n_classes).

        Returns:
            (traj, cond_onehot):
              - traj: (B, T, 2)
              - cond_onehot: (B, n_classes)
        """
        B = int(labels.shape[0])
        T = self.horizon

        labels = labels.to(device=device)
        cond = _one_hot(labels, self.n_classes).to(device=device, dtype=dtype)

        s = torch.linspace(0, 1, T, device=device, dtype=dtype)[None, :, None].expand(B, T, 1)  # (B,T,1)
        traj = torch.zeros(B, T, 2, device=device, dtype=dtype)

        # Random phase / parameters per sample (unobserved nuisance variables).
        phase = (2 * torch.pi) * torch.rand(B, 1, 1, device=device, dtype=dtype)
        angle = (2 * torch.pi) * torch.rand(B, 1, 1, device=device, dtype=dtype)
        # For class 1 we want a tight family of similar lines (not all angles/lengths).
        base_start = torch.tensor(self.line_base_start, device=device, dtype=dtype).view(1, 1, 2)
        base_end = torch.tensor(self.line_base_end, device=device, dtype=dtype).view(1, 1, 2)
        jitter_std = float(self.line_endpoint_noise_std)
        start = base_start.expand(B, 1, 2) + jitter_std * torch.randn(B, 1, 2, device=device, dtype=dtype)
        end = base_end.expand(B, 1, 2) + jitter_std * torch.randn(B, 1, 2, device=device, dtype=dtype)

        # Circle/spiral share the same phase.
        theta = phase + (2 * torch.pi) * s

        circle = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) * 0.7
        line = (1 - s) * start + s * end

        r = 0.1 + 0.8 * s
        spiral = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)

        dir_vec = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)  # (B,1,2)
        perp = torch.cat([-dir_vec[..., 1:2], dir_vec[..., 0:1]], dim=-1)
        base = (s - 0.5) * dir_vec * 1.4
        offset = (0.25 * torch.sin(6 * torch.pi * s)) * perp
        zigzag = base + offset

        for k, proto in enumerate([circle, line, spiral, zigzag]):
            mask = labels == k
            if mask.any():
                traj[mask] = proto[mask]

        if self.noise_std > 0:
            traj = traj + self.noise_std * torch.randn_like(traj)

        traj = traj.clamp(-1.0, 1.0)
        return traj, cond

    def sample_batch(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        labels = torch.randint(0, self.n_classes, (batch_size,), device=device)
        return self._sample_with_labels(labels, device=device, dtype=dtype)

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

        # We sample a pool per class and then visualize a "local neighborhood" (similar-but-not-identical).
        n_pool = max(512, n_samples // self.n_classes)
        n_show = 48  # trajectories to plot per class (real + gen), chosen by closeness to an anchor
        T = self.horizon

        fig = plt.figure(figsize=(4 * self.n_classes, 6.8))
        # Row 0: local neighborhood (many similar trajectories) for each class
        # Row 1: loss curve (shared)
        gs = fig.add_gridspec(2, self.n_classes, height_ratios=[4.2, 1.0])

        def wrap_angle(a: Tensor) -> Tensor:
            """Map angle to (-pi, pi]."""
            return torch.atan2(torch.sin(a), torch.cos(a))

        def angle_diff(a: Tensor, b: Tensor) -> Tensor:
            return wrap_angle(a - b).abs()

        def traj_stats(x: Tensor, class_id: int) -> Tensor:
            """Compute per-trajectory stats for similarity search.

            Returns: (B, S) where S depends on class but is fixed per class.
            """
            x0 = x[:, 0, :]  # (B,2)
            x1 = x[:, -1, :]  # (B,2)
            d = x1 - x0
            ang = torch.atan2(d[:, 1], d[:, 0])  # (B,)
            length = torch.linalg.norm(d, dim=-1)  # (B,)

            if class_id in (1, 3):  # line / zigzag: direction + start/center/length
                center = 0.5 * (x0 + x1)
                return torch.stack([ang, length, center[:, 0], center[:, 1]], dim=-1)  # (B,4)
            if class_id == 0:  # circle: start angle on circle
                ang0 = torch.atan2(x0[:, 1], x0[:, 0])
                return torch.stack([ang0], dim=-1)  # (B,1)
            if class_id == 2:  # spiral: start angle + end radius
                ang0 = torch.atan2(x0[:, 1], x0[:, 0])
                r_end = torch.linalg.norm(x1, dim=-1)
                return torch.stack([ang0, r_end], dim=-1)  # (B,2)
            raise ValueError(f"Unknown class_id={class_id}")

        def neighborhood_indices(stats: Tensor, anchor: Tensor, class_id: int, k: int) -> Tensor:
            """Pick indices of k items closest to anchor in a class-aware metric."""
            if stats.ndim != 2:
                raise ValueError("stats must be (B,S)")
            if anchor.ndim != 1:
                raise ValueError("anchor must be (S,)")

            if class_id in (1, 3):  # [ang, length, cx, cy]
                ang_d = angle_diff(stats[:, 0], anchor[0])
                d_len = (stats[:, 1] - anchor[1]).abs()
                d_c = torch.linalg.norm(stats[:, 2:4] - anchor[2:4], dim=-1)
                dist = (ang_d / torch.pi) + 0.5 * d_len + d_c
            elif class_id == 0:  # [start_angle]
                dist = angle_diff(stats[:, 0], anchor[0]) / torch.pi
            elif class_id == 2:  # [start_angle, r_end]
                ang_d = angle_diff(stats[:, 0], anchor[0]) / torch.pi
                r_d = (stats[:, 1] - anchor[1]).abs()
                dist = ang_d + 0.5 * r_d
            else:
                raise ValueError(f"Unknown class_id={class_id}")

            # smallest distances
            return torch.topk(dist, k=min(k, dist.shape[0]), largest=False).indices

        for k in range(self.n_classes):
            labels = torch.full((n_pool,), k, device=device, dtype=torch.long)
            x_real, cond = self._sample_with_labels(labels, device=device, dtype=dtype)

            x0 = torch.randn(n_pool, T, 2, device=device, dtype=dtype)
            x_gen = algo.generate_samples(vector_field_net=vector_field_net, initial_noise=x0, global_cond=cond)

            # Choose an anchor from the real pool, then visualize neighbors around it
            stats_real = traj_stats(x_real, k)
            stats_gen = traj_stats(x_gen, k)
            anchor_idx = torch.randint(0, n_pool, (1,), device=device).item()
            anchor = stats_real[anchor_idx]

            # For class 1 (line), the global distribution can be very broad (many angles/lengths),
            # which makes it hard to visually compare "in-distribution" vs "out-of-distribution".
            # We instead visualize a *slice* of the distribution around a canonical anchor:
            # a roughly horizontal, mid-length line near the origin.
            if k == 1 and anchor.numel() == 4:
                anchor = torch.tensor([0.0, 1.4, 0.0, 0.0], device=device, dtype=stats_real.dtype)

            idx_r = neighborhood_indices(stats_real, anchor, k, n_show)
            idx_g = neighborhood_indices(stats_gen, anchor, k, n_show)

            ax = fig.add_subplot(gs[0, k])
            for i in idx_r[:n_show]:
                ax.plot(
                    x_real[i, :, 0].cpu(),
                    x_real[i, :, 1].cpu(),
                    color="tab:blue",
                    alpha=0.18,
                    linewidth=1.0,
                )
            for i in idx_g[:n_show]:
                ax.plot(
                    x_gen[i, :, 0].cpu(),
                    x_gen[i, :, 1].cpu(),
                    color="tab:orange",
                    alpha=0.18,
                    linewidth=1.0,
                )

            # Highlight the anchor (real)
            ax.plot(
                x_real[anchor_idx, :, 0].cpu(),
                x_real[anchor_idx, :, 1].cpu(),
                color="tab:blue",
                alpha=0.9,
                linewidth=2.0,
            )

            if k == 1 and anchor.numel() == 4:
                ang_deg = float(anchor[0].detach().cpu().numpy() * 180.0 / torch.pi)
                ax.set_title(
                    f"class {k}: local neighborhood\n(anchor ~ {ang_deg:.0f}° line, len~{float(anchor[1]):.1f})"
                )
            else:
                ax.set_title(f"class {k}: local neighborhood (real=blue, gen=orange)")
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


def build_conditional_policy_head(*, horizon: int, n_classes: int) -> MultiModalPolicyHeadAsVectorField:
    cfg = PolicyHeadConfig(
        action_dim=2,
        horizon=horizon,
        modalities=[ModalityConfig(name="label", dim=n_classes)],
        down_dims=(64, 128, 256),
        kernel_size=5,
        n_groups=8,
        use_film_scale_modulation=True,
        film_fusion="gated",
    )
    head = MultiModalPolicyHead(cfg)
    return MultiModalPolicyHeadAsVectorField(head, condition_name="label")

