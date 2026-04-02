#!/usr/bin/env python

"""Flow Matching Algorithm Implementation.

Moved from `flow_models.flow.flow_matching` as part of the method-agnostic refactor.

This module provides a modular, extensible flow matching framework integrating
state-of-the-art techniques from recent research in generative modeling and
robot learning.
"""

from abc import ABC, abstractmethod
import warnings
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import Tensor


# =============================================================================
# Flow Paths - Define how we interpolate between noise and data
# =============================================================================

class FlowPath(ABC):
    """Abstract base class for flow interpolation paths.

    A flow path defines:
    1. How to interpolate between x_0 (noise) and x_1 (data): x_t = path(x_0, x_1, t)
    2. The target velocity field: v_target = d(x_t)/dt
    """

    @abstractmethod
    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute x_t along the path."""
        pass

    @abstractmethod
    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target velocity v_t = dx_t/dt."""
        pass

    def expand_t(self, t: Tensor, target_shape: tuple) -> Tensor:
        """Expand time tensor to broadcast with target shape."""
        n_dims = len(target_shape) - 1  # Exclude batch dimension
        return t.view(-1, *([1] * n_dims))


class LinearPath(FlowPath):
    """Linear/Rectified Flow path."""

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        t_expanded = self.expand_t(t, x_0.shape)
        return (1 - t_expanded) * x_0 + t_expanded * x_1

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return x_1 - x_0


class GaussianPath(FlowPath):
    """Gaussian/Conditional Flow Matching path with added noise."""

    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_schedule: Literal["constant", "linear", "sqrt"] = "constant",
    ):
        self.sigma_min = sigma_min
        self.sigma_schedule = sigma_schedule

    def get_sigma(self, t: Tensor) -> Tensor:
        if self.sigma_schedule == "constant":
            return torch.full_like(t, self.sigma_min)
        elif self.sigma_schedule == "linear":
            return self.sigma_min * (1 - t)
        elif self.sigma_schedule == "sqrt":
            return self.sigma_min * torch.sqrt(1 - t + 1e-8)
        else:
            raise ValueError(f"Unknown sigma_schedule: {self.sigma_schedule}")

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        t_expanded = self.expand_t(t, x_0.shape)
        sigma_t = self.expand_t(self.get_sigma(t), x_0.shape)
        epsilon = torch.randn_like(x_0)
        return (1 - t_expanded) * x_0 + t_expanded * x_1 + sigma_t * epsilon

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return x_1 - x_0


class OptimalTransportPath(FlowPath):
    """Optimal Transport Conditional Flow Matching (OT-CFM) path."""

    def __init__(self, reg: float = 0.05, num_iters: int = 50):
        self.reg = reg
        self.num_iters = num_iters

    def compute_ot_plan(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        batch_size = x_0.shape[0]
        x_0_flat = x_0.reshape(batch_size, -1)
        x_1_flat = x_1.reshape(batch_size, -1)
        cost = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2)
        K = torch.exp(-cost / self.reg)
        u = torch.ones(batch_size, device=x_0.device, dtype=x_0.dtype)
        v = torch.ones(batch_size, device=x_0.device, dtype=x_0.dtype)
        for _ in range(self.num_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        perm = plan.argmax(dim=1)
        return perm

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        perm = self.compute_ot_plan(x_0, x_1)
        x_1_matched = x_1[perm]
        t_expanded = self.expand_t(t, x_0.shape)
        return (1 - t_expanded) * x_0 + t_expanded * x_1_matched

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        perm = self.compute_ot_plan(x_0, x_1)
        x_1_matched = x_1[perm]
        return x_1_matched - x_0


class VPPath(FlowPath):
    """Variance-Preserving path connecting flow matching to diffusion."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alpha_sigma(self, t: Tensor) -> tuple[Tensor, Tensor]:
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_t = torch.exp(log_mean_coeff)
        sigma_t = torch.sqrt(1 - alpha_t**2 + 1e-8)
        return alpha_t, sigma_t

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_t = self.expand_t(alpha_t, x_0.shape)
        sigma_t = self.expand_t(sigma_t, x_0.shape)
        return alpha_t * x_1 + sigma_t * x_0

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_t = self.expand_t(alpha_t, x_0.shape)
        sigma_t = self.expand_t(sigma_t, x_0.shape)
        d_alpha = -0.5 * (self.beta_max - self.beta_min) * t - 0.5 * self.beta_min
        d_alpha = self.expand_t(d_alpha * alpha_t.squeeze(), x_0.shape)
        d_sigma = -alpha_t * d_alpha / (sigma_t + 1e-8)
        return d_alpha * x_1 + d_sigma * x_0


# =============================================================================
# Time Schedulers - Define how we sample training times
# =============================================================================

class TimeScheduler(ABC):
    @abstractmethod
    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pass


class UniformScheduler(TimeScheduler):
    def __init__(self, eps: float = 0.0):
        self.eps = eps

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        t = torch.rand(batch_size, device=device, dtype=dtype)
        return t * (1 - 2 * self.eps) + self.eps


class LogitNormalScheduler(TimeScheduler):
    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self.loc = loc
        self.scale = scale

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        u = torch.randn(batch_size, device=device, dtype=dtype)
        u = self.loc + self.scale * u
        return torch.sigmoid(u)


class CosineScheduler(TimeScheduler):
    def __init__(self, s: float = 0.008):
        self.s = s

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        u = torch.rand(batch_size, device=device, dtype=dtype)
        t = torch.acos(1 - 2 * u) / torch.pi
        return t.clamp(self.s, 1 - self.s)


class BetaScheduler(TimeScheduler):
    """Beta-distribution time scheduler biased toward t≈0 (noisy actions).

    Parameterised by the user's formulation:
      u = (t_max - t) / t_max,   u ~ Beta(alpha, beta)
      t = t_max * (1 - u)

    Beta(alpha=1.5, beta=1) density: f(u) ∝ u^0.5, mode at u=1 → t=0.
    CDF: F(u) = u^alpha, so P(t ≤ 0.6 | t_max=0.999) = 1 - 0.4^1.5 ≈ 75%.
    In other words ~75% of training samples land in t ∈ [0, 0.6],
    and only ~25% in t ∈ [0.6, 0.999] — fast drop-off past the midpoint.

    Raising alpha (e.g. 2.0, 3.0) steepens the decay and concentrates
    even more mass near t=0.
    """

    def __init__(self, alpha: float = 1.5, beta: float = 1.0, t_max: float = 1.0):
        self.alpha = alpha
        self.beta  = beta
        self.t_max = t_max

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        a = torch.tensor(self.alpha, device=device, dtype=dtype)
        b = torch.tensor(self.beta,  device=device, dtype=dtype)
        u = torch.distributions.Beta(a, b).sample((batch_size,))  # u = (t_max-t)/t_max ∈ [0,1]
        return (self.t_max * (1.0 - u)).to(dtype)                  # t = t_max*(1-u) ∈ [0, t_max]


class DenseJumpScheduler(TimeScheduler):
    def __init__(self, num_inference_steps: int = 10, density_scale: float = 2.0):
        self.num_inference_steps = num_inference_steps
        self.density_scale = density_scale

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        dt = 1.0 / self.num_inference_steps
        segment = torch.randint(0, self.num_inference_steps, (batch_size,), device=device)
        u = torch.rand(batch_size, device=device, dtype=dtype)
        u = torch.pow(u, 1.0 / self.density_scale)
        if torch.rand(1).item() > 0.5:
            u = 1 - u
        t = (segment.float() + u) * dt
        return t.clamp(0, 1).to(dtype)


# =============================================================================
# ODE Solvers - Define how we integrate the ODE during inference
# =============================================================================

class ODESolver(ABC):
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    @abstractmethod
    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        pass

    def get_time_steps(self, t_start: float, t_end: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(t_start, t_end, self.num_steps + 1, device=device, dtype=dtype)


class EulerSolver(ODESolver):
    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            v = vector_field(x, t_batch)
            x = x + dt * v
        return x


class HeunSolver(ODESolver):
    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            t_next_batch = torch.full((x.shape[0],), t_next.item(), device=x.device, dtype=x.dtype)
            v1 = vector_field(x, t_batch)
            x_pred = x + dt * v1
            v2 = vector_field(x_pred, t_next_batch)
            x = x + dt * 0.5 * (v1 + v2)
        return x


class MidpointSolver(ODESolver):
    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_mid = t + 0.5 * dt
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            t_mid_batch = torch.full((x.shape[0],), t_mid.item(), device=x.device, dtype=x.dtype)
            v1 = vector_field(x, t_batch)
            x_mid = x + 0.5 * dt * v1
            v_mid = vector_field(x_mid, t_mid_batch)
            x = x + dt * v_mid
        return x


class RK4Solver(ODESolver):
    def solve(
        self,
        vector_field: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tensor:
        x = x_0
        timesteps = self.get_time_steps(t_start, t_end, x_0.device, x_0.dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            t_mid_batch = torch.full((x.shape[0],), (t + 0.5 * dt).item(), device=x.device, dtype=x.dtype)
            t_next_batch = torch.full((x.shape[0],), t_next.item(), device=x.device, dtype=x.dtype)
            k1 = vector_field(x, t_batch)
            k2 = vector_field(x + 0.5 * dt * k1, t_mid_batch)
            k3 = vector_field(x + 0.5 * dt * k2, t_mid_batch)
            k4 = vector_field(x + dt * k3, t_next_batch)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


PATH_REGISTRY: dict[str, type[FlowPath]] = {
    "linear": LinearPath,
    "gaussian": GaussianPath,
    "ot": OptimalTransportPath,
    "vp": VPPath,
}
SCHEDULER_REGISTRY: dict[str, type[TimeScheduler]] = {
    "uniform": UniformScheduler,
    "logit_normal": LogitNormalScheduler,
    "cosine": CosineScheduler,
    "dense_jump": DenseJumpScheduler,
    "beta": BetaScheduler,
}
SOLVER_REGISTRY: dict[str, type[ODESolver]] = {
    "euler": EulerSolver,
    "heun": HeunSolver,
    "midpoint": MidpointSolver,
    "rk4": RK4Solver,
}


def create_path(path_type: str, **kwargs) -> FlowPath:
    name = path_type.lower()
    if name not in PATH_REGISTRY:
        raise ValueError(f"Unknown path type: '{path_type}'. Available: {list(PATH_REGISTRY.keys())}")
    return PATH_REGISTRY[name](**kwargs)


def create_scheduler(scheduler_type: str, **kwargs) -> TimeScheduler:
    name = scheduler_type.lower()
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return SCHEDULER_REGISTRY[name](**kwargs)


def create_solver(solver_type: str, num_steps: int = 10, **kwargs) -> ODESolver:
    name = solver_type.lower()
    if name not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver type: '{solver_type}'. Available: {list(SOLVER_REGISTRY.keys())}")
    return SOLVER_REGISTRY[name](num_steps=num_steps, **kwargs)


class FlowMatchingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path_type: str = "linear"
    scheduler_type: str = "uniform"
    solver_type: str = "euler"
    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    velocity_clamp: Optional[float] = None
    sigma_min: float = 1e-4
    sigma_schedule: str = "constant"
    ot_reg: float = 0.05
    time_eps: float = 0.0
    # Upper bound on sampled t. E.g. 0.999 keeps t in [time_eps, 0.999].
    t_max: float = 1.0
    logit_normal_loc: float = 0.0
    logit_normal_scale: float = 1.0
    dense_jump_density: float = 2.0
    # beta: s ~ Beta(alpha, beta), t = t_max * (1 - s) → high density near t=0
    beta_alpha: float = 1.5
    beta_beta: float = 1.0


class FlowMatchingAlgorithm:
    def __init__(self, config: FlowMatchingConfig | None = None):
        if config is None:
            config = FlowMatchingConfig()
        self.config = config

        path_kwargs = {}
        if config.path_type == "gaussian":
            path_kwargs = {"sigma_min": config.sigma_min, "sigma_schedule": config.sigma_schedule}
        elif config.path_type == "ot":
            path_kwargs = {"reg": config.ot_reg}
        self.path = create_path(config.path_type, **path_kwargs)

        scheduler_kwargs = {}
        if config.scheduler_type == "uniform":
            scheduler_kwargs = {"eps": config.time_eps}
        elif config.scheduler_type == "logit_normal":
            scheduler_kwargs = {"loc": config.logit_normal_loc, "scale": config.logit_normal_scale}
        elif config.scheduler_type == "dense_jump":
            scheduler_kwargs = {"num_inference_steps": config.num_inference_steps, "density_scale": config.dense_jump_density}
        elif config.scheduler_type == "beta":
            scheduler_kwargs = {"alpha": config.beta_alpha, "beta": config.beta_beta, "t_max": config.t_max}
        self.scheduler = create_scheduler(config.scheduler_type, **scheduler_kwargs)
        self.solver = create_solver(config.solver_type, num_steps=config.num_inference_steps)

    def compute_loss(
        self,
        vector_field_net: nn.Module,
        x_1: Tensor,
        global_cond: Tensor,
        action_is_pad: Tensor | None = None,
        do_mask_loss_for_padding: bool = False,
    ) -> Tensor:
        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = self.scheduler.sample(batch_size, x_1.device, x_1.dtype)
        if self.config.t_max < 1.0:
            t = t.clamp(max=self.config.t_max)
        x_t = self.path.interpolate(x_0, x_1, t)
        target_v = self.path.velocity(x_0, x_1, t)
        predicted_v = vector_field_net(x_t, t, global_cond=global_cond)
        if self.config.velocity_clamp is not None:
            predicted_v = torch.clamp(predicted_v, -self.config.velocity_clamp, self.config.velocity_clamp)
        loss = F.mse_loss(predicted_v, target_v, reduction="none")
        if do_mask_loss_for_padding:
            if action_is_pad is None:
                raise ValueError("You need to provide 'action_is_pad' when do_mask_loss_for_padding=True")
            in_episode_bound = ~action_is_pad
            loss = loss * in_episode_bound.unsqueeze(-1)
        return loss.mean()

    def generate_samples(
        self,
        vector_field_net: nn.Module,
        initial_noise: Tensor,
        global_cond: Tensor,
        guidance_scale: float | None = None,
        uncond_global_cond: Tensor | None = None,
    ) -> Tensor:
        guidance_scale = guidance_scale or self.config.guidance_scale

        def vector_field(x: Tensor, t: Tensor) -> Tensor:
            v = vector_field_net(x, t, global_cond=global_cond)
            if guidance_scale != 1.0 and uncond_global_cond is not None:
                v_uncond = vector_field_net(x, t, global_cond=uncond_global_cond)
                v = v_uncond + guidance_scale * (v - v_uncond)
            if self.config.velocity_clamp is not None:
                v = torch.clamp(v, -self.config.velocity_clamp, self.config.velocity_clamp)
            return v

        return self.solver.solve(vector_field, initial_noise, t_start=0.0, t_end=1.0)

    def generate_actions(
        self,
        vector_field_net: nn.Module,
        initial_noise: Tensor,
        global_cond: Tensor,
        guidance_scale: float | None = None,
        uncond_global_cond: Tensor | None = None,
    ) -> Tensor:
        warnings.warn(
            "`FlowMatchingAlgorithm.generate_actions()` is deprecated; use `generate_samples()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.generate_samples(
            vector_field_net=vector_field_net,
            initial_noise=initial_noise,
            global_cond=global_cond,
            guidance_scale=guidance_scale,
            uncond_global_cond=uncond_global_cond,
        )


def create_flow_matching(**kwargs) -> FlowMatchingAlgorithm:
    return FlowMatchingAlgorithm(config=FlowMatchingConfig(**kwargs))

