#!/usr/bin/env python
"""Env-state token conditioned flow matching policy (no images).

This is the state-based analogue of `FlowMatchingAttentionNetwork`, but instead of
pooling DINO image patch tokens, it pools *tokenized env_state* (object poses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from policies.modules.fusion.attention_projector import CrossAttentionPooling
from policies.modules.policy_head.unet import FiLMConvPolicyHead
from policies.modules.task.learnable_query import LearnableQueryTokens
from policies.policies.flow_policy_config import AttentionModelConfig


class FlowMatchingEnvStateAttentionNetwork(nn.Module):
    """Flow matching network with query-conditioned attention pooling over env_state tokens."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        robot_state_dim: int,
        env_state_dim: int,
        n_obs_steps: int = 1,
        num_objects: int = 3,
        num_containers: int = 3,
        config: AttentionModelConfig | None = None,
    ):
        super().__init__()
        if config is None:
            config = AttentionModelConfig()
        self._config = config

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.robot_state_dim = int(robot_state_dim)
        self.env_state_dim = int(env_state_dim)
        self.n_obs_steps = int(n_obs_steps)

        self.num_objects = int(num_objects)
        self.num_containers = int(num_containers)
        if int(config.num_query_tokens) != 2:
            raise ValueError(
                "FlowMatchingEnvStateAttentionNetwork expects exactly 2 query tokens: "
                "one for object_id and one for container_id."
            )
        self.num_query_tokens = 2
        self.pooled_dim = int(config.pooled_dim)
        self.embed_dim = int(config.env_token_embed_dim)

        # Tokenization config
        self.env_num_entities = int(config.env_num_entities)
        self.env_entity_dim = int(config.env_entity_dim)
        if self.env_num_entities <= 0 or self.env_entity_dim <= 0:
            raise ValueError("env_num_entities and env_entity_dim must be positive")
        base = self.env_num_entities * self.env_entity_dim
        if self.env_state_dim < base:
            raise ValueError(
                f"env_state_dim={self.env_state_dim} is smaller than "
                f"env_num_entities*env_entity_dim={base}."
            )

        self.env_entity_proj = nn.Sequential(
            nn.LayerNorm(self.env_entity_dim),
            nn.Linear(self.env_entity_dim, self.embed_dim),
        )
        # One learnable vector per entity slot so the cross-attention can
        # structurally distinguish slot 0 from slot 1 … N regardless of pose.
        self.entity_slot_emb = nn.Embedding(self.env_num_entities, self.embed_dim)

        self.obj_query = LearnableQueryTokens(num_query_tokens=1, dim=self.embed_dim, num_tasks=self.num_objects)
        self.cont_query = LearnableQueryTokens(num_query_tokens=1, dim=self.embed_dim, num_tasks=self.num_containers)

        self.fusion = CrossAttentionPooling(
            input_shape=(self.embed_dim, 1, 1),
            embed_dim=self.embed_dim,
            out_dim=self.pooled_dim,
            num_layers=config.fusion_num_layers,
            num_heads=config.fusion_num_heads,
            ff_dim=config.fusion_ff_dim,
            dropout=config.fusion_dropout,
            mask_type=config.fusion_mask_type,
        )

        global_cond_dim = self.robot_state_dim + (self.num_query_tokens * self.pooled_dim)
        self.policy_head = FiLMConvPolicyHead(
            action_dim=self.action_dim,
            horizon=self.horizon,
            global_cond_dim=global_cond_dim,
            n_obs_steps=self.n_obs_steps,
            time_embed_dim=config.time_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            use_film_scale_modulation=config.use_film_scale_modulation,
        )

        self.flow_algo = None
        self._architecture_config: dict[str, Any] = {
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "robot_state_dim": self.robot_state_dim,
            "env_state_dim": self.env_state_dim,
            "n_obs_steps": self.n_obs_steps,
            "num_objects": self.num_objects,
            "num_containers": self.num_containers,
            "config": config.model_dump(),
        }

    def set_flow_algo(self, flow_algo):
        self.flow_algo = flow_algo
        return self

    def _tokenize_env(self, env_state_flat: Tensor) -> Tensor:
        """Tokenize env_state: (Bflat, Denv) -> (Bflat, N, embed_dim)."""
        Bflat = int(env_state_flat.shape[0])
        base = self.env_num_entities * self.env_entity_dim
        entities = env_state_flat[:, :base].reshape(Bflat, self.env_num_entities, self.env_entity_dim)
        ent_tokens = self.env_entity_proj(entities)  # (Bflat, N, D)

        slot_ids = torch.arange(self.env_num_entities, device=env_state_flat.device)
        ent_tokens = ent_tokens + self.entity_slot_emb(slot_ids).unsqueeze(0)  # (1, N, D) broadcast
        return ent_tokens

    def encode_observations(
        self,
        *,
        robot_state: Tensor,
        env_state: Tensor,
        task_indices: Tensor | None,
    ) -> Tensor:
        # robot_state: (B, n_obs_steps, Dr)
        B, n_obs_steps = int(robot_state.shape[0]), int(robot_state.shape[1])
        if n_obs_steps != self.n_obs_steps:
            raise ValueError(f"Expected n_obs_steps={self.n_obs_steps}, got {n_obs_steps}")

        # env_state: (B, n_obs_steps, Denv)
        if env_state.ndim != 3:
            raise ValueError(f"Expected env_state to have shape (B,n_obs_steps,D), got {tuple(env_state.shape)}")
        if int(env_state.shape[0]) != B or int(env_state.shape[1]) != n_obs_steps or int(env_state.shape[2]) != self.env_state_dim:
            raise ValueError(
                f"env_state shape mismatch: got {tuple(env_state.shape)}, expected (B={B},n_obs_steps={n_obs_steps},Denv={self.env_state_dim})"
            )

        if task_indices is None:
            raise ValueError("task_indices is required (object_id, container_id)")
        if task_indices.ndim != 2 or int(task_indices.shape[1]) != 2:
            raise ValueError(f"task_indices must have shape (B,2) or (B*n_obs_steps,2), got {tuple(task_indices.shape)}")

        # Flatten per-step.
        Bflat = B * n_obs_steps
        env_flat = env_state.reshape(Bflat, self.env_state_dim)
        env_tokens = self._tokenize_env(env_flat)  # (Bflat, Tk, embed_dim)

        # Build per-step task indices (Bflat,2).
        if int(task_indices.shape[0]) == B:
            task_step = task_indices.repeat_interleave(n_obs_steps, dim=0)
        elif int(task_indices.shape[0]) == Bflat:
            task_step = task_indices
        else:
            raise ValueError(f"task_indices batch mismatch: got {int(task_indices.shape[0])}, expected {B} or {Bflat}")

        obj = task_step[:, 0].to(device=env_tokens.device, dtype=torch.long)
        cont = task_step[:, 1].to(device=env_tokens.device, dtype=torch.long)
        if (obj < 0).any() or (obj >= self.num_objects).any():
            raise ValueError("object_id out of range")
        if (cont < 0).any() or (cont >= self.num_containers).any():
            raise ValueError("container_id out of range")

        q_obj = self.obj_query(batch_size=Bflat, device=env_tokens.device, dtype=env_tokens.dtype, task_ids=obj)
        q_cont = self.cont_query(batch_size=Bflat, device=env_tokens.device, dtype=env_tokens.dtype, task_ids=cont)
        q = torch.cat([q_obj, q_cont], dim=1)  # (Bflat, 2, embed_dim)

        pooled = self.fusion(env_tokens, query_tokens=q)  # (Bflat, 2, pooled_dim)
        env_feat_step = pooled.flatten(start_dim=1)  # (Bflat, 2*pooled_dim)
        env_feat = env_feat_step.reshape(B, n_obs_steps, -1)  # (B, n_obs_steps, 2*pooled_dim)

        return torch.cat([robot_state, env_feat], dim=-1)

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        return self.policy_head(x, timestep, global_cond)

    def compute_loss(
        self,
        actions: Tensor,
        robot_state: Tensor,
        env_state: Tensor,
        task_indices: Tensor,
        flow_algo=None,
    ) -> Tensor:
        if flow_algo is None:
            flow_algo = self.flow_algo
        if flow_algo is None:
            raise ValueError("flow_algo must be provided or set via set_flow_algo()")

        global_cond = self.encode_observations(robot_state=robot_state, env_state=env_state, task_indices=task_indices)

        def vector_field_net(x_t, t, global_cond):
            return self(x_t, t, global_cond)

        return flow_algo.compute_loss(vector_field_net=vector_field_net, x_1=actions, global_cond=global_cond)

    def generate_actions(
        self,
        *,
        robot_state: Tensor,
        env_state: Tensor,
        task_indices: Tensor,
        batch_size: int | None = None,
        flow_algo=None,
        guidance_scale: float | None = None,
        uncond_task_indices: Tensor | None = None,
    ) -> Tensor:
        if flow_algo is None:
            flow_algo = self.flow_algo
        if flow_algo is None:
            raise ValueError("flow_algo must be provided or set via set_flow_algo()")

        if batch_size is None:
            batch_size = int(robot_state.shape[0])

        global_cond = self.encode_observations(robot_state=robot_state, env_state=env_state, task_indices=task_indices)
        device = robot_state.device
        initial_noise = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=robot_state.dtype)

        def vector_field_net(x_t, t, global_cond):
            return self(x_t, t, global_cond)

        uncond_global_cond = None
        if uncond_task_indices is not None:
            uncond_global_cond = self.encode_observations(
                robot_state=robot_state, env_state=env_state, task_indices=uncond_task_indices
            )

        return flow_algo.generate_samples(
            vector_field_net=vector_field_net,
            initial_noise=initial_noise,
            global_cond=global_cond,
            guidance_scale=guidance_scale,
            uncond_global_cond=uncond_global_cond,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer_state_dict: dict | None = None,
        epoch: int | None = None,
        loss: float | None = None,
        extra_info: dict[str, Any] | None = None,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"model_state_dict": self.state_dict(), "architecture_config": self._architecture_config}

        if self.flow_algo is not None:
            checkpoint["flow_config"] = self.flow_algo.config.model_dump()

        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if loss is not None:
            checkpoint["loss"] = loss
        if extra_info is not None:
            checkpoint["extra_info"] = extra_info

        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        load_flow_algo: bool = True,
    ) -> tuple["FlowMatchingEnvStateAttentionNetwork", dict]:
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        arch = dict(checkpoint["architecture_config"])
        if "config" in arch and isinstance(arch["config"], dict):
            arch["config"] = AttentionModelConfig(**arch["config"])
        model = cls(**arch)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        if load_flow_algo and "flow_config" in checkpoint:
            from policies.algorithms.flow_matching import FlowMatchingAlgorithm, FlowMatchingConfig

            flow_config = FlowMatchingConfig(**checkpoint["flow_config"])
            model.set_flow_algo(FlowMatchingAlgorithm(flow_config))

        extra_dict = {
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "epoch": checkpoint.get("epoch"),
            "loss": checkpoint.get("loss"),
            "extra_info": checkpoint.get("extra_info"),
            "flow_config": checkpoint.get("flow_config"),
        }
        return model, extra_dict

