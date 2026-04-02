#!/usr/bin/env python

"""Integrated Flow Matching policy network.

Moved from `flow_models.models.flow_network` as part of the method-agnostic refactor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from policies.modules.vision.dino import DinoImageEncoder
from policies.modules.policy_head.unet import FiLMConvPolicyHead
from policies.modules.fusion.pooling import SpatialSoftmax
from policies.modules.fusion.attention_projector import CrossAttentionPooling
from policies.modules.task.learnable_query import LearnableQueryTokens
from policies.policies.flow_policy_config import AttentionModelConfig


class FlowMatchingNetwork(nn.Module):
    """Complete flow matching network for robot manipulation.

    Combines:
    - Image encoder: processes visual observations
    - Policy head: predicts velocity field for flow matching
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        robot_state_dim: int = 9,
        n_obs_steps: int = 1,
        vision_encoder: str = "dino",
        vision_model_size: str = "small",
        input_shape: tuple[int, int, int] = (3, 224, 224),
        crop_shape: tuple[int, int] | None = (224, 224),
        crop_is_random: bool = True,
        freeze_vision_backbone: bool = True,
        spatial_softmax_num_keypoints: int = 64,
        num_cameras: int = 2,
        use_separate_pooling_per_camera: bool = True,
        num_obj_queries: int = 3,
        num_container_queries: int = 3,
        policy_head_type: str = "conv",
        time_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
        mlp_hidden_dims: tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.robot_state_dim = robot_state_dim
        self.n_obs_steps = n_obs_steps
        self.num_cameras = num_cameras
        self.policy_head_type = policy_head_type
        self.use_separate_pooling_per_camera = use_separate_pooling_per_camera

        self.image_encoder = DinoImageEncoder(
            model_size=vision_model_size,
            input_shape=input_shape,
            crop_shape=crop_shape,
            crop_is_random=crop_is_random,
            freeze_backbone=freeze_vision_backbone,
        )

        feature_map_shape = (self.image_encoder.model_dim, *self.image_encoder.feature_map_hw)
        if use_separate_pooling_per_camera:
            self.image_projector = nn.ModuleList(
                [
                    SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)
                    for _ in range(num_cameras)
                ]
            )
        else:
            self.image_projector = SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)

        self.image_feature_dim = spatial_softmax_num_keypoints * 2

        if use_separate_pooling_per_camera:
            effective_image_feature_dim = self.image_feature_dim
        else:
            effective_image_feature_dim = self.image_feature_dim * num_cameras

        global_cond_dim = robot_state_dim + effective_image_feature_dim

        if policy_head_type == "conv":
            self.policy_head = FiLMConvPolicyHead(
                action_dim=action_dim,
                horizon=horizon,
                global_cond_dim=global_cond_dim,
                n_obs_steps=n_obs_steps,
                time_embed_dim=time_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                use_film_scale_modulation=use_film_scale_modulation,
            )
        else:
            raise ValueError(f"Unknown policy head type: {policy_head_type}")

        self.flow_algo = None

        self._architecture_config = {
            "action_dim": action_dim,
            "horizon": horizon,
            "robot_state_dim": robot_state_dim,
            "n_obs_steps": n_obs_steps,
            "vision_encoder": vision_encoder,
            "vision_model_size": vision_model_size,
            "input_shape": input_shape,
            "crop_shape": crop_shape,
            "crop_is_random": crop_is_random,
            "freeze_vision_backbone": freeze_vision_backbone,
            "spatial_softmax_num_keypoints": spatial_softmax_num_keypoints,
            "num_cameras": num_cameras,
            "use_separate_pooling_per_camera": use_separate_pooling_per_camera,
            "num_obj_queries": num_obj_queries,
            "num_container_queries": num_container_queries,
            "policy_head_type": policy_head_type,
            "time_embed_dim": time_embed_dim,
            "down_dims": down_dims,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
            "mlp_hidden_dims": mlp_hidden_dims,
        }

    def set_flow_algo(self, flow_algo):
        self.flow_algo = flow_algo
        return self

    def encode_observations(
        self,
        robot_state: Tensor,
        images: Tensor,
        task_indices: Tensor | None = None,
    ) -> Tensor:
        batch_size, n_obs_steps = robot_state.shape[:2]

        images_flat = images.reshape(-1, *images.shape[-3:])
        patch_tokens = self.image_encoder(images_flat, output="patch")

        # NOTE: `task_indices` is currently unused for this image encoder+pooling path.
        # (Task-conditioning for vision can be added by generating query tokens in `policies.modules.task`
        # and using attention-based fusion modules.)
        _ = task_indices

        Bflat, L, D = patch_tokens.shape
        Hf, Wf = self.image_encoder.feature_map_hw
        if int(L) != int(Hf * Wf):
            raise ValueError(
                f"Unexpected patch token length L={int(L)}; expected Hf*Wf={int(Hf*Wf)} "
                f"from encoder.feature_map_hw={self.image_encoder.feature_map_hw}."
            )
        if Bflat % self.num_cameras != 0:
            raise ValueError(f"Expected flattened image batch Bflat={Bflat} divisible by num_cameras={self.num_cameras}.")

        effective_batch = Bflat // self.num_cameras  # == batch_size * n_obs_steps
        toks = patch_tokens.reshape(effective_batch, self.num_cameras, L, D)  # (B*n_obs_steps, Nc, L, D)

        def _tokens_to_feature_map(t: Tensor) -> Tensor:
            # t: (B, L, D) -> (B, D, Hf, Wf)
            return t.reshape(t.shape[0], Hf, Wf, D).permute(0, 3, 1, 2).contiguous()

        if self.use_separate_pooling_per_camera:
            per_cam_feats = []
            for cam_idx in range(self.num_cameras):
                fmap = _tokens_to_feature_map(toks[:, cam_idx])
                kp = self.image_projector[cam_idx](fmap)  # type: ignore[index]
                per_cam_feats.append(kp.flatten(start_dim=1))  # (B*n_obs_steps, K*2)
            # Average across cameras to keep feature dim = K*2.
            image_features_step = torch.stack(per_cam_feats, dim=1).mean(dim=1)  # (B*n_obs_steps, K*2)
        else:
            fmap = _tokens_to_feature_map(patch_tokens)
            kp = self.image_projector(fmap)  # type: ignore[operator]
            per_image = kp.flatten(start_dim=1)  # (Bflat, K*2)
            per_step = per_image.reshape(effective_batch, self.num_cameras, -1)  # (B*n_obs_steps, Nc, K*2)
            image_features_step = per_step.flatten(start_dim=1)  # (B*n_obs_steps, Nc*K*2)

        image_features = image_features_step.reshape(batch_size, n_obs_steps, -1)

        global_cond = torch.cat([robot_state, image_features], dim=-1)
        return global_cond

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        return self.policy_head(x, timestep, global_cond)

    def compute_loss(
        self,
        actions: Tensor,
        robot_state: Tensor,
        images: Tensor,
        task_indices: Tensor | None = None,
        flow_algo=None,
    ) -> Tensor:
        if flow_algo is None:
            flow_algo = self.flow_algo
        if flow_algo is None:
            raise ValueError("flow_algo must be provided or set via set_flow_algo()")

        global_cond = self.encode_observations(robot_state=robot_state, images=images, task_indices=task_indices)

        def vector_field_net(x_t, t, global_cond):
            return self(x_t, t, global_cond)

        return flow_algo.compute_loss(vector_field_net=vector_field_net, x_1=actions, global_cond=global_cond)

    def generate_actions(
        self,
        robot_state: Tensor,
        images: Tensor,
        task_indices: Tensor | None = None,
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
            batch_size = robot_state.shape[0]

        global_cond = self.encode_observations(robot_state=robot_state, images=images, task_indices=task_indices)

        device = robot_state.device
        initial_noise = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=robot_state.dtype)

        def vector_field_net(x_t, t, global_cond):
            return self(x_t, t, global_cond)

        uncond_global_cond = None
        if uncond_task_indices is not None:
            uncond_global_cond = self.encode_observations(
                robot_state=robot_state, images=images, task_indices=uncond_task_indices
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
    ) -> tuple["FlowMatchingNetwork", dict]:
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        architecture_config = checkpoint["architecture_config"]
        model = cls(**architecture_config)
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


class FlowMatchingNetworkWithEncoder(nn.Module):
    """Convenience wrapper that handles observation encoding."""

    def __init__(self, **kwargs):
        super().__init__()
        self.network = FlowMatchingNetwork(**kwargs)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        robot_state: Tensor,
        images: Tensor,
        task_indices: Tensor | None = None,
    ) -> Tensor:
        global_cond = self.network.encode_observations(robot_state, images, task_indices)
        return self.network(x, timestep, global_cond)

    def encode_observations(self, *args, **kwargs):
        return self.network.encode_observations(*args, **kwargs)


def create_flow_network(
    action_dim: int,
    horizon: int,
    robot_state_dim: int = 9,
    vision_encoder: str = "dino",
    vision_model_size: str = "small",
    policy_head_type: str = "conv",
    **kwargs,
) -> FlowMatchingNetwork:
    return FlowMatchingNetwork(
        action_dim=action_dim,
        horizon=horizon,
        robot_state_dim=robot_state_dim,
        vision_encoder=vision_encoder,
        vision_model_size=vision_model_size,
        policy_head_type=policy_head_type,
        **kwargs,
    )


class FlowMatchingAttentionNetwork(nn.Module):
    """Flow matching network with task-conditioned query-based attention fusion.

    Pipeline:
    - DINOv2 patch tokens (per camera image)
    - task-conditioned query tokens from (object_id, container_id)
    - CrossAttentionPooling to obtain pooled image tokens
    - flatten pooled tokens + concat robot state -> global_cond
    - FiLMConvPolicyHead predicts vector field
    """

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        robot_state_dim: int,
        n_obs_steps: int = 1,
        num_cameras: int = 2,
        num_objects: int = 3,
        num_containers: int = 3,
        input_shape: tuple[int, int, int] = (3, 224, 224),
        config: AttentionModelConfig | None = None,
    ):
        super().__init__()
        if config is None:
            config = AttentionModelConfig()
        self._config = config

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.robot_state_dim = int(robot_state_dim)
        self.n_obs_steps = int(n_obs_steps)
        self.num_cameras = int(num_cameras)
        self.num_query_tokens = int(config.num_query_tokens)
        self.pooled_dim = int(config.pooled_dim)
        self.camera_feature_combine = str(config.camera_feature_combine)

        # Derive the effective crop target: prefer crop_size, fall back to image_size
        # (image_size is kept only for old checkpoint compatibility).
        effective_crop_size = config.crop_size if config.crop_size != 224 or config.raw_image_size != 256 else config.crop_size
        crop_shape = (effective_crop_size, effective_crop_size) if config.crop else None

        # auto_resize_to_224: only needed when crop=False and the input isn't already 224.
        # When crop=True, the RandomCrop/CenterCrop handles the downscaling.
        auto_resize = (not config.crop) and (int(input_shape[1]) != 224 or int(input_shape[2]) != 224)

        self.image_encoder = DinoImageEncoder(
            model_size=config.vision_model_size,
            input_shape=input_shape,
            crop_shape=crop_shape,
            crop_is_random=config.crop_is_random,
            freeze_backbone=config.freeze_vision_backbone,
            auto_resize_to_224=auto_resize,
        )

        d = int(self.image_encoder.model_dim)
        self.num_objects = int(num_objects)
        self.num_containers = int(num_containers)
        if self.num_query_tokens != 2:
            raise ValueError(
                "FlowMatchingAttentionNetwork expects exactly 2 query tokens: "
                "one for object_id and one for container_id."
            )

        self.obj_query = LearnableQueryTokens(num_query_tokens=1, dim=d, num_tasks=self.num_objects)
        self.cont_query = LearnableQueryTokens(num_query_tokens=1, dim=d, num_tasks=self.num_containers)

        self.fusion = CrossAttentionPooling(
            input_shape=(d, 1, 1),
            embed_dim=d,
            out_dim=self.pooled_dim,
            num_layers=config.fusion_num_layers,
            num_heads=config.fusion_num_heads,
            ff_dim=config.fusion_ff_dim,
            dropout=config.fusion_dropout,
            mask_type=config.fusion_mask_type,
        )

        per_image_dim = self.num_query_tokens * self.pooled_dim
        if self.camera_feature_combine == "concat":
            image_feature_dim = self.num_cameras * per_image_dim
        else:
            image_feature_dim = per_image_dim
        global_cond_dim = self.robot_state_dim + image_feature_dim

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
            "n_obs_steps": self.n_obs_steps,
            "num_cameras": self.num_cameras,
            "num_objects": self.num_objects,
            "num_containers": self.num_containers,
            "input_shape": input_shape,
            "config": config.model_dump(),
        }

    def set_flow_algo(self, flow_algo):
        self.flow_algo = flow_algo
        return self

    def encode_observations(
        self,
        *,
        robot_state: Tensor,
        images: Tensor,
        task_indices: Tensor | None = None,
    ) -> Tensor:
        # robot_state: (B, n_obs_steps, Dr)
        B, n_obs_steps = int(robot_state.shape[0]), int(robot_state.shape[1])
        if n_obs_steps != self.n_obs_steps:
            raise ValueError(f"Expected n_obs_steps={self.n_obs_steps}, got {n_obs_steps}")

        # images: (B, n_obs_steps, Nc, C, H, W)
        if images.ndim != 6:
            raise ValueError(f"Expected images to have shape (B,n_obs_steps,Nc,C,H,W), got {tuple(images.shape)}")
        if int(images.shape[2]) != self.num_cameras:
            raise ValueError(f"Expected num_cameras={self.num_cameras}, got {int(images.shape[2])}")

        images_flat = images.reshape(-1, *images.shape[-3:])  # (B*n_obs_steps*Nc, C, H, W)
        patch_tokens = self.image_encoder(images_flat, output="patch")  # (Bflat, L, D)
        Bflat = int(patch_tokens.shape[0])

        # Build per-image task indices of shape (Bflat,2).
        task_flat: Tensor | None = None
        if task_indices is not None:
            if task_indices.ndim != 2 or int(task_indices.shape[1]) != 2:
                raise ValueError(f"task_indices must have shape (B,2) or (B*n_obs_steps,2) or (Bflat,2), got {tuple(task_indices.shape)}")
            if int(task_indices.shape[0]) == B:
                task_step = task_indices.repeat_interleave(n_obs_steps, dim=0)  # (B*n_obs_steps,2)
            else:
                task_step = task_indices
            if int(task_step.shape[0]) == B * n_obs_steps:
                task_flat = task_step.repeat_interleave(self.num_cameras, dim=0)  # (Bflat,2)
            elif int(task_step.shape[0]) == Bflat:
                task_flat = task_step
            else:
                raise ValueError(
                    f"task_indices batch mismatch: got {int(task_indices.shape[0])}, expected {B}, {B*n_obs_steps}, or {Bflat}"
                )

        if task_flat is None:
            raise ValueError("task_indices is required for FlowMatchingAttentionNetwork (object_id, container_id)")

        obj = task_flat[:, 0].to(device=patch_tokens.device, dtype=torch.long)
        cont = task_flat[:, 1].to(device=patch_tokens.device, dtype=torch.long)
        if (obj < 0).any() or (obj >= self.num_objects).any():
            raise ValueError("object_id out of range")
        if (cont < 0).any() or (cont >= self.num_containers).any():
            raise ValueError("container_id out of range")
        q_obj = self.obj_query(
            batch_size=Bflat,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
            task_ids=obj,
        )  # (Bflat,1,D)
        q_cont = self.cont_query(
            batch_size=Bflat,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
            task_ids=cont,
        )  # (Bflat,1,D)
        q = torch.cat([q_obj, q_cont], dim=1)  # (Bflat,2,D)
        pooled = self.fusion(patch_tokens, query_tokens=q)  # (Bflat, Tq, pooled_dim)
        per_image_feat = pooled.flatten(start_dim=1)  # (Bflat, Tq*pooled_dim)

        # Reshape back to per-step, per-camera.
        per_step = per_image_feat.reshape(B, n_obs_steps, self.num_cameras, -1)  # (B, n_obs_steps, Nc, F)
        if self.camera_feature_combine == "concat":
            image_features = per_step.flatten(start_dim=2)  # (B, n_obs_steps, Nc*F)
        else:
            image_features = per_step.mean(dim=2)  # (B, n_obs_steps, F)

        return torch.cat([robot_state, image_features], dim=-1)

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        return self.policy_head(x, timestep, global_cond)

    def compute_loss(
        self,
        actions: Tensor,
        robot_state: Tensor,
        images: Tensor,
        task_indices: Tensor,
        flow_algo=None,
    ) -> Tensor:
        if flow_algo is None:
            flow_algo = self.flow_algo
        if flow_algo is None:
            raise ValueError("flow_algo must be provided or set via set_flow_algo()")

        global_cond = self.encode_observations(robot_state=robot_state, images=images, task_indices=task_indices)

        def vector_field_net(x_t, t, global_cond):
            return self(x_t, t, global_cond)

        return flow_algo.compute_loss(vector_field_net=vector_field_net, x_1=actions, global_cond=global_cond)

    def generate_actions(
        self,
        *,
        robot_state: Tensor,
        images: Tensor,
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

        global_cond = self.encode_observations(robot_state=robot_state, images=images, task_indices=task_indices)
        device = robot_state.device
        initial_noise = torch.randn(batch_size, self.horizon, self.action_dim, device=device, dtype=robot_state.dtype)

        def vector_field_net(x_t, t, global_cond):
            return self(x_t, t, global_cond)

        uncond_global_cond = None
        if uncond_task_indices is not None:
            uncond_global_cond = self.encode_observations(
                robot_state=robot_state, images=images, task_indices=uncond_task_indices
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
    ) -> tuple["FlowMatchingAttentionNetwork", dict]:
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



