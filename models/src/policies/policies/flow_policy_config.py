"""Shared configuration for flow policy networks."""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict


class AttentionModelConfig(BaseModel):
    """Configuration for vision+fusion flow policy networks.

    Also reused by state-only policies for shared policy-head hyperparameters.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Vision — image dimensions
    raw_image_size: int = 256
    crop_size: int = 224
    crop: bool = True
    crop_is_random: bool = True
    # image_size kept for backward compat with old checkpoints — do not use in new configs.
    image_size: int = 224
    vision_model_size: str = "small"
    freeze_vision_backbone: bool = False
    # Fusion
    pooled_dim: int = 64
    fusion_num_layers: int = 4
    fusion_num_heads: int = 8
    fusion_ff_dim: int = 512
    fusion_dropout: float = 0.1
    fusion_mask_type: str = "none"
    camera_feature_combine: Literal["concat", "mean"] = "concat"
    num_query_tokens: int = 2
    # Policy head
    down_dims: Tuple[int, ...] = (128, 256, 512)
    time_embed_dim: int = 128
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    # ---------------------------------------------------------------------
    # Env-state token conditioning (state-based policies, no images)
    # ---------------------------------------------------------------------
    # env_state is expected to start with:
    #   env_num_entities * env_entity_dim
    # pose values. These are tokenized into per-entity tokens and pooled using
    # query-conditioned cross-attention (object/container queries).
    # Any trailing env_state dims beyond this base layout are ignored.
    env_num_entities: int = 6
    env_entity_dim: int = 7
    env_token_embed_dim: int = 256

