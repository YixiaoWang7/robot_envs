"""
Fusion / pooling modules used to combine modalities.

These modules are shared utilities that compress spatial feature maps into compact
representations (e.g., keypoints, query-pooled tokens).
"""

from policies.modules.fusion.attention_projector import CrossAttentionPooling, SelfAttentionPooling
from policies.modules.fusion.pooling import SpatialSoftmax

__all__ = [
    "SpatialSoftmax",
    "SelfAttentionPooling",
    "CrossAttentionPooling",
]

