"""
Vision modules.

This subpackage is intentionally kept *encoder-only* (clean surface area).
Pooling/projection utilities live in `policies.modules.fusion`.
"""

from policies.modules.vision.dino import DinoImageEncoder

__all__ = [
    "DinoImageEncoder",
]

