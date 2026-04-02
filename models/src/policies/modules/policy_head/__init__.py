"""
Policy head modules.

Heads map a trajectory-like tensor and timestep conditioning to a predicted output
over the same horizon (e.g., velocity field for flow matching).

See `policies.modules.policy_head.unet` for detailed shape contracts in docstrings.
"""

from policies.modules.policy_head.unet import (
    FiLMConvPolicyHead,
    ModalityConfig,
    MultiModalPolicyHead,
    PolicyHeadConfig,
    create_policy_head,
)

__all__ = [
    "ModalityConfig",
    "PolicyHeadConfig",
    "MultiModalPolicyHead",
    "FiLMConvPolicyHead",
    "create_policy_head",
]

