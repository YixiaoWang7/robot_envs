"""
Reusable neural network components (method-agnostic).

This package contains building blocks that can be composed into policies:

- `policies.modules.vision`: image encoders (e.g., DINOv2/Theia/ResNet backbones).
- `policies.modules.fusion`: fusion / pooling layers used by encoders and multi-modal policies.
- `policies.modules.policy_head`: policy heads that map (x_t, t, cond) -> predicted output.
- `policies.modules.task`: task-conditioning utilities (learnable queries, language prompts, etc.).

All modules follow PyTorch conventions and typically operate on batch-first tensors.
Where relevant, docstrings specify expected tensor shapes using symbols like:
- B: batch size, T: horizon/sequence length, A: action dimension, C/H/W: image channels/height/width.
"""

