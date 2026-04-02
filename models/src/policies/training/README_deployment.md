# Deployment: Load Checkpoint And Run Inference

This document defines the runtime contract for loading a trained policy checkpoint.

## API entrypoint

Use `load_robot_flow_policy`:

```python
from policies.training.policy_loading import load_robot_flow_policy

policy = load_robot_flow_policy("path/to/ckpt.pt", device="cpu")
policy.model.eval()
```

The checkpoint contains:

- model weights
- processor stats
- serialized run config

## Inference input contract

Call `policy.model.generate_actions(...)` with tensors:

- `robot_state`: `(B, n_obs_steps, state_dim)`, `torch.float32`
- `images`: `(B, n_obs_steps, n_cams, C, H, W)`, `torch.float32` (normalized)
- `task_indices`: `(B, 2)`, `torch.int64` (`[object_idx, container_idx]`)

Then denormalize actions:

```python
actions_norm = policy.model.generate_actions(
    robot_state=robot_state,
    images=images,
    task_indices=task_indices,
    flow_algo=policy.model.flow_algo,
    batch_size=B,
)
actions = policy.denormalize_actions(actions_norm)
```

`actions` is in real robot output units.

## Important

- `generate_actions` expects normalized tensor inputs.
- If your runtime observations are raw, you must apply the same preprocessing/normalization expected by training.

## Runnable example

For a complete executable demo, see:

- `repos/model_loading_test/example_inference.py`
- `repos/model_loading_test/README.md`
