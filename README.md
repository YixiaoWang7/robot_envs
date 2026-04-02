## `robot-envs`

Standalone env repo that vendors CG's `robosuite` and exposes CG-L2 wrappers.

### Install (uv)

From this repo root:

```bash
uv venv -p 3.10
source .venv/bin/activate
uv pip install -e .
```

Notes:
- `robosuite` is installed from `CG/robosuite` via `[tool.uv.sources]` in `pyproject.toml`.
- MuJoCo + an OpenGL backend must be available for simulation and (optional) rendering.

### Smoke test

```bash
export MUJOCO_GL=egl   # common for headless GPU; alternatives: osmesa (CPU), glfw (onscreen)
python tests/smoke_cg_l2_image.py --steps 50
```

Expected:
- prints observation keys and shapes
- prints non-zero mean EEF deltas (actions are affecting the robot)
- prints non-zero mean image deltas (rendered frames change over time)

### Imports after install

```python
from robosuite.environments.manipulation.CG_L2 import CG_L2
from CG_L2_image_wrapper import ImageBasedCGWrapper
from CG_L2_state_wrapper import StateBasedCGWrapper
```

### Wrapper constraints (important)

- `num_envs` must be even (`reset()` will raise otherwise)
- you must set `env.train_task` before `reset()`
  - allowed values: `"all"`, `"S"`, `"L"`, `"L2"`, `"diag"`, `"Sfull"`, `"diagmid"`, `"diagcorner"`, `"only-00"`



python tests/eval_policy.py --mode flow --checkpoint models/ckpt/ckpt_step_0050000.pt --steps 200 --device cuda:1 --out-dir artifacts


python tests/eval_policy.py   --checkpoint /home/yixiao/Documents/code/robopolicy/runs/robot_flow_train_lr_cosine_ac20_k8_amp/checkpoints/ckpt_step_0050000.pt   --n-episodes 50 --num-envs 4 --horizon 400   --n-execute 8 --device cuda:1   --max-videos 10 --out-dir artifacts/ac20_k8_amp


/home/yixiao/Documents/code/robopolicy/runs/robot_flow_train_lr_cosine_state_only_ac20_k8/checkpoints/ckpt_step_0050000.pt


python tests/eval_policy.py   --checkpoint /home/yixiao/Documents/code/robopolicy/runs/robot_flow_train_lr_cosine_state_only_ac20_k8/checkpoints/ckpt_step_0050000.pt   --n-episodes 50 --num-envs 4 --horizon 400   --n-execute 8 --device cuda:1   --max-videos 24 --out-dir results/state_attn


python tests/eval_policy.py   --checkpoint /home/yixiao/Documents/code/robopolicy/runs/robot_flow_train_lr_cosine_state_direct_select_ac20_k8/checkpoints/ckpt_step_0050000.pt   --n-episodes 50 --num-envs 4 --horizon 400   --n-execute 8 --device cuda:1   --max-videos 24 --out-dir results/state_direct



python tests/eval_policy.py   --checkpoint /home/yixiao/Documents/code/robopolicy/runs/state_all_10kstep/checkpoints/ckpt_step_0100000.pt   --n-episodes 100 --num-envs 4 --horizon 400   --n-execute 8 --device cuda:1   --max-videos 24 --out-dir results/state_direct
