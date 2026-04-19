# robot-envs

Standalone env repo that vendors CG's `robosuite` and exposes CG-L2/L4 wrappers.

---

## Install

```bash
uv venv -p 3.10
source .venv/bin/activate
uv pip install -e .
```

- `robosuite` is installed from `CG/robosuite` via `[tool.uv.sources]` in `pyproject.toml`.
- MuJoCo and an OpenGL backend must be available (`egl` for headless GPU, `osmesa` for CPU, `glfw` for onscreen).

---

## Smoke test

```bash
export MUJOCO_GL=egl
python tests/smoke_cg_l2_image.py --steps 50
```

Expected output: observation shapes, non-zero EEF deltas, non-zero image deltas.

---

## Wrappers

```python
from CG_L2_image_wrapper import ImageBasedCGWrapper   # image + state obs
from CG_L2_state_wrapper import StateBasedCGWrapper    # state-only obs
```

Both wrappers batch `num_envs` independent robosuite environments and expose a gym-style interface (`reset`, `step`, `close`).

**Key property — `train_task`** (set before `reset()`):

| Value | Behaviour |
|---|---|
| `"all"` | Sample uniformly from the full task set each episode |
| `["task a", "task b", ...]` | Sample uniformly from the given allow-list each episode |
| `"place the cross into the bin"` | Fixed task for every episode |

---

## Policy Evaluation — CG_L4 (`eval_policy_l4.py`)

### Option A — Config file (recommended)

Create a JSON config (see templates in `tests/`) and run:

```bash
cd tests
python eval_policy_l4.py --config eval_config.json
```

Config file format (`eval_config.json`):

```json
{
  "eval_config": {
    "checkpoint": "/path/to/ckpt_step_XXXXXX.pt",
    "device": "cuda",
    "n_episodes": 100,
    "num_envs": 4,
    "horizon": 400,
    "n_execute": 8,
    "max_videos": 24,
    "seed": 0,
    "out_dir": "results/my_run"
  },
  "tasks": [
    "place the cross into the bin",
    "place the cross into the mug",
    "place the cross into the mug_no_handle",
    "place the cross into the plate",
    "place the cube into the bin",
    "place the cube into the mug",
    "place the cube into the mug_no_handle",
    "place the cube into the plate",
    "place the cylinder into the bin",
    "place the cylinder into the mug",
    "place the cylinder into the mug_no_handle",
    "place the cylinder into the plate",
    "place the milk into the bin",
    "place the milk into the mug",
    "place the milk into the mug_no_handle",
    "place the milk into the plate"
  ],
  "result_config": {
    "save_rollouts": true,
    "save_videos": true,
    "save_per_episode_details": true,
    "save_per_task_breakdown": true,
    "include_action_statistics": true,
    "include_reward_statistics": true
  }
}
```

You can override any `eval_config` field from the command line:

```bash
python eval_policy_l4.py --config eval_config.json --n-episodes 200 --seed 42
```

python tests/eval_policy_l4.py --config tests/test_configs/full.json --checkpoint 


python tests/eval_policy_l4.py --config tests/test_configs/full.json --seed 213 ; python tests/eval_policy_l4.py --config tests/test_configs/full.json --seed 763 ; python tests/eval_policy_l4.py --config tests/test_configs/full.json --seed 7  


To generate a config from presets:

```bash
# full: all 16 tasks, 50 episodes
python create_eval_config.py --preset full  --checkpoint /path/to/ckpt.pt --output my_full.json

# quick: 1 task, 10 episodes
python create_eval_config.py --preset quick --checkpoint /path/to/ckpt.pt --output my_quick.json

# custom object/container subset
python tests/create_eval_config.py --objects cross,cube --containers bin,mug \
    --checkpoint /home/yixiao/Documents/code/robopolicy/runs/l4_single_stage_longer_full/checkpoints/ckpt_step_0085000.pt --output tests/test_configs/my_subset.json


python tests/create_eval_config.py --preset full  --checkpoint /home/yixiao/Documents/code/robopolicy/runs/l4_single_stage_longer_full/checkpoints/ckpt_step_0085000.pt --output tests/test_configs/full.json

```

### Option B — Command line only

```bash
python tests/eval_policy_l4.py \
    --checkpoint /path/to/ckpt_step_XXXXXX.pt \
    --n-episodes 100 \
    --num-envs 4 \
    --horizon 400 \
    --n-execute 8 \
    --device cuda \
    --max-videos 24 \
    --out-dir results/my_run
```

With `--config` absent, the script evaluates **all 16 L4 tasks** by default.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | — | Path to JSON config file |
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--n-episodes` | `50` | Total episodes to run |
| `--num-envs` | `2` | Parallel environments per batch |
| `--horizon` | `200` | Max steps per episode |
| `--n-execute` | `8` | Actions executed per policy query (MPC) |
| `--max-videos` | `10` | Max episode videos saved |
| `--seed` | `0` | Random seed |
| `--out-dir` | `artifacts` | Root output directory |

### Output

Results are written to `<out-dir>/eval_<YYYYMMDD-HHMMSS>/`:

```
eval_20260418-120000/
├── eval_config_used.json   # exact config used for this run
├── eval_summary.json       # all metrics (see below)
├── videos/
│   ├── ep000_Success_place_the_cross_into_the_bin.mp4
│   ├── ep001_Failure_place_the_cube_into_the_mug.mp4
│   └── ...
└── rollouts/
    ├── ep000.npz           # actions, rewards, success flag
    └── ...
```

`eval_summary.json` structure:

```json
{
  "aggregated": {
    "n_episodes": 100,
    "pc_success": 72.0,
    "avg_sum_reward": 1.23,
    "avg_max_reward": 2.10,
    "avg_ep_length": 185.4,
    "std_ep_length": 42.1,
    "eval_s": 610.3,
    "eval_ep_s": 6.1
  },
  "per_task": {
    "place the cross into the bin": {
      "n_episodes": 8,
      "n_success": 7,
      "pc_success": 87.5,
      "avg_sum_reward": 1.45,
      "std_sum_reward": 0.31,
      "avg_length": 162.0,
      "std_length": 28.4
    }
  },
  "per_episode": [
    { "episode_ix": 0, "task": "...", "success": true, "sum_reward": 1.5, "length": 140, ... }
  ]
}
```

---

## Data Generation — CG_L4 (`generate_cg_l4_motion_planning.py`)

Generate motion-planning demonstrations for L4 tasks:

```bash
# All 16 tasks, 500 successes each
python tests/generate_cg_l4_motion_planning.py \
    --per-task-success 500 \
    --num-envs 1 \
    --horizon 300 \
    --out-dir /mnt/ssd/data/cg/l4/single_stage \
    --max-videos 5

# Single task
python tests/generate_cg_l4_motion_planning.py \
    --per-task-success 500 \
    --num-envs 1 \
    --horizon 300 \
    --task "place the cross into the mug_no_handle" \
    --out-dir /mnt/ssd/data/cg/l4 \
    --max-videos 5
```
