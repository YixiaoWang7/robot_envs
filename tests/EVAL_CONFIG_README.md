# Evaluation Configuration Guide

## Overview

The evaluation script `eval_policy_l4.py` now supports loading configuration from JSON files, making it easy to define evaluation tasks and parameters without modifying the code.

## Usage

### Option 1: Using Config File

```bash
python eval_policy_l4.py --config eval_config.json
```

### Option 2: Config File + Command Line Override

You can override specific parameters from the config file using command line arguments:

```bash
python eval_policy_l4.py --config eval_config.json --n-episodes 100 --seed 42
```

### Option 3: Command Line Only (Legacy)

```bash
python eval_policy_l4.py --checkpoint /path/to/model.pt --n-episodes 50 --num-envs 2
```

## Config File Format

The config file is a JSON file with three main sections:

### 1. `eval_config` - Evaluation Parameters

```json
{
  "eval_config": {
    "checkpoint": "/path/to/checkpoint.pt",  // Path to model checkpoint
    "device": "cuda",                         // Device: "cuda" or "cpu"
    "n_episodes": 50,                         // Total number of episodes
    "num_envs": 2,                           // Parallel environments
    "horizon": 200,                          // Max steps per episode
    "n_execute": 8,                          // MPC horizon (actions per query)
    "max_videos": 10,                        // Maximum videos to save
    "seed": 0,                               // Random seed
    "out_dir": "artifacts"                   // Output directory
  }
}
```

### 2. `tasks` - List of Tasks to Evaluate

Specify which tasks to evaluate. The script will sample from this list for each episode.

**All 16 tasks (L4 full evaluation):**
```json
{
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
  ]
}
```

**Single task:**
```json
{
  "tasks": [
    "place the cross into the mug_no_handle"
  ]
}
```

**Subset of tasks:**
```json
{
  "tasks": [
    "place the cross into the bin",
    "place the cube into the mug",
    "place the cylinder into the plate"
  ]
}
```

### 3. `result_config` - Result Saving Options

Control what details are saved in the results:

```json
{
  "result_config": {
    "save_rollouts": true,              // Save action/reward NPZ files
    "save_videos": true,                // Save episode videos
    "save_per_episode_details": true,   // Include per-episode info in JSON
    "save_per_task_breakdown": true,    // Include per-task statistics
    "include_action_statistics": true,  // Include length stats (mean/std/min/max)
    "include_reward_statistics": true   // Include reward stats (mean/std/min/max)
  }
}
```

## Example Config Files

### Example 1: Full L4 Evaluation

See `eval_config.json` - evaluates all 16 L4 tasks with detailed statistics.

### Example 2: Single Task Testing

See `eval_config_single_task.json` - evaluates only one task for quick testing.

### Example 3: Minimal Output

For quick evaluation without detailed stats:

```json
{
  "eval_config": {
    "checkpoint": "/path/to/checkpoint.pt",
    "device": "cuda",
    "n_episodes": 10,
    "num_envs": 2,
    "horizon": 200,
    "n_execute": 8,
    "max_videos": 0,
    "seed": 0,
    "out_dir": "artifacts"
  },
  "tasks": ["place the cross into the mug"],
  "result_config": {
    "save_rollouts": false,
    "save_videos": false,
    "save_per_episode_details": false,
    "save_per_task_breakdown": true,
    "include_action_statistics": false,
    "include_reward_statistics": false
  }
}
```

## Output Structure

After evaluation, the output directory contains:

```
artifacts/eval_YYYYMMDD-HHMMSS/
├── eval_config_used.json       # Config used for this run
├── eval_summary.json           # Detailed results (see below)
├── videos/                     # Episode videos (if enabled)
│   ├── ep000_Success_place_the_cross_into_the_mug.mp4
│   ├── ep001_Failure_place_the_cube_into_the_bin.mp4
│   └── ...
└── rollouts/                   # Episode data (if enabled)
    ├── ep000.npz
    ├── ep001.npz
    └── ...
```

### eval_summary.json Structure

```json
{
  "aggregated": {
    "n_episodes": 50,
    "pc_success": 75.5,           // Overall success rate (%)
    "avg_sum_reward": 1.234,      // Average cumulative reward
    "avg_max_reward": 2.345,      // Average max single-step reward
    "avg_ep_length": 123.4,       // Average episode length
    "std_ep_length": 45.6,        // Standard deviation of length
    "eval_s": 456.7,              // Total evaluation time (seconds)
    "eval_ep_s": 9.13             // Time per episode (seconds)
  },
  "per_task": {
    "place the cross into the bin": {
      "n_episodes": 5,
      "n_success": 4,
      "pc_success": 80.0,
      "avg_sum_reward": 1.5,      // Only if include_reward_statistics=true
      "std_sum_reward": 0.3,
      "min_sum_reward": 1.0,
      "max_sum_reward": 2.0,
      "avg_length": 120.5,        // Only if include_action_statistics=true
      "std_length": 15.2,
      "min_length": 100,
      "max_length": 150
    },
    // ... more tasks
  },
  "per_episode": [               // Only if save_per_episode_details=true
    {
      "episode_ix": 0,
      "task": "place the cross into the bin",
      "success": true,
      "sum_reward": 1.5,
      "max_reward": 2.0,
      "length": 120,
      "seed": 0,
      "video": "artifacts/eval_20260418-123456/videos/ep000_Success_place_the_cross_into_the_bin.mp4"
    },
    // ... more episodes
  ]
}
```

## Tips

1. **Quick Testing**: Use a single task with fewer episodes and no videos for rapid iteration
2. **Full Evaluation**: Use all 16 tasks with 50+ episodes per task for comprehensive results
3. **Debugging**: Enable all statistics and videos to understand failure modes
4. **Production**: Disable videos and per-episode details to save disk space
5. **Reproducibility**: Always set the seed and save the config file with results

## Creating Custom Configs

### Method 1: Use the Config Generator (Recommended)

Use `create_eval_config.py` to generate config files interactively:

```bash
# Quick testing preset (1 task, 10 episodes)
python create_eval_config.py --output my_test.json --preset quick --checkpoint /path/to/model.pt

# Full evaluation preset (all 16 tasks, 50 episodes)
python create_eval_config.py --output my_full.json --preset full --checkpoint /path/to/model.pt

# Minimal preset (fast iteration, minimal output)
python create_eval_config.py --output my_minimal.json --preset minimal --checkpoint /path/to/model.pt

# Custom task selection
python create_eval_config.py --output my_custom.json \
    --checkpoint /path/to/model.pt \
    --objects cross,cube \
    --containers bin,mug \
    --n-episodes 20

# Custom with result options
python create_eval_config.py --output my_config.json \
    --checkpoint /path/to/model.pt \
    --objects cross \
    --containers mug_no_handle \
    --no-videos \
    --no-rollouts
```

Generator options:
- `--preset`: Use predefined configs (full/quick/minimal)
- `--objects`: Comma-separated object names (cross, cube, cylinder, milk)
- `--containers`: Comma-separated container names (bin, mug, plate, mug_no_handle)
- `--no-videos`, `--no-rollouts`, `--no-per-episode`, etc.: Disable specific outputs
- All standard eval parameters: `--n-episodes`, `--num-envs`, `--seed`, etc.

### Method 2: Manual Creation

1. Copy `eval_config.json` or `eval_config_single_task.json`
2. Modify the checkpoint path
3. Select tasks to evaluate
4. Adjust episode count and other parameters
5. Configure result saving options
6. Run: `python eval_policy_l4.py --config your_config.json`
