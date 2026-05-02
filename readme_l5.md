# CG_L5: Pick, Place, Then Press Colored Button

`CG_L5` extends the CG pick-and-place task with a final language-conditioned button press. A task string has the form:

```text
place the <object> into the <container> then press the <color> button
```

Valid objects are `cross`, `cube`, `cylinder`, and `milk`. Valid containers are `bin`, `mug`, `plate`, and `mug_no_handle`. Valid button colors are `red`, `green`, `blue`, and `yellow`. This gives `4 x 4 x 4 = 64` possible L5 tasks.

The environment first requires the existing pick-and-place success condition, including lifting the gripper away after placement. Only after that stage is complete can the episode finish by pressing the correct colored button.

## Data Generation

The L5 motion-planning data generator is:

```bash
python tests/generate_cg_l5_motion_planning.py \
  --n-success 200 \
  --num-envs 4 \
  --horizon 320 \
  --out-dir results/cg_l5_motion_gen
```

To generate a fixed task:

```bash
python tests/generate_cg_l5_motion_planning.py \
  --task "place the cube into the bin then press the red button" \
  --n-success 20 \
  --num-envs 1
```

To save a fixed number of demos for each L5 task:

```bash
python tests/generate_cg_l5_motion_planning.py \
  --per-task-success 10 \
  --num-envs 4
```

Available generator arguments:

- `--n-success`: number of successful episodes to save when not using `--per-task-success` (default: `200`).
- `--per-task-success`: if greater than `0`, save this many successful demos for each task. Since there are 64 tasks, `--per-task-success 10` saves up to `640` demos.
- `--task`: optional exact language task string, for example `"place the cube into the bin then press the red button"`.
- `--num-task-shards`, `--num-shards`: split the 64 tasks into this many shards (default: `1`).
- `--task-shard-index`, `--shard-index`: shard index to run, in `[0, num_task_shards)` (default: `0`).
- `--num-envs`: number of parallel environments (default: `1`).
- `--horizon`: maximum rollout steps per episode (default: `320`).
- `--out-dir`: output root directory (default: `results/cg_l5_motion_gen`).
- `--run-id`: optional run id. Shards with the same run id write to the same `gen_<run-id>` folder.
- `--max-videos`: maximum number of review videos to save under the run's `videos/` directory; use `0` to disable (default: `20`).
- `--fps`: video FPS and post-success hold timing basis (default: `20`).
- `--seed`: NumPy random seed (default: `0`).
- `--log-level`: logging level (default: `INFO`).
- `--use-orientation-control`: enable grasp yaw/orientation control.
- `--no-rotate-for-grasp`: disable the planner's rotate-for-grasp yaw alignment.

Each saved demo directory contains:

- `demo.hdf5`
- `agentview.mp4`
- `robot0_eye_in_hand.mp4`

The HDF5 file stores the following per-step observations under `obs/`:

- `robot0_eef_pos`: `(T, 3)` end-effector position.
- `robot0_eef_quat`: `(T, 4)` end-effector quaternion.
- `robot0_gripper_qpos`: `(T, 2)` gripper joint positions.
- `object`: `(T, N * 7)` world poses of all posed entities, including objects, containers, and all four buttons. Each pose is `[x, y, z, qx, qy, qz, qw]`.
- `environment_state`: relative state used by the policy. It contains entity poses relative to the end-effector, plus target object and target button relative positions.
- `task_indices`: `(T, 3)` integer array `[object_index, container_index, button_color_index]`.
- `subtask_id`: `(T, 1)` integer stage label: `0` before grasp, `1` after grasp before place completion, and `2` during the button-press stage.
- `stage_success`: `(T, 3)` boolean array in the order `[grasp, place, button_press]`.
- `grasp_success`: `(T, 1)` boolean signal indicating the object has been grasped.
- `place_success`: `(T, 1)` boolean signal indicating the pick-and-place stage has completed.
- `button_press_success`: `(T, 1)` boolean signal indicating the correct colored button has been pressed.

The HDF5 file also stores:

- `actions`: `(T, 7)` robot action array, usually `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.

The HDF5 attributes are:

- `task`: original language task string.
- `planner`: planner name, currently `l5_pick_place_button`.
- `success`: whether the full three-stage episode succeeded.
- `stage_success_order`: fixed string `grasp,place,button_press`.

Each generation run also writes:

- `args.json`: command-line arguments used for the run.
- `gen_info.json`: run summary, including attempted episodes, successful episodes, saved episodes, success rates, elapsed time, output demo directory, number of videos saved, and per-task success statistics.
- `dataset_manifest.json`: dataset index with task slugs, task strings, task indices, demo directories, demo counts, and stage order.

The generator reuses the CG_L4 waypoint pick-and-place planner for the first two stages, then runs a small position-only button press planner over the target colored button.

### 2x3090 Sharded Generation

The 64 tasks can be split across multiple terminals with `--num-task-shards` and `--task-shard-index`. For example, with 8 shards on 2 GPUs, run one command per terminal and change only `SHARD`:

```bash
SHARD=3
GPU_ID=$((SHARD % 2))

CUDA_VISIBLE_DEVICES=${GPU_ID} python tests/generate_cg_l5_motion_planning.py \
  --per-task-success 300 \
  --num-task-shards 4 \
  --task-shard-index ${SHARD} \
  --num-envs 1 \
  --horizon 400 \
  --out-dir "/media/db4/yixiao/cg/data/l5_shards" \
  --run-id "l5_300_per_task" \
  --max-videos 0 \
  --seed $((1000 + SHARD))
```

Repeat this for `SHARD=0` through `SHARD=7`. Each shard generates 8 of the 64 tasks, so with `--per-task-success 300` each terminal saves about `8 * 300 = 2400` successful demos. Since every shard uses the same `--run-id`, all demos are written under the same run folder:

```text
/media/db4/yixiao/cg/data/l5_shards/gen_l5_300_per_task/task/
```

Each shard writes its own metadata files, such as `args_shard_00_of_08.json`, `gen_info_shard_00_of_08.json`, and `dataset_manifest_shard_00_of_08.json`, so the terminals do not overwrite each other's summaries. The script also refreshes a shared `dataset_manifest.json` in the run folder. While shards are still running, this combined manifest may be partial; after all shards finish, it should contain all 64 task entries.