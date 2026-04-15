### Motion planner README (`generate_cg_l4_motion_planning.py`)

This document explains how to run and tune the waypoint motion planner in `repos/robot_envs/tests/generate_cg_l4_motion_planning.py`.

### What this script does

`generate_cg_l4_motion_planning.py` generates CG_L4 pick-and-place demonstrations **without a learned policy**:

- **Environment**: robosuite CG_L4 (one-stage “place the {object} into the {container}”)
- **Planner**: scripted waypoint sequence (pregrasp → (optional rotate) → grasp → post_grasp_lift → prerelease → release → lift)
- **Controller**: OSC (`OSC_POSE`) with **delta actions**; position-only by default, optional yaw-only orientation control
- **Artifacts**: successful demos saved to **HDF5**, optional debug **MP4 videos**, plus logs and metadata

### Quickstart

Activate the environment first (adjust to your setup):

```bash
source repos/robot_envs/.venv/bin/activate
```

Run from the repo root:

```bash
python repos/robot_envs/tests/generate_cg_l4_motion_planning.py \
  --n-success 200 \
  --num-envs 4 \
  --horizon 220 \
  --out-dir repos/robot_envs/results/test \
  --max-videos 20
```

Notes:
- The script sets `MUJOCO_GL=egl` by default.
- Videos are saved for both successes and failures (bounded by `--max-videos`).

### Output directory layout

For `--out-dir X`, each run writes a new folder:

- **`X/gen_YYYYMMDD-HHMMSS/`**
  - **`demos.hdf5`**: saved successful episodes (one `demo_{k}` group per saved episode)
  - **`videos/`** (only if `--max-videos > 0`): debug videos `ep{attempt_id}_{task}_{success|fail}.mp4`
  - **`args.json`**: the parsed CLI args
  - **`gen.log`**: log file (same content as stdout)
  - **`gen_info.json`**: aggregate generation stats (overall + per-task breakdown)

### CLI arguments (every argument explained)

- **`--n-success`** (int, default: 200)
  - Number of **successful episodes to save** into `demos.hdf5`.
  - The script keeps sampling rollouts until it has saved this many successes.

- **`--num-envs`** (int, default: 1)
  - Number of **parallel environments** to run in a batch.
  - Larger values increase throughput, but may increase variability / GPU load for rendering.

- **`--horizon`** (int, default: 220)
  - Maximum steps per episode. If the planner doesn’t finish before this, the rollout is marked as failure.

- **`--out-dir`** (str, default: `results/cg_l4_motion_gen`)
  - Parent output directory. The script creates a timestamped run folder under this directory.

- **`--max-videos`** (int, default: 20)
  - Maximum number of MP4 videos to write to `videos/`.
  - Set to **0** to disable video writing entirely.

- **`--fps`** (int, default: 20)
  - Output video framerate (frames per second).

- **`--seed`** (int, default: 0)
  - Seeds NumPy (`np.random.seed(seed)`), which affects noise injection and some sampling paths.

- **`--log-level`** (str, default: `INFO`)
  - Logging verbosity. Common values: `DEBUG`, `INFO`, `WARNING`.
  - Logs go to both stdout and `gen.log`.

- **`--use-orientation-control`** (flag, default: off)
  - Enables OSC **orientation deltas** (axis-angle) in addition to translation deltas.
  - When enabled, cross/cube can optionally do a **yaw-only** alignment for flatter grasps.

- **`--no-rotate-for-grasp`** (flag, default: off)
  - Disables the yaw-alignment logic entirely.
  - When set, the planner becomes **position-only** even if `--use-orientation-control` is enabled.
  - Cylinder never needs rotation anyway; this flag is useful when you want “no rotation at all” for every object.

### Planner tuning knobs (edit `PickPlaceParams` in `main()`)

Most motion-planner behavior is controlled by the `PickPlaceParams` object created in `main()` and passed to `generate_dataset(...)` as `planner_params=...`.

The key fields you’re likely to tune:

- **`use_orientation_control`** (bool)
  - Turns orientation tracking on/off globally.

- **`rotate_for_grasp_enable`** (bool)
  - If false, removes the `rotate_for_grasp` stage and disables yaw alignment for `grasp`.

- **`stage_offset_rules`** (list of `StageOffsetRule`)
  - **Manual per object/container offsets** (world frame, meters) applied to generated targets:
    - `pregrasp`, `grasp`, `preplace` (prerelease), `place` (release)
  - Rules can match by **stage only**, **stage+object**, **stage+container**, or **stage+object+container**.
  - Multiple matching rules **add** together.

- **`pregrasp_h`**, **`preplace_h`** (floats, meters)
  - Vertical offsets above object / place targets for approach waypoints.

- **`post_grasp_lift_z`** (float, meters)
  - Small vertical lift after grasp, before translating toward the container (collision avoidance).

- **`release_z_above_container_center*`** (floats, meters)
  - Z offset for the release target relative to container body center (different defaults for mug / plate).

- **`pose_noise_enable`** and `noise_*` (floats, meters)
  - Gaussian noise injected into pregrasp/grasp/preplace/place targets to diversify trajectories.

#### Example: add object/container specific offsets

Edit the `PickPlaceParams(...)` call in `main()` (near the end of the file) like this:

```python
from generate_cg_l4_motion_planning import PickPlaceParams, StageOffsetRule

planner_params = PickPlaceParams(
    use_orientation_control=True,
    rotate_for_grasp_enable=True,
    stage_offset_rules=[
        # Per-object tweak (all containers)
        StageOffsetRule(stage="pregrasp", object="cube", delta_xyz=(0.0, 0.0, 0.01)),
        # Per-container tweak (all objects)
        StageOffsetRule(stage="place", container="mug", delta_xyz=(0.0, 0.01, 0.0)),
        # Specific pair tweak
        StageOffsetRule(stage="grasp", object="cross", container="plate", delta_xyz=(0.005, 0.0, 0.0)),
    ],
)
```

Valid stage names are: **`pregrasp`**, **`grasp`**, **`preplace`**, **`place`**.

