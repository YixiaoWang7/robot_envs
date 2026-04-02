# Dataloader And Dataset Guide

This document is the source-of-truth for dataset layout and dataloader usage.

## Dataset format (small-files)

Expected layout:

```text
<data_root>/
  dataset_manifest.json
  <task_slug>/
    demo_manifest.json
    shards/shard_XXX.hdf5
    videos_shards/<camera>/shard_XXX.mp4
    videos_shards/<camera>/shard_XXX.json
```

## Core code

- Schema: `src/policies/datasets/schema.py`
- Dataloader (v2): `src/policies/datasets/robot_datasetv2.py`
- Alignment utilities: `src/policies/datasets/alignment_cache.py`, `alignment_precompute.py`

## Feature config contract

Training expects at least these features in `RunConfig.feature`:

- `actions`
- `state` (or `env_state` for env/object-pose-only policies)
- `images` (optional; omit for state-only policies)

Each feature has shape and temporal window (`history`, `future`).

---

## Dataloader parameter reference

Parameters live under `dataset.sampling` and `dataset.loader` in the run config.

### Parameters that affect data quality (what the model trains on)

These change which samples are drawn, how diverse each training window is, or
what observations the model receives. Changing them alters training behavior.

| Parameter | Where | Effect |
|-----------|-------|--------|
| `val_fraction` | `dataset.split` | Fraction of demos held out for validation. Determines how much training data the model sees. Too high → less training data; too low → unreliable validation signal. |
| `split_seed` | `dataset.split` | Seed for deterministic train/val assignment. Changing it gives a different set of training demos. Use the same seed across experiments for fair comparison. |
| `active_window_shards` | `dataset.sampling` | Number of shards loaded simultaneously per window. Controls diversity within one training window. **Small value (e.g. 1–2):** model trains on one task/shard at a time before moving on — temporally correlated. **Large value (e.g. 8–16):** samples are interleaved from many shards per window — more diverse, better gradient estimates. Recommended: at least 4–8. |
| `k_passes` | `dataset.sampling` | Number of disjoint coverage passes per shard window. Each pass visits a disjoint 1/k subset of the window's samples in a different random order. **k=1:** one big shuffle, may skip some samples before revisiting others. **k>1:** guarantees uniform coverage within a window — every sample appears exactly once across all k passes. Increasing k improves within-window uniformity but does not change total samples seen. For 50k training steps with ~236k samples (13× coverage), values of k=1–20 all give adequate coverage. |
| `cameras` | `dataset` | Which camera views are loaded. Changing this directly changes the visual observations the model receives. |

### Parameters that affect only throughput (speed, not quality)

These control how fast samples are decoded and delivered. They have **no effect**
on which samples are drawn, in what order, or what data they contain.

| Parameter | Where | Effect on speed |
|-----------|-------|-----------------|
| `locality_block_size` | `dataset.sampling` | Number of samples decoded together per video-seek batch. Larger → fewer MP4 seeks per unit time → higher throughput. Smaller → more seeks → lower throughput. Has no effect on sample order or distribution. Default 16 is a good balance. |
| `predecode_next_block` | `dataset.sampling` | If `true`, the next decode block runs in a background thread while the current block is being yielded (read-ahead). Hides I/O latency at the cost of one extra thread per worker. Disable only for debugging. |
| `frame_cache_max_entries` | `dataset.sampling` | Size of per-worker LRU frame cache. Frames already decoded and cached are not re-read from disk. Larger cache → fewer redundant disk reads when a frame is needed by multiple nearby samples (common with `history > 0`). Set to at least `2 × locality_block_size × history_window` for best reuse. |
| `num_workers` | `dataset.loader` | Number of parallel DataLoader worker processes. More workers → higher decode parallelism → higher throughput, up to the point where disk I/O or CPU becomes the bottleneck. Each worker owns a disjoint slice of shards (stride assignment). |
| `prefetch_factor` | `dataset.loader` | How many batches each worker pre-stages before the training loop requests them. Hides worker→trainer transfer latency. Default 4 is sufficient; raising it uses more RAM with minimal gain. |

### Gray area: batch_size

`batch_size` (under `dataset.loader`) affects both:

- **Quality**: larger batches give less-noisy gradient estimates; smaller batches
  introduce more stochasticity (can help generalization in some regimes).
- **Speed**: larger batches amortize per-batch overhead and improve GPU utilization,
  but also require more GPU memory.

---

## Coverage test

To verify sampling quality and coverage without running a full training job:

```bash
python scripts/policies/test_dataset_coverage.py \
    --config src/policies/config/default_run_config_key.json \
    --max_samples 500000 \
    --report_every 10000
```

Key metrics to check:

- **coverage_pct**: % of unique samples seen. Should reach 100% in roughly 1.6× the
  dataset size (overhead comes from per-worker shard ownership).
- **max_hits**: maximum times any single sample was repeated. Should be small (≤ 3–4
  after 2× dataset).
- **Per-task coverage**: should be balanced across tasks.

Typical result with 236k samples and 16 workers: 100% coverage at ~1.6× = 379k samples,
max_hits = 3 after 500k samples.

---

## Data conversion

Use:

```bash
python scripts/policies/convert_hdf5_to_small_files.py --help
```

This produces the small-files structure used by the dataloader.

---

## Notes on v1 vs v2

- Current training script uses `create_robot_dataloader` from `robot_datasetv2.py`.
- Some older docs/scripts may still mention v1 behavior/flags; prefer v2 behavior for new work.

## Related docs

- Training runbook: `scripts/policies/README.md`
- Deployment contract: `src/policies/training/README_deployment.md`
