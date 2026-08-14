# STAC-QM Implementation and Runbook

STAC-QM adds Spatio-Temporally Aligned, Reliability-Gated Causal Query Memory to SparseWorld trajectory forecasting. The authoritative modeling and safety details are in [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md).

## Implemented Data Flow

Only raw observation queries from real past frames are cached:

```text
base SparseWorld RAP observation queries
  -> schema-v2 per-sample cache
  -> target-age same-scene history selection
  -> ego-pose alignment
  -> zero-initialized motion residual
  -> reliability-aware causal attention
  -> confidence-gated residual fusion
  -> original recursive SCF future prediction
```

STAC-QM does not add predicted-query memory, instance tracking, query-to-track conversion, dynamic/static query splitting, SCF parallelization, fixed OccWorld tokens, or VQ-VAE.

## Schema-v2 Cache Record

Each selected observation-query record contains:

```text
query_feat                     [M, C]
query_points_metric            [M, R, 3]
query_conf                     [M]
query_semantic_distribution    [M, C_sem]
query_label                    [M]
query_margin                   [M]
query_entropy                  [M]
query_reliability              [M]
valid_mask                     [M]
ego2global                     [4, 4]
timestamp, frame_idx, scene_id, sample_idx
pc_range, embed_dims, num_points, num_classes
source_config, source_checkpoint, schema_version
```

Schema v1 remains readable with `query_reliability=query_conf` and `query_label=-1`. Formal Memory-only runs require complete schema-v2 caches.

## SCF Integration

Memory reads are bounded by construction:

- observation queries read once before SCF at `future_offset=0`;
- each of six scheduled groups reads once when introduced;
- already-active queries never re-read Memory.

Default runtime expectation:

```text
query counts:   [720, 60, 60, 60, 60, 40, 40]
future offsets: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
reads:          7
fused queries:  1040
```

## Configurations

### Split Cache Roots

```text
train: data/query_memory/sparseworld_epoch56_schema2_train
val:   data/query_memory/sparseworld_epoch56_schema2_val
test:  data/query_memory/sparseworld_epoch56_schema2_val
```

Train, val, and test routes are assigned separately. Formal loaders use `strict=True`.

### Formal Memory-Only Config

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py
```

It sets:

- `load_from='ckpts/epoch_56.pth'`;
- `resume_from=None`;
- `memory_finetune_mode=True`;
- `freeze_base_model=True`;
- `source='cache'`;
- AdamW, LR `1e-4`, weight decay `1e-2`;
- gradient clipping at norm 5;
- 12 epochs and one checkpoint per epoch.

Only `query_memory.*` is trainable and optimized. TASS assignment is finalized once after checkpoint loading and then asserted immutable.

### Connectivity Smoke Config

```text
configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only-smoke.py
```

This is a separate 200-iteration run. Its optimizer hook checks base gradients/state, TASS immutability, 7/1040 runtime behavior, and gradient connectivity for fusion and attention. It reports motion connectivity without claiming success when the motion final layer has no gradient.

## User-Run Commands

These commands require the repository's real nuScenes data, checkpoint, and GPU environment. They were prepared but **not run** in the implementation session.

### 1. Generate Train Cache

```bash
/data/jxy/projects/env/bin/python tools/query_memory/precompute_query_memory.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --split train \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_train \
  --max-queries-per-frame 256 \
  --write-threshold 0.35 \
  --min-reliability 0.0 \
  --spatial-cell-size 4.0 \
  --max-per-spatial-cell 16 \
  --max-per-class 64 \
  --workers-per-gpu 2 \
  --skip-existing
```

### 2. Generate Val Cache

```bash
/data/jxy/projects/env/bin/python tools/query_memory/precompute_query_memory.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --split val \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_val \
  --max-queries-per-frame 256 \
  --write-threshold 0.35 \
  --min-reliability 0.0 \
  --spatial-cell-size 4.0 \
  --max-per-spatial-cell 16 \
  --max-per-class 64 \
  --workers-per-gpu 2 \
  --skip-existing
```

Do not pass `--overwrite` unless replacing existing caches is intentional and verified.

### 3. Audit Train and Val Caches

```bash
/data/jxy/projects/env/bin/python tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --split train \
  --expected-schema-version 2 \
  --expected-source-checkpoint ckpts/epoch_56.pth \
  --json-out work_dirs/stacqm_cache_audit_train.json
```

```bash
/data/jxy/projects/env/bin/python tools/query_memory/audit_query_memory_cache.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --split val \
  --expected-schema-version 2 \
  --expected-source-checkpoint ckpts/epoch_56.pth \
  --json-out work_dirs/stacqm_cache_audit_val.json
```

Do not continue to training if either audit exits nonzero.

### 4. Run Zero-Initialization C0/C1 Identity

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python \
  tools/query_memory/check_query_memory_identity.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --mode zero \
  --split val \
  --gpu-id 0 \
  --atol 1e-6
```

The tool automatically chooses the first sample with all target-age slots unless `--sample-index` is specified. It must report valid histories/candidates, 7 reads, 1040 fused queries, zero residual, and identity within `1e-6`.

### 5. Run the 200-Iteration Connectivity Smoke

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only-smoke.py \
  --work-dir work_dirs/sparseworld-traj-memory-only-smoke \
  --gpu-id 0 \
  --deterministic
```

Inspect the hook output and final run status. Do not treat a produced checkpoint alone as proof that connectivity checks passed.

### 6. Check Trained Memory ON/OFF Behavior

After a successful smoke run, point `--memory-checkpoint` to its checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python \
  tools/query_memory/check_query_memory_identity.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint ckpts/epoch_56.pth \
  --memory-checkpoint work_dirs/sparseworld-traj-memory-only-smoke/iter_200.pth \
  --mode trained \
  --split val \
  --gpu-id 0 \
  --atol 1e-6 \
  --min-future-diff 1e-6
```

This mode expects current-frame identity and a nonzero difference in at least one future output.

### 7. Start Formal 12-Epoch Training

Only after cache audits, C0/C1 identity, smoke connectivity, and trained ON/OFF checks pass:

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --work-dir work_dirs/sparseworld-traj-memory-only \
  --gpu-id 0 \
  --deterministic
```

Do not use `--auto-resume` for the initial run. The config loads `ckpts/epoch_56.pth` and sets `resume_from=None`.

### 8. Evaluate a Formal Checkpoint

Example only; replace the checkpoint path with an actually completed epoch:

```bash
CUDA_VISIBLE_DEVICES=0 /data/jxy/projects/env/bin/python tools/test.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-only.py \
  --checkpoint work_dirs/sparseworld-traj-memory-only/epoch_12.pth \
  --gpu-id 0 \
  --eval segm
```

No evaluation or ablation result is recorded until the command is run and its output is reviewed.

## Completed Quick Verification

The implementation session ran only static and synthetic checks:

```text
32 passed, 19 warnings in 4.24s
```

It also compiled the modified Python/config files, loaded all four STAC-QM configs, confirmed strict train/val/test routing, verified smoke-hook registration, and checked both new tools import successfully. No real cache generation, training, or evaluation was launched.
