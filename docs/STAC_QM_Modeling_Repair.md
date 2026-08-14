# STAC-QM Modeling Repair and Memory-Only Acceptance

This document records the STAC-QM modeling repair and the follow-up Memory-only training-safety implementation completed in 2026-08.

## Status

Completed in code:

- the six STAC-QM modeling repairs;
- complete schema-v2 cache fields and strict validation;
- runtime synthetic proof of 7 Memory reads and 1040 fused queries;
- explicit `memory_finetune_mode=True` behavior;
- frozen TASS temporal assignment after loading the base checkpoint;
- query-memory-only optimizer construction and guarded smoke-training hooks;
- split-specific strict train/val cache routing;
- cache-audit and C0/C1 identity tools;
- synthetic CPU tests and configuration compilation.

Not run in this repository session:

- full train or val cache generation;
- cache-wide real-data audit;
- real-data C0/C1 identity;
- 100–500 iteration GPU smoke training;
- 12-epoch Memory-only training;
- nuScenes evaluation or ablations.

No metric improvement, cache completion, training success, or evaluation result is claimed. Generated `.pt` caches, checkpoints, datasets, prediction files, output PKLs, and large logs must not be committed.

## Six Modeling Repairs

### 1. One Real-History Read Per Query

Observation queries read Memory once before SCF:

```text
720 observation queries
  -> STAC-QM(memory, future_offset=0.0)
  -> SCF recursion
```

Each scheduled future-query group reads Memory once when introduced:

```text
[60, 60, 60, 60, 40, 40]
future_offset = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

Already-active queries are never re-read. The default schedule therefore has:

```text
7 Memory calls
720 + 60 + 60 + 60 + 60 + 40 + 40 = 1040 fused queries
```

`tests/test_query_memory_integration.py` exercises the actual `SparseWorld4DTraj.forward_backbone()` control flow with lightweight CPU fakes and asserts those exact calls and offsets.

### 2. Future-Aware Effective Age

Caches retain immutable base timestamps/ages:

```text
base_age = current_timestamp - history_timestamp
```

At each read:

```text
effective_age = base_age + future_offset
```

Causal filtering, maximum-age filtering, temporal attention penalty, motion compensation, and diagnostics use `effective_age`. Cached timestamps are never modified.

### 3. Zero-Initialized Motion Compensation

`QueryMotionCompensator` predicts a bounded per-memory-query velocity:

```text
velocity = v_max * tanh(MLP([LN(memory_feature), time_features]))
aligned_points = ego_aligned_points + effective_age * velocity
```

The final motion MLP layer is initialized to zero, so initial behavior is exactly ego-pose alignment only.

### 4. Per-Query Semantic Reliability

Schema v2 stores:

```text
query_semantic_distribution [M, C_sem]
query_label                 [M]
query_margin                [M]
query_entropy               [M]
query_reliability           [M]
```

Reliability is derived from top-1 probability, top-1/top-2 margin, and normalized entropy:

```text
query_reliability = mean(top1, margin, 1 - H(p) / log(C))
```

The result is clamped to `[0, 1]`. `query_conf` remains for schema-v1 compatibility:

```text
schema v1 fallback:
query_reliability = query_conf
query_label = -1
```

Schema-v2 validation checks exact shapes, finite values, valid ranges, and semantic-distribution row sums.

### 5. Shared Deterministic Diversity Selection

One `select_diverse_memory_queries(...)` implementation is used by:

- cache precompute;
- cache loading;
- the online memory bank.

The selector filters invalid/low-reliability rows, sorts stably by reliability, favors novel classes and spatial cells, and then fills remaining capacity subject to class and cell caps. Schema-v1 caches degrade to reliability plus spatial diversity because their label is unknown (`-1`).

### 6. Target-Age History Selection

The default repaired configuration uses:

```python
history_selection_mode = 'target_age'
history_target_ages = [2.5, 3.5, 4.5]
history_age_tolerance = 0.35
```

History candidates must be strictly past and from the same scene. Each target slot gets at most one frame, and one frame cannot fill multiple slots. Frames outside the dataset split's cache-generatable index space are excluded so strict cache loading does not request records the precompute dataloader cannot produce. Legacy `recent` mode remains supported.

## Memory-Only Training Safety

### Explicit Mode

Formal tuning uses:

```python
query_memory_cfg = dict(
    enabled=True,
    source='cache',
    memory_finetune_mode=True,
    freeze_base_model=True)
```

The mode rejects disabled Memory, online-memory training, or an unfrozen base model. Only `query_memory.*` parameters have `requires_grad=True`.

### Optimizer Scope

MMCV's default optimizer constructor includes frozen parameters. Therefore `mmdet3d/apis/train.py` builds the optimizer directly from `model.query_memory` in Memory-only mode.

After checkpoint loading, `validate_query_memory_training_setup(...)` verifies:

- every trainable parameter is under `query_memory.*`;
- every optimizer parameter belongs to `query_memory.*`;
- every trainable query-memory parameter is present in the optimizer;
- the trainable parameter names and scalar count are logged.

### Frozen TASS Assignment

After `ckpts/epoch_56.pth` is loaded, Memory-only setup runs once:

1. normalize restored `num_stamps_all`;
2. derive `ind_stamps_all` once;
3. rebuild RAP masks once;
4. clone `num_stamps_all`, `ind_stamps_all`, and all RAP masks;
5. set model/head `pretrain=False`;
6. set `pts_bbox_head.freeze_tass_state=True`.

The OPUS loss no longer accumulates into `num_stamps_all` while this guard is active. `set_epoch()` only records the epoch and asserts the frozen state. Every Memory-only iteration uses all six future horizons from the beginning.

### Runtime Train/Eval Modes

The root model remains `training=True`, allowing MMDetection to execute `forward_train()` and compute losses. During Memory-only training:

- `query_memory` remains in training mode;
- every base child module stays in evaluation mode;
- frozen BN statistics and dropout behavior remain unchanged.

Calling `eval()` still puts the whole model into evaluation mode.

### Smoke Optimizer Guard

`QueryMemoryConnectivityOptimizerHook` performs the normal backward/clip/step sequence and also:

- rejects nonzero base-model gradients;
- checks frozen TASS state every iteration;
- logs gradient norms for fusion output/gate, attention Q/K/V, and the motion final layer;
- requires fusion and Q/K/V connectivity after warm-up;
- reports, but does not fabricate, motion-gradient connectivity;
- asserts 7 reads and 1040 fused queries per forward;
- logs valid slots, candidate counts, gates, residuals, effective age, and motion residuals;
- hashes all base parameters and buffers before/after the run to detect any base or BN-state change.

The hook is enabled only by the dedicated smoke config.

## Strict Split Cache Routing

Both repaired STAC-QM configs route caches as follows:

```text
train -> data/query_memory/sparseworld_epoch56_schema2_train
val   -> data/query_memory/sparseworld_epoch56_schema2_val
test  -> data/query_memory/sparseworld_epoch56_schema2_val
```

The shared dataset config no longer overrides the split root. Formal loaders and dataset metadata use `strict=True`, so selected missing histories fail instead of silently becoming empty Memory.

Configs:

- `sparseworld-traj-finetune-stacqm.py`: repaired STAC-QM plumbing;
- `sparseworld-traj-finetune-stacqm-val.py`: matching split-safe validation plumbing;
- `sparseworld-traj-memory-only.py`: formal 12-epoch Memory-only run;
- `sparseworld-traj-memory-only-smoke.py`: 200-iteration connectivity gate.

The original baseline config `sparseworld-traj-finetune.py` is unchanged.

## Verification Tools

### Cache Audit

`tools/query_memory/audit_query_memory_cache.py` builds the actual configured dataset and reuses its target-age history selection plus loader validation. It reports:

- split dataset/cache counts and current-sample coverage;
- schema, source-checkpoint, and source-config distributions;
- missing, corrupt, shape, scene, temporal, and noncausal failures;
- target-age slot coverage;
- reliability min/median/max;
- class histogram and spatial-cell coverage.

It exits nonzero when formal-cache failures are found.

### C0/C1 Identity and Trained Difference

`tools/query_memory/check_query_memory_identity.py` runs a real validation sample twice with identical base state:

- C0: Memory disabled;
- C1: cache Memory enabled.

Zero-initialized mode requires `max_abs_diff <= 1e-6` for current logits/points, future logits/points, trajectory output, and final occupancy-evaluation inputs. It also requires valid Memory, candidates, 7 reads, 1040 fused queries, zero fusion projection, zero motion final layer, and zero applied residual.

Trained mode requires current-frame identity while at least one future output differs, and requires a nonzero trained fusion projection/residual.

## Synthetic Verification Result

Executed:

```bash
/data/jxy/projects/env/bin/python -m pytest -q \
  tests/test_query_memory.py tests/test_query_memory_integration.py
```

Observed:

```text
32 passed, 19 warnings in 4.24s
```

The suite includes schema-v1/v2 behavior, reliability/diversity, effective age, target-age selection, cache-generatable history filtering, zero motion/fusion identity, real `forward_backbone()` 7/1040 instrumentation, Memory-only trainability/module modes, and TASS immutability.

Configuration inheritance/routing, hook registration, tool imports, and Python compilation were also checked. Real-data/GPU commands remain user-run and are not reported as successful here.
