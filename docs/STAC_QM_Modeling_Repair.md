# STAC-QM Modeling-Structure Repair

This document records the 2026-08 modeling-structure-only repair for STAC-QM in SparseWorld4DTraj.

## Scope And Non-Scope

Completed in this phase:

- repaired the STAC-QM control flow and data structures for query-memory modeling;
- added schema-v2 memory reliability / label fields with schema-v1 fallback;
- added synthetic CPU tests covering the repaired behavior;
- updated STAC-QM configs and cache tooling interfaces.

Not completed in this phase:

- no training was started;
- no full nuScenes evaluation was run;
- no full train/val query cache was generated;
- no metric improvement is claimed;
- no optimizer, learning-rate schedule, epoch schedule, checkpoint loading, resume logic, TASS stage logic, `num_stamps_all`, or `ind_stamps_all` training-stage logic was intentionally changed.

This repair intentionally does **not** convert SparseWorld queries into track queries, split dynamic/static queries, parallelize SCF future prediction, copy OccWorld fixed-grid tokens, add VQ-VAE, or solve any untrained-parameter issue for `query_memory.*`.

## Pre-Repair Audit

The previous STAC-QM wiring had six modeling issues:

1. **Repeated memory reads.** Active queries could re-read the same real history memory during each SCF interval, producing repeated memory injection rather than one causal read per query.
2. **Future queries used base history age only.** Scheduled future query groups did not add their future offset to the memory age used by causal filtering, time penalty, and diagnostics.
3. **No learnable residual motion compensation.** Ego-pose alignment handled ego motion only; there was no zero-initialized object-motion residual path.
4. **Confidence was overloaded.** The cache stored `query_conf`, but did not store per-query semantic reliability, label, margin, entropy, or semantic distribution.
5. **Query selection was not shared.** Cache generation, cache loading, and the online bank did not share a single deterministic reliability/class/spatial diversity selector.
6. **History selection was recent-only.** The data path could not choose target-age history slots such as 2.5s / 3.5s / 4.5s with tolerance and no duplicate-frame assignment.

## Repaired Modeling Flow

### 1. Single Memory Read Per Query

Observation queries read real history memory once before the SCF loop:

```text
obs_query_feat
  -> STAC-QM(memory, future_offset=0.0)
  -> SCF interval loop
```

Scheduled future query group `k` reads real history memory once when appended:

```text
scheduled_query_group_k
  -> STAC-QM(memory, future_offset=(k + 1) * frame_interval)
  -> enters SCF scene update
```

Already-fused active queries are not re-read in later intervals. With the default schedule this bounds STAC-QM reads to:

```text
1 observation read + 6 scheduled-group reads = 7 reads max
720 observation queries + 320 scheduled queries = 1040 fused queries total
```

This is a control-flow repair only; it does not change the SparseWorld query schedule itself.

### 2. Future-Aware Effective Age

Memory caches and the online bank store only base age:

```text
base_age = current_timestamp - history_timestamp
```

Cached timestamps are never mutated. STAC-QM receives an explicit future offset and computes:

```text
effective_age = base_age + future_offset
```

where:

```text
future_offset = 0.0                         for observation queries
future_offset = (k + 1) * frame_interval    for scheduled future group k
```

The causal filter, max-age filter, time penalty, motion compensation input, and diagnostics all use `effective_age`.

### 3. Zero-Initialized Motion Compensation

A learnable `QueryMotionCompensator` predicts a per-memory-query velocity:

```text
v_i = v_max * tanh(MLP([LN(m_i), phi(effective_age_i)]))
P_hat_i = P_ego_aligned_i + effective_age_i * v_i
```

The final MLP layer is zero-initialized, so at initialization:

```text
v_i = 0
P_hat_i = P_ego_aligned_i
```

Therefore the repaired model is exactly ego-alignment-only at initialization. The module can be disabled with `motion_compensation=False`.

### 4. Per-Query Semantic Reliability

Schema v2 stores additional per-query fields:

- `query_semantic_distribution`
- `query_label`
- `query_margin`
- `query_entropy`
- `query_reliability`

Reliability is computed from the point-averaged semantic distribution:

```text
top1 = max_c p_c
margin = top1 - top2
normalized_entropy = 1 - H(p) / log(C)
query_reliability = mean(top1, margin, normalized_entropy)
```

The result is clamped to `[0, 1]`. This is semantic reliability over occupancy classes, **not foreground probability**.

The old `query_conf` remains in the cache for schema-v1 compatibility. When a schema-v1 cache is loaded:

```text
query_reliability = query_conf
query_label = -1
```

`CausalQueryMemoryAttention` uses reliability in the score:

```text
score = semantic_score
      - lambda_position * distance^2 / radius^2
      - lambda_time * effective_age / max_age
      + lambda_reliability * log(query_reliability + eps)
```

Legacy `lambda_confidence` is mapped to the reliability weight with a one-time deprecation warning when legacy config keys are used.

### 5. Shared Deterministic Diversity Selection

The same function is used by:

- `QueryMemoryBank.write(...)`
- `LoadQueryMemoryFromFiles`
- `tools/query_memory/precompute_query_memory.py`

Selection order is deterministic:

1. filter invalid / too-low-reliability queries;
2. sort by reliability descending with stable tie behavior;
3. prefer novel spatial cells and novel classes;
4. fill remaining capacity under `max_per_spatial_cell` and `max_per_class` caps.

When schema-v1 labels are unavailable (`query_label == -1` for all queries), class diversity is disabled and selection degrades to reliability + spatial diversity.

### 6. Target-Age History Selection

`history_selection_mode='target_age'` supports target slots such as:

```python
history_target_ages = [2.5, 3.5, 4.5]
history_age_tolerance = 0.35
visual_history_window = 2.0
```

The dataset selects strictly-past same-scene frames closest to each target age. A frame cannot occupy two slots. A slot remains invalid when no candidate is within tolerance. The dataset attaches:

```text
slot_index
target_age
```

The loader preserves these slots instead of right-aligning them. The legacy `recent` mode remains available and retains the previous right-aligned history behavior.

## File-Level Changes

- `mmdet3d/models/sparsedetectors/query_memory.py`
  - added `compute_query_reliability(...)`, `compute_effective_age(...)`, and `select_diverse_memory_queries(...)`;
  - added schema-v2 reliability / label storage to the online bank;
  - added target-age online-bank history selection;
  - added zero-initialized `QueryMotionCompensator`;
  - made attention use effective age and reliability-aware scoring;
  - added diagnostics for support reliability, effective age, and motion residuals.

- `mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py`
  - added full `query_memory_cfg` defaults and legacy-key mapping;
  - changed SCF integration so observation queries and scheduled query groups read memory at most once;
  - passed explicit `future_offset` into STAC-QM;
  - passed schema-v2 memory fields through cache context;
  - preserved disabled-mode tensor identity.

- `mmdet3d/datasets/pipelines/loading_query_memory.py`
  - added schema-v2 validation and schema-v1 fallback;
  - added target-age slot placement;
  - added shared deterministic diversity selection;
  - now emits `memory_reliability` and `memory_label`.

- `tools/query_memory/precompute_query_memory.py`
  - bumped cache schema to v2;
  - writes `query_reliability` and `query_label`;
  - uses the shared diversity selector;
  - added CLI knobs for reliability and diversity caps.

- `mmdet3d/datasets/nuscenes_dataset_occ_trajectory.py`
  - added dataset-side target-age history collection;
  - added `slot_index` and `target_age` metadata;
  - retained recent-mode compatibility.

- `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py`
  - expanded `query_memory_cfg` with repair parameters;
  - configured target-age history selection and diversity selection;
  - added schema-v2 loader keys to `Collect4D`;
  - preserved optimizer / LR / runner / checkpoint fields.

- `configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm-val.py`
  - applied the same modeling-repair config structure;
  - preserved val-specific cache root and `log_diagnostics=True`.

- `tests/test_query_memory.py`
  - extended unit tests for reliability, future-aware age, diversity selection, motion no-op, and reliability-aware attention.

- `tests/test_query_memory_integration.py`
  - added synthetic CPU integration tests across bank, loader, and STAC-QM forward behavior.

- `docs/STAC_QM_Implementation.md`
  - updated to point to this modeling-repair document and describe the repaired SCF integration/cache schema.

## Baseline Compatibility

When query memory is disabled:

```python
query_memory_cfg = dict(enabled=False)
```

or equivalent config disables memory, the model does not read memory, align poses, run attention/fusion, or write the online bank. The repaired STAC-QM wrapper returns the original query tensor unchanged at tensor level:

```text
fused_query_feat is query_feat content-wise identical
```

The SCF loop still uses the original query schedule, causal mask, branches, losses, and output dictionaries. The repair also avoids changes to optimizer, LR schedule, runner, checkpoint loading, resume behavior, and TASS training-stage logic.

Schema compatibility:

```text
schema v2: uses query_reliability + query_label
schema v1: query_reliability = query_conf, query_label = -1
```

## Test Results

Synthetic CPU tests were run with:

```bash
/data/jxy/projects/env/bin/python -m pytest tests/test_query_memory.py tests/test_query_memory_integration.py -q
```

Observed result:

```text
26 passed, 19 warnings in 4.32s
```

The 26 passing tests cover:

1. ego-pose identity / translation / rotation / batch isolation;
2. sigmoid-max-mean query confidence;
3. multi-head attention shape, age/radius/top-k filtering;
4. all-invalid / empty-memory safe identity behavior;
5. STAC disabled exact tensor identity;
6. online-bank read-after-write causality and scene isolation;
7. online-bank batch-size-one enforcement;
8. cache-loader padding and non-strict missing-cache behavior;
9. strict missing-cache error reporting;
10. configuration validation errors;
11. reliability keys, ranges, labels, and empty-query behavior;
12. peaked semantic logits producing higher reliability than uniform logits;
13. observation and scheduled effective-age offsets;
14. deterministic reliability ordering;
15. spatial diversity caps;
16. class diversity caps;
17. schema-v1 unknown-label degradation;
18. zero-initialized motion compensator exact no-op;
19. reliability-weighted attention preference;
20. online-bank target-age slot assignment and tolerance;
21. online-bank all-out-of-tolerance target-age read returns `None`;
22. cache-loader target-age slot placement with schema-v2 fields;
23. cache-loader schema-v1 reliability / label fallback;
24. future offset affecting effective age and causal max-age filtering;
25. motion compensation diagnostics after simulated trained shift;
26. disabled STAC identity regardless of memory contents.

Not run in this phase:

- full query-cache generation;
- full nuScenes train/val data loading;
- SparseWorld4DTraj forward on real images;
- training;
- full nuScenes evaluation;
- metric comparison.

## Remaining Work For Later Phases

Before making any performance claims, a later phase still needs to:

1. generate real schema-v2 train/val cache files with the approved checkpoint;
2. train the newly introduced `query_memory.*` parameters under the intended schedule;
3. run validation / evaluation on the intended split;
4. compare against the baseline with identical training/eval settings;
5. inspect checkpoint missing/unexpected keys separately if checkpoint policy is changed in a future phase.
