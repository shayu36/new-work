# STAC-QM Implementation

This repository now implements STAC-QM: Spatio-Temporally Aligned and Confidence-Gated Causal Query Memory for SparseWorld trajectory forecasting.

## Scope

The first version uses only observation memory from real past frames. It does not add predicted query memory, instance tracking, fixed query-index matching, or an independent confidence head.

> **Modeling-structure repair (2026-08).** Six modeling-structure repairs have
> since been applied (single-read control flow, future-aware age, a
> zero-initialized motion residual, per-query semantic reliability, deterministic
> diversity selection, and target-age history selection). See
> [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md) for the authoritative
> description. This section documents the original v1 behavior; where the two
> differ, the repair doc governs. No training or evaluation has been run as part
> of the repair — only the modeling structure and synthetic CPU tests changed.

## Memory Record

Each observation memory record stores CPU-detached tensors before STAC-QM fusion:

- `query_feat`: raw RAP observation query feature `[M, C]`
- `query_points_metric`: full decoded metric point set `[M, R, 3]`
- `query_conf`: sigmoid-max-mean confidence `[M]`
- `valid_mask`: valid cache rows `[M]`
- `ego2global`: source ego-to-global pose `[4, 4]`
- `timestamp`, `frame_idx`, `scene_id`, `sample_idx`

Only `ind_stamps_all == 0` observation queries are written. Future scheduled queries and fused/SCF outputs are never written.

## Training Data Flow

```text
past real observation frame
-> baseline SparseWorld RAP precompute script
-> per-sample query cache under <cache_root>/<scene>/<sample>.pt
-> NuScenesDatasetOccpancy4DTraj builds same-scene strictly past history_infos
-> LoadQueryMemoryFromFiles loads/pads K x M cache tensors
-> SparseWorld4DTraj source='cache'
-> every SCF interval reads aligned observation memory
-> confidence-gated residual updates active/scheduled query features
-> original SCF refine/classification/velocity/loss flow
```

`configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune-stacqm.py` enables stage A by setting `freeze_base_model=True`, so only `query_memory.*` parameters remain trainable.

## Online Inference Flow

```text
read query_memory_bank for current frame metadata
-> run RAP and STAC-QM-enhanced SCF
-> after prediction, write raw current observation query to query_memory_bank
```

The online bank is strict batch-size one. It clears on scene change, frame rollback, timestamp rollback, or configured time-gap reset. It filters the current frame and duplicate frames, so the current sample cannot read itself.

## SCF Integration

STAC-QM is inserted without rewriting the original SCF loop:

- The observation queries are fused **once**, before the SCF interval loop begins.
- After the scheduled queries for an interval are appended, only that new scheduled slice is fused **once**, with a future-aware age offset of `(interval + 1) * frame_interval`.
- Already-fused active queries are never re-read on later intervals — each query reads real history memory at most once (repair Problem 1).
- The original interval count, query schedule, causal mask, position encoding, `ego_cross_attn`, branches, moving mask, `refine_points`, detach behavior, losses, and output dictionaries are preserved.

## Attention And Fusion

`CausalQueryMemoryAttention` performs real multi-head attention with `[B, H, Q, d_head]` and `[B, H, M, d_head]` tensors. Candidate filtering requires:

```text
memory_valid
age = current_timestamp - source_timestamp > 0
age <= max_age
distance <= spatial_radius
per-query, per-head Top-K
```

The masked softmax is safe for fully invalid rows: weights and readout are exactly zero and no NaN/Inf is produced.

`ConfidenceGatedFusion` uses:

```text
LN(q), LN(h), LN(q-h), current_confidence, support_confidence
```

The final output is:

```text
q + has_candidate * gate * W_o(h)
```

There is no unconditional LayerNorm on the final output, so empty memory and no-candidate rows are exact identity.

## Cache Precompute

`tools/query_memory/precompute_query_memory.py` runs a baseline config/checkpoint in eval/no_grad mode and saves cache files. As of the modeling repair it writes **schema version 2**, adding per-query `query_reliability` and `query_label`, and uses the shared deterministic diversity selection. Schema-v1 caches remain loadable (reliability degrades to `query_conf`, label to `-1`). It refuses to overwrite existing files unless `--overwrite` is passed.

Example:

```bash
python tools/query_memory/precompute_query_memory.py \
  --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py \
  --checkpoint ckpts/baseline.pth \
  --split train \
  --output-dir data/query_memory/sparseworld_train \
  --max-queries-per-frame 256 \
  --write-threshold 0.35
```

Do not commit generated cache files.

## Baseline Compatibility

When `query_memory_cfg is None` or `enabled=False`, SparseWorld4DTraj does not read cache, align poses, run attention/fusion, or write the online bank. The original forward path, loss inputs, output format, and old checkpoint behavior are preserved. New checkpoint missing keys should be limited to `query_memory.*` when STAC-QM is enabled.

## Synthetic Validation

`tests/test_query_memory.py` and `tests/test_query_memory_integration.py` use only temporary synthetic tensors and temporary cache files (CPU only, no dataset/checkpoint/GPU). Together they cover ego-pose alignment, confidence computation, multi-head shapes, age/radius/top-k filtering, safe all-invalid behavior, exact identity fallback, scene isolation, online read-before-write behavior, loader padding, strict missing-cache behavior, configuration errors, per-query reliability, future-aware effective age, deterministic diversity selection (spatial/class caps + v1 degradation), the zero-initialized motion compensator, reliability-weighted attention, target-age history slot assignment (bank + loader), schema-v1/v2 loader behavior, and the future-offset causal filter. See [STAC_QM_Modeling_Repair.md](STAC_QM_Modeling_Repair.md#test-results) for the full case list and the tests that could not be run in this phase.
