# STAC-QM Implementation

This repository now implements STAC-QM: Spatio-Temporally Aligned and Confidence-Gated Causal Query Memory for SparseWorld trajectory forecasting.

## Scope

The first version uses only observation memory from real past frames. It does not add predicted query memory, instance tracking, fixed query-index matching, future ego poses, motion residual networks, or an independent confidence head.

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

- At the start of each interval, all active queries are fused once before `ego_cross_attn` and scene updates.
- After scheduled queries for that interval are appended, only the new scheduled slice is fused once before it first enters the scene update branches.
- Existing active queries are not fused twice in the same interval.
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

`tools/query_memory/precompute_query_memory.py` runs a baseline config/checkpoint in eval/no_grad mode and saves schema version 1 cache files. It refuses to overwrite existing files unless `--overwrite` is passed.

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

`tests/test_query_memory.py` uses only temporary synthetic tensors and temporary cache files. It covers ego-pose alignment, confidence computation, multi-head shapes, age/radius/top-k filtering, safe all-invalid behavior, exact identity fallback, scene isolation, online read-before-write behavior, loader padding, strict missing-cache behavior, and configuration errors.
