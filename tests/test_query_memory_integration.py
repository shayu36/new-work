"""STAC-QM modeling-repair integration tests (synthetic, CPU only).

These exercise the end-to-end wiring of the modeling-structure repairs
(Problems 1-6) across the online bank, the cache loader, and the STAC-QM
module -- WITHOUT any dataset, checkpoint, GPU, or training. Everything runs on
hand-built tensors so the tests are deterministic and fast.

Documented as NOT run here (require the full dataset / GPU / checkpoints and are
therefore out of scope for this modeling-structure-only phase):
  * tools/query_memory/precompute_query_memory.py end-to-end cache generation
    (needs nuScenes + a trained checkpoint).
  * A real nuScenes forward pass through SparseWorld4DTraj.forward_backbone
    (needs the dataset and the base checkpoint).
  * Any metric / mIoU evaluation.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qm = load_module(
    'query_memory_under_test_integration',
    ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
loader_mod = load_module(
    'loading_query_memory_under_test_integration',
    ROOT / 'mmdet3d/datasets/pipelines/loading_query_memory.py')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_cache(path, sample_idx, scene_id, frame_idx, timestamp,
               conf=(0.9, 0.8), reliability=None, label=None,
               schema_version=1, C=4, R=2):
    M = len(conf)
    cache = dict(
        schema_version=schema_version,
        sample_idx=sample_idx,
        scene_id=scene_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        ego2global=torch.eye(4),
        query_feat=torch.arange(M * C, dtype=torch.float32).reshape(M, C),
        query_points_metric=torch.zeros(M, R, 3),
        query_conf=torch.tensor(list(conf), dtype=torch.float32),
        valid_mask=torch.ones(M, dtype=torch.bool),
        pc_range=[-1, -1, -1, 1, 1, 1],
        embed_dims=C,
        num_points=R,
        num_classes=3,
        source_config='cfg.py',
        source_checkpoint='model.pth')
    if schema_version >= 2:
        cache['query_reliability'] = torch.tensor(
            list(reliability), dtype=torch.float32)
        cache['query_label'] = torch.tensor(list(label), dtype=torch.long)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def _write_bank_frame(bank, scene, sample, frame, ts, Q=3, C=8, R=2):
    feat = torch.randn(1, Q, C)
    points = torch.zeros(1, Q, R, 3)
    logits = torch.zeros(1, Q, R, 4)
    ego = torch.eye(4).unsqueeze(0)
    return bank.write(
        feat, points, cls_scores=logits, ego2global=ego, timestamp=ts,
        scene_id=scene, sample_idx=sample, frame_idx=frame)


# ---------------------------------------------------------------------------
# Problem 6 - online bank target-age slot assignment
# ---------------------------------------------------------------------------
def test_online_bank_target_age_slot_assignment_and_tolerance():
    # targets 1s / 2s match written frames; 5s has no frame within tolerance.
    bank = qm.QueryMemoryBank(
        history_selection_mode='target_age',
        history_target_ages=[1.0, 2.0, 5.0],
        history_age_tolerance=0.35,
        retention_seconds=20.0,
        max_bank_entries=16,
        write_threshold=0.0,
        max_queries_per_frame=4)
    for frame, ts in enumerate([0.0, 1.0, 2.0]):
        assert _write_bank_frame(bank, 'scene-a', f'sample-{frame}', frame, ts)

    mem = bank.read('scene-a', 'sample-cur', 3, 3.0, device=torch.device('cpu'))
    assert mem is not None
    # 3 slots (one per target age); slot0<-age1 (frame2), slot1<-age2 (frame1),
    # slot2 (target 5s) unmatched -> stays fully invalid.
    assert mem['memory_valid'].shape[1] == 3
    assert mem['memory_valid'][0, 0].sum().item() == 3
    assert mem['memory_valid'][0, 1].sum().item() == 3
    assert mem['memory_valid'][0, 2].sum().item() == 0
    # ages land on their target slots.
    assert abs(float(mem['memory_age'][0, 0].max()) - 1.0) < 1e-5
    assert abs(float(mem['memory_age'][0, 1].max()) - 2.0) < 1e-5
    # schema-v2 outputs are present.
    assert 'memory_reliability' in mem and 'memory_label' in mem


def test_online_bank_target_age_all_out_of_tolerance_returns_none():
    bank = qm.QueryMemoryBank(
        history_selection_mode='target_age',
        history_target_ages=[1.0, 2.0, 3.0],
        history_age_tolerance=0.2,
        retention_seconds=20.0,
        write_threshold=0.0,
        max_queries_per_frame=4)
    for frame, ts in enumerate([0.0, 1.0, 2.0]):
        _write_bank_frame(bank, 'scene-a', f'sample-{frame}', frame, ts)
    # read at 3.4s -> ages 3.4/2.4/1.4, none within 0.2 of any target.
    mem = bank.read('scene-a', 'sample-cur', 3, 3.4, device=torch.device('cpu'))
    assert mem is None


# ---------------------------------------------------------------------------
# Problem 5/6 - cache loader target-age slots + schema versions
# ---------------------------------------------------------------------------
def test_loader_target_age_slot_placement_schema_v2(tmp_path):
    root = tmp_path / 'cache'
    make_cache(root / 'scene-a' / 'sample-0.pt', 'sample-0', 'scene-a', 0, 4.0,
               conf=(0.9, 0.8), reliability=(0.9, 0.8), label=(0, 1),
               schema_version=2)
    make_cache(root / 'scene-a' / 'sample-1.pt', 'sample-1', 'scene-a', 1, 2.0,
               conf=(0.7, 0.6), reliability=(0.7, 0.6), label=(2, 3),
               schema_version=2)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_selection_mode='target_age',
        history_target_ages=[1.0, 2.0, 3.0], max_queries_per_frame=3,
        strict=True, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-cur', scene_token='scene-a', frame_idx=5,
        timestamp=5.0,
        query_memory_history_infos=[
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=4.0, slot_index=0, target_age=1.0),
            dict(sample_idx='sample-1', scene_id='scene-a', frame_idx=1,
                 timestamp=2.0, slot_index=2, target_age=3.0),
        ])

    out = loader(results)
    assert out['memory_query_feat'].shape == (3, 3, 4)
    assert out['memory_valid'][0].sum().item() == 2   # slot 0 filled
    assert out['memory_valid'][1].sum().item() == 0   # slot 1 empty
    assert out['memory_valid'][2].sum().item() == 2   # slot 2 filled
    # schema-v2 labels survive selection (reliability-descending order).
    assert out['memory_label'][0, :2].tolist() == [0, 1]
    assert out['memory_label'][2, :2].tolist() == [2, 3]
    # base_age = current_ts - history_ts.
    assert abs(float(out['memory_age'][0, 0]) - 1.0) < 1e-5
    assert abs(float(out['memory_age'][2, 0]) - 3.0) < 1e-5


def test_loader_schema_v1_reliability_label_fallback(tmp_path):
    root = tmp_path / 'cache'
    make_cache(root / 'scene-a' / 'sample-0.pt', 'sample-0', 'scene-a', 0, 4.0,
               conf=(0.9, 0.8), schema_version=1)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_frames=3, max_queries_per_frame=3,
        strict=True, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-cur', scene_token='scene-a', frame_idx=5,
        timestamp=5.0,
        query_memory_history_infos=[
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=4.0),
        ])
    out = loader(results)
    # recent mode right-aligns the single frame into the last slot.
    slot = out['memory_valid'].shape[0] - 1
    assert out['memory_valid'][slot].sum().item() == 2
    # v1 has no reliability/label -> reliability degrades to conf, label to -1.
    assert torch.allclose(
        out['memory_reliability'][slot, :2], torch.tensor([0.9, 0.8]))
    assert out['memory_label'][slot, :2].tolist() == [-1, -1]


# ---------------------------------------------------------------------------
# Problems 1/2/3 - STAC forward: future-aware age + causal filter + motion
# ---------------------------------------------------------------------------
def _single_candidate_memory(base_age, C=8, R=2):
    return dict(
        memory_query_feat=torch.randn(1, 1, 1, C),
        memory_points_metric=torch.zeros(1, 1, 1, R, 3),
        memory_conf=torch.full((1, 1, 1), 0.9),
        memory_reliability=torch.full((1, 1, 1), 0.9),
        memory_valid=torch.ones(1, 1, 1, dtype=torch.bool),
        memory_source_ego2global=torch.eye(4).repeat(1, 1, 1, 1),
        memory_age=torch.full((1, 1, 1), float(base_age)))


def test_future_offset_effective_age_and_causal_filter():
    stac = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=3.0, dropout=0.0)
    query_feat = torch.randn(1, 1, 8)
    query_points = torch.zeros(1, 1, 2, 3)
    query_conf = torch.ones(1, 1)
    target = torch.eye(4).unsqueeze(0)
    memory = _single_candidate_memory(base_age=2.9)

    fused0, diag0 = stac(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target, future_offset=0.0)
    fused1, diag1 = stac(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target, future_offset=0.5)

    # base age unchanged; effective age shifts by exactly the future offset.
    assert torch.allclose(diag0['base_age'], diag1['base_age'])
    assert abs(float(diag1['effective_age_mean'] -
                     diag0['effective_age_mean']) - 0.5) < 1e-5
    # 2.9s is within max_age=3s; 2.9+0.5=3.4s exceeds it -> candidate filtered.
    assert bool(diag0['has_candidate'].any())
    assert not bool(diag1['has_candidate'].any())
    # zero-init fusion => exact identity at initialization either way.
    assert torch.equal(fused0, query_feat)
    assert torch.equal(fused1, query_feat)


def test_motion_compensation_toggle_and_trained_shift():
    query_feat = torch.randn(1, 1, 8)
    query_points = torch.zeros(1, 1, 2, 3)
    query_conf = torch.ones(1, 1)
    target = torch.eye(4).unsqueeze(0)
    memory = _single_candidate_memory(base_age=2.0)

    # motion ON but zero-init -> no velocity, no residual.
    stac_on = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=8.0, dropout=0.0, motion_compensation=True)
    _, diag_init = stac_on(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target)
    assert float(diag_init['motion_velocity_norm']) == 0.0
    assert float(diag_init['motion_residual_mean']) == 0.0

    # after "training" the last layer, velocity + residual become non-zero.
    with torch.no_grad():
        stac_on.motion_compensator.mlp[-1].weight.fill_(0.05)
        stac_on.motion_compensator.mlp[-1].bias.fill_(0.05)
    _, diag_trained = stac_on(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target)
    assert float(diag_trained['motion_velocity_norm']) > 0.0
    assert float(diag_trained['motion_residual_mean']) > 0.0

    # motion OFF -> module emits no motion diagnostics at all.
    stac_off = qm.STACQueryMemory(
        enabled=True, embed_dims=8, num_heads=2, spatial_radius=100.0,
        topk=8, max_age=8.0, dropout=0.0, motion_compensation=False)
    _, diag_off = stac_off(
        query_feat, query_points, query_conf, memory=memory,
        target_ego2global=target)
    assert 'motion_velocity_norm' not in diag_off


# ---------------------------------------------------------------------------
# Baseline compatibility - disabled STAC is a tensor-level identity
# ---------------------------------------------------------------------------
def test_disabled_stac_is_tensor_identity_regardless_of_memory():
    stac = qm.STACQueryMemory(enabled=False, embed_dims=8, num_heads=2)
    query_feat = torch.randn(1, 5, 8)
    memory = _single_candidate_memory(base_age=1.0)
    fused, diag = stac(
        query_feat, torch.zeros(1, 5, 2, 3), torch.ones(1, 5),
        memory=memory, target_ego2global=torch.eye(4).unsqueeze(0),
        future_offset=0.5)
    assert torch.equal(fused, query_feat)  # byte-for-byte identity
    assert diag['enabled'] is False
