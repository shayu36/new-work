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
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    num_classes = max(3, max(label) + 1) if label is not None else 3
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
        num_classes=num_classes,
        source_config='cfg.py',
        source_checkpoint='model.pth')
    if schema_version >= 2:
        rel = torch.tensor(list(reliability), dtype=torch.float32)
        labels = torch.tensor(list(label), dtype=torch.long)
        distribution = torch.zeros(M, cache['num_classes'])
        distribution[torch.arange(M), labels] = 1.0
        cache['query_semantic_distribution'] = distribution
        cache['query_label'] = labels
        cache['query_margin'] = rel.clone()
        cache['query_entropy'] = torch.zeros(M)
        cache['query_reliability'] = rel
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


def test_dataset_target_age_uses_only_cacheable_split_frames():
    from mmdet3d.datasets.nuscenes_dataset_occ_trajectory import (
        NuScenesDatasetOccpancy4DTraj)

    dataset = NuScenesDatasetOccpancy4DTraj.__new__(
        NuScenesDatasetOccpancy4DTraj)
    dataset.data_infos = [
        dict(token=f'sample-{index}', scene_token='scene-a',
             scene_name='scene-a', frame_idx=index,
             timestamp=index * 1_000_000)
        for index in range(5)
    ]
    dataset.query_memory_history_frames = 2
    dataset.query_memory_history_selection_mode = 'target_age'
    dataset.query_memory_history_target_ages = [1.0, 3.0]
    dataset.query_memory_history_age_tolerance = 0.1
    dataset._query_memory_cacheable_indices = {1, 3, 4}

    histories = dataset._build_query_memory_history_infos(4)
    assert [item['sample_idx'] for item in histories] == [
        'sample-3', 'sample-1']
    assert [item['slot_index'] for item in histories] == [0, 1]
    assert len({item['sample_idx'] for item in histories}) == 2


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


def test_loader_rejects_incomplete_schema_v2(tmp_path):
    root = tmp_path / 'cache'
    path = root / 'scene-a' / 'sample-0.pt'
    make_cache(
        path, 'sample-0', 'scene-a', 0, 4.0, conf=(0.9, 0.8),
        reliability=(0.9, 0.8), label=(0, 1), schema_version=2)
    cache = torch.load(path, map_location='cpu')
    cache.pop('query_entropy')
    torch.save(cache, path)
    loader = loader_mod.LoadQueryMemoryFromFiles(
        cache_root=str(root), history_frames=1, max_queries_per_frame=3,
        strict=True, embed_dims=4, num_points=2)
    results = dict(
        sample_idx='sample-cur', scene_token='scene-a', frame_idx=5,
        timestamp=5.0,
        query_memory_history_infos=[
            dict(sample_idx='sample-0', scene_id='scene-a', frame_idx=0,
                 timestamp=4.0),
        ])
    with pytest.raises(KeyError, match='query_entropy'):
        loader(results)


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


# ---------------------------------------------------------------------------
# Real SparseWorld4DTraj control-flow integration (synthetic CPU fakes)
# ---------------------------------------------------------------------------
class _ShapeModule(nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.out_dims = int(out_dims)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, tensor):
        return tensor.new_zeros(*tensor.shape[:-1], self.out_dims) + \
            self.anchor * 0


class _FakeEgoCrossAttention(nn.Module):
    def forward(self, query_pos, query_feat, memory_pos, memory_feat):
        del query_pos, memory_pos, memory_feat
        return query_feat.new_zeros(query_feat.shape), None


class _ForwardHead(nn.Module):
    def __init__(self, ind_stamps_all):
        super().__init__()
        self.register_buffer('ind_stamps_all', ind_stamps_all)
        self.points_scale = None


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ind_mask = None


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttention()


class _FakeTemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('num_stamps_all', torch.tensor([
            [10, 1], [9, 2], [1, 10]], dtype=torch.long))
        self.pretrain = True
        self.freeze_tass_state = False
        decoder = nn.Module()
        decoder.decoder_layers = nn.ModuleList([_FakeDecoderLayer()])
        self.transformer = nn.Module()
        self.transformer.decoder = decoder

    def reset_mask(self):
        mask = torch.zeros(3, 3)
        for stamp in range(2):
            rows = (self.ind_stamps_all == stamp).nonzero(as_tuple=True)[0]
            cols = (self.ind_stamps_all > stamp).nonzero(as_tuple=True)[0]
            if rows.numel() and cols.numel():
                grid_row, grid_col = torch.meshgrid(rows, cols, indexing='ij')
                mask[grid_row, grid_col] = -1e5
        self.transformer.decoder.decoder_layers[0].self_attn.ind_mask = mask


def _make_sparseworld_shell(memory_finetune_mode=True):
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import \
        SparseWorld4DTraj

    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    nn.Module.__init__(model)
    model.query_memory_enabled = True
    model.memory_finetune_mode = memory_finetune_mode
    model.query_memory_source = 'cache'
    model.query_memory_cfg = dict(
        freeze_base_model=True, memory_finetune_mode=memory_finetune_mode)
    model.query_memory = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))
    model.img_backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))
    model.img_neck = nn.Linear(4, 4)
    model.pts_bbox_head = _FakeTemporalHead()
    model.num_query = 2
    model.num_fu_query = [1]
    model.num_fu_frames = 1
    model.pretrain = True
    model.frozen_num_stamps_all = None
    model.frozen_ind_stamps_all = None
    model._frozen_rap_masks = None
    model._configure_query_memory_trainability()
    return model


def test_memory_finetune_trainability_optimizer_and_module_modes():
    model = _make_sparseworld_shell()
    model._freeze_memory_finetune_temporal_state()
    optimizer = torch.optim.Adam(model.query_memory.parameters(), lr=1e-4)
    summary = model.validate_query_memory_training_setup(optimizer=optimizer)
    bad_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(RuntimeError, match='non-query-memory'):
        model.validate_query_memory_training_setup(optimizer=bad_optimizer)
    assert summary['trainable_names']
    assert all(name.startswith('query_memory.')
               for name in summary['trainable_names'])
    assert all(not param.requires_grad for param in model.img_backbone.parameters())

    model.train()
    assert model.training is True
    assert model.query_memory.training is True
    assert model.img_backbone.training is False
    assert model.img_neck.training is False
    assert model.pts_bbox_head.training is False
    model.eval()
    assert model.training is False
    assert model.query_memory.training is False


def test_memory_finetune_requires_strict_loader_tensors():
    model = _make_sparseworld_shell()
    with pytest.raises(KeyError, match='strict loader'):
        model._query_memory_context(
            {}, [], torch.device('cpu'), torch.float32)


def test_memory_finetune_tass_state_is_frozen_across_epochs():
    model = _make_sparseworld_shell()
    model._freeze_memory_finetune_temporal_state()
    frozen_num = model.frozen_num_stamps_all.clone()
    frozen_ind = model.frozen_ind_stamps_all.clone()
    frozen_mask = model._frozen_rap_masks[0].clone()

    model.set_epoch(0)
    model.set_epoch(11)
    assert torch.equal(model.pts_bbox_head.num_stamps_all, frozen_num)
    assert torch.equal(model.pts_bbox_head.ind_stamps_all, frozen_ind)
    assert torch.equal(
        model.pts_bbox_head.transformer.decoder.decoder_layers[
            0].self_attn.ind_mask,
        frozen_mask)
    assert model.pretrain is False
    assert model.pts_bbox_head.pretrain is False
    assert model.pts_bbox_head.freeze_tass_state is True

    model.pts_bbox_head.num_stamps_all[0, 0] += 1
    with pytest.raises(RuntimeError, match='num_stamps_all changed'):
        model.set_epoch(12)


def test_forward_backbone_runtime_reads_seven_groups_and_1040_queries(
        monkeypatch):
    import mmdet3d.models.sparsedetectors.sparseworld_4d_traj as sw_mod
    SparseWorld4DTraj = sw_mod.SparseWorld4DTraj
    monkeypatch.setattr(sw_mod, 'device', torch.device('cpu'))

    model = SparseWorld4DTraj.__new__(SparseWorld4DTraj)
    nn.Module.__init__(model)
    counts = [720, 60, 60, 60, 60, 40, 40]
    ind_stamps = torch.cat([
        torch.full((count,), stamp, dtype=torch.long)
        for stamp, count in enumerate(counts)
    ])
    B, C, R = 1, 8, 2
    Q = int(ind_stamps.numel())
    outs = dict(
        query_feat=torch.zeros(B, Q, C),
        all_refine_pts=[torch.zeros(B, Q, R, 3)],
        all_cls_scores=[torch.zeros(B, Q, R, 17)])

    model.query_memory_enabled = True
    model.memory_finetune_mode = True
    model.query_memory_log_diagnostics = False
    model.query_memory_frame_interval = 0.5
    model.num_refines = R
    model.num_fu_frames = 6
    model.finetune_epoch = 5
    model.curr_epoch = 0
    model.pretrain = False
    model.pc_range = torch.tensor([-40., -40., -1., 40., 40., 5.4])
    model.pts_bbox_head = _ForwardHead(ind_stamps)
    model.plan_head = _ShapeModule(C)
    model.points_scale_branch = _ShapeModule(3)
    model.ego_cross_attn = _FakeEgoCrossAttention()
    model.traj_head = _ShapeModule(2)
    model.position_encoder = _ShapeModule(C)
    model.reg_branch = _ShapeModule(R * 3)
    model.cls_branch = _ShapeModule(R * 17)
    model.vel_branch = _ShapeModule(R * 2)

    model.extract_feat = types.MethodType(
        lambda self, img, img_metas: None, model)
    model.pts_bbox_head.forward = types.MethodType(
        lambda self, img_feats, img_metas: outs, model.pts_bbox_head)
    model._query_memory_context = types.MethodType(
        lambda self, kwargs, img_metas, device, dtype: object(), model)
    model.refine_points = types.MethodType(
        lambda self, points, delta: points, model)
    model.trans_points = types.MethodType(
        lambda self, points, delta, transform: points, model)

    calls = []

    def _record(self, feat, pos, cls, img_metas, memory, future_offset=0.0):
        del pos, cls, img_metas, memory
        calls.append((int(feat.shape[1]), float(future_offset)))
        return feat

    model._apply_query_memory_once = types.MethodType(_record, model)
    model.training = True
    img = torch.zeros(B, 1)
    img_metas = [dict(ego2lidar=torch.eye(4).numpy())]
    outputs = model.forward_backbone(
        img, img_metas,
        temporal_ego_states=[torch.zeros(B, 1, 4)],
        temporal_trajs=torch.zeros(B, 6, 2))

    assert [count for count, _ in calls] == counts
    assert [offset for _, offset in calls] == [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert len(calls) == 7
    assert sum(count for count, _ in calls) == 1040
    assert len(outputs['forecast_semantics_list']) == 6
    # The first read is only the 720 observation queries; later reads are only
    # newly scheduled groups, never the growing active set.
    assert max(count for count, _ in calls[1:]) == 60
