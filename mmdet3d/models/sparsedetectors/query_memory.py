import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


def _as_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _as_float(value[0], default)
    return float(value)


def decode_points_metric(points, pc_range):
    """Decode normalized SparseWorld points to metric coordinates."""
    pc_range = torch.as_tensor(pc_range, device=points.device, dtype=points.dtype)
    out = points.clone()
    out[..., 0] = out[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    out[..., 1] = out[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    out[..., 2] = out[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return out


def encode_points_normalized(points_metric, pc_range):
    """Encode metric points back to SparseWorld normalized coordinates."""
    pc_range = torch.as_tensor(
        pc_range, device=points_metric.device, dtype=points_metric.dtype)
    out = points_metric.clone()
    out[..., 0] = (out[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    out[..., 1] = (out[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    out[..., 2] = (out[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return out


def logits_to_query_confidence(logits):
    """Sigmoid-max-mean confidence for independent semantic logits.

    Args:
        logits (Tensor): [B, Q, R, C_sem]

    Returns:
        Tensor: [B, Q] float32 confidence.
    """
    if logits.dim() != 4:
        raise ValueError(
            'query logits must have shape [B, Q, R, C_sem], '
            f'got {tuple(logits.shape)}')
    return logits.float().sigmoid().amax(dim=-1).mean(dim=-1)


def safe_masked_softmax(scores, mask, dim=-1, eps=_EPS):
    """Softmax that returns strict zeros for fully masked rows."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    scores = scores.float()
    mask = mask.to(device=scores.device)
    has_candidate = mask.any(dim=dim, keepdim=True)
    neg_inf = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, neg_inf)
    row_max = masked_scores.max(dim=dim, keepdim=True).values
    row_max = torch.where(has_candidate, row_max, torch.zeros_like(row_max))
    exp_scores = torch.where(
        mask, torch.exp(masked_scores - row_max), torch.zeros_like(scores))
    denom = exp_scores.sum(dim=dim, keepdim=True)
    weights = exp_scores / denom.clamp_min(eps)
    weights = torch.where(has_candidate, weights, torch.zeros_like(weights))
    return weights


def _inverse_softplus(value):
    value = float(value)
    if value <= 0:
        return torch.tensor(-20.0, dtype=torch.float32)
    return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32)))



@dataclass
class ObservationMemoryEntry:
    query_feat: torch.Tensor
    query_points_metric: torch.Tensor
    query_conf: torch.Tensor
    valid_mask: torch.Tensor
    ego2global: torch.Tensor
    timestamp: float
    frame_idx: Optional[int]
    scene_id: Optional[str]
    sample_idx: Optional[str]


class QueryMemoryBank:
    """Online observation memory for strict sequential batch-size-one eval."""

    def __init__(self,
                 history_frames=3,
                 max_queries_per_frame=256,
                 write_threshold=0.35,
                 max_time_gap=None,
                 bank_size=None,
                 confidence_threshold=None):
        if bank_size is not None:
            history_frames = bank_size
        if confidence_threshold is not None:
            write_threshold = confidence_threshold
        self.history_frames = int(history_frames)
        self.max_queries_per_frame = int(max_queries_per_frame)
        self.write_threshold = float(write_threshold)
        self.max_time_gap = max_time_gap
        self.entries: List[ObservationMemoryEntry] = []
        self._last_scene_id = None
        self._last_sample_idx = None
        self._last_frame_idx = None
        self._last_timestamp = None

    def clear(self):
        self.entries.clear()
        self._last_scene_id = None
        self._last_sample_idx = None
        self._last_frame_idx = None
        self._last_timestamp = None

    def _maybe_reset_for_sequence(self, scene_id, sample_idx, frame_idx,
                                  timestamp):
        if self._last_scene_id is not None and scene_id != self._last_scene_id:
            self.clear()
            return 'scene_change'
        if sample_idx is not None and sample_idx == self._last_sample_idx:
            return 'duplicate_sample'
        if frame_idx is not None and self._last_frame_idx is not None:
            if frame_idx < self._last_frame_idx:
                self.clear()
                return 'frame_rollback'
            if frame_idx == self._last_frame_idx:
                return 'duplicate_frame'
        if timestamp is not None and self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                self.clear()
                return 'time_rollback'
            if timestamp == self._last_timestamp:
                return 'duplicate_timestamp'
            if self.max_time_gap is not None:
                if timestamp - self._last_timestamp > float(self.max_time_gap):
                    self.clear()
                    return 'time_gap_reset'
        return None

    def write(self,
              query_feat,
              query_points_metric,
              cls_scores=None,
              ego2global=None,
              timestamp=None,
              scene_id=None,
              sample_idx=None,
              frame_idx=None,
              query_conf=None,
              source_type='observation'):
        del source_type
        if query_feat.dim() != 3:
            raise ValueError('query_feat must have shape [B, Q, C]')
        if query_feat.shape[0] != 1:
            raise RuntimeError(
                'QueryMemoryBank online mode only supports batch_size=1; '
                f'got B={query_feat.shape[0]}')
        if query_points_metric.dim() != 4:
            raise ValueError(
                'query_points_metric must have shape [B, Q, R, 3]')
        if ego2global is None:
            raise ValueError('ego2global is required for query memory writes')
        if ego2global.dim() == 2:
            ego2global = ego2global.unsqueeze(0)
        if ego2global.shape[0] != 1 or ego2global.shape[-2:] != (4, 4):
            raise ValueError('ego2global must have shape [1, 4, 4] or [4, 4]')
        timestamp = _as_float(timestamp, 0.0)
        frame_idx = None if frame_idx is None else int(frame_idx)
        sample_idx = None if sample_idx is None else str(sample_idx)
        scene_id = None if scene_id is None else str(scene_id)

        reset_reason = self._maybe_reset_for_sequence(
            scene_id, sample_idx, frame_idx, timestamp)
        if reset_reason and reset_reason.startswith('duplicate'):
            return False

        if query_conf is None:
            if cls_scores is None:
                raise ValueError('cls_scores or query_conf is required')
            query_conf = logits_to_query_confidence(cls_scores)
        if query_conf.dim() == 3 and query_conf.shape[-1] == 1:
            query_conf = query_conf.squeeze(-1)
        if query_conf.shape[:2] != query_feat.shape[:2]:
            raise ValueError('query_conf must have shape [B, Q]')

        conf = query_conf[0].detach().float().cpu()
        valid = conf >= self.write_threshold
        valid_inds = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_inds.numel() > self.max_queries_per_frame:
            _, order = torch.topk(conf[valid_inds], self.max_queries_per_frame)
            valid_inds = valid_inds[order]

        entry = ObservationMemoryEntry(
            query_feat=query_feat[0, valid_inds].detach().cpu(),
            query_points_metric=query_points_metric[0, valid_inds].detach().cpu(),
            query_conf=conf[valid_inds].detach().cpu(),
            valid_mask=torch.ones(valid_inds.numel(), dtype=torch.bool),
            ego2global=ego2global[0].detach().float().cpu(),
            timestamp=float(timestamp),
            frame_idx=frame_idx,
            scene_id=scene_id,
            sample_idx=sample_idx)
        self.entries.append(entry)
        while len(self.entries) > self.history_frames:
            self.entries.pop(0)

        self._last_scene_id = scene_id
        self._last_sample_idx = sample_idx
        self._last_frame_idx = frame_idx
        self._last_timestamp = float(timestamp)
        return True

    def read(self,
             scene_id=None,
             sample_idx=None,
             frame_idx=None,
             timestamp=None,
             device=None,
             dtype=torch.float32):
        if not self.entries:
            return None
        timestamp = _as_float(timestamp, 0.0)
        frame_idx = None if frame_idx is None else int(frame_idx)
        sample_idx = None if sample_idx is None else str(sample_idx)
        scene_id = None if scene_id is None else str(scene_id)

        selected = []
        for entry in self.entries:
            if scene_id is not None and entry.scene_id != scene_id:
                continue
            if sample_idx is not None and entry.sample_idx == sample_idx:
                continue
            if frame_idx is not None and entry.frame_idx is not None:
                if entry.frame_idx >= frame_idx:
                    continue
            age = timestamp - entry.timestamp
            if age <= 0:
                continue
            selected.append(entry)
        selected = selected[-self.history_frames:]
        if not selected:
            return None

        max_m = self.max_queries_per_frame
        embed_dims = selected[0].query_feat.shape[-1]
        num_points = selected[0].query_points_metric.shape[-2]
        device = torch.device('cpu') if device is None else device
        feat = torch.zeros(
            1, self.history_frames, max_m, embed_dims, device=device,
            dtype=dtype)
        points = torch.zeros(
            1, self.history_frames, max_m, num_points, 3, device=device,
            dtype=dtype)
        conf = torch.zeros(1, self.history_frames, max_m, device=device)
        valid = torch.zeros(
            1, self.history_frames, max_m, device=device, dtype=torch.bool)
        source_ego = torch.eye(4, device=device).repeat(
            1, self.history_frames, 1, 1)
        age = torch.zeros(1, self.history_frames, max_m, device=device)

        offset = self.history_frames - len(selected)
        for k, entry in enumerate(selected, start=offset):
            n = min(entry.query_feat.shape[0], max_m)
            feat[0, k, :n] = entry.query_feat[:n].to(device=device, dtype=dtype)
            points[0, k, :n] = entry.query_points_metric[:n].to(
                device=device, dtype=dtype)
            conf[0, k, :n] = entry.query_conf[:n].to(device=device)
            source_ego[0, k] = entry.ego2global.to(device=device)
            curr_age = timestamp - entry.timestamp
            age[0, k, :n] = curr_age
            valid[0, k, :n] = entry.valid_mask[:n].to(device=device)
            valid[0, k, :n] &= curr_age > 0
        return dict(
            memory_query_feat=feat,
            memory_points_metric=points,
            memory_conf=conf,
            memory_valid=valid,
            memory_source_ego2global=source_ego,
            memory_age=age)

    def read_all(self, current_timestamp=0.0):
        return self.read(timestamp=current_timestamp)

    def __len__(self):
        return len(self.entries)


class EgoPoseAligner(nn.Module):
    """Align metric points from source ego frames to current ego frame."""

    def __init__(self, pc_range=None):
        super().__init__()
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.register_buffer(
            'pc_range', torch.as_tensor(pc_range, dtype=torch.float32))

    def forward(self, points_metric, source_ego2global, target_ego2global):
        orig_dtype = points_metric.dtype
        points = points_metric.float()
        source = source_ego2global.to(points.device).float()
        target = target_ego2global.to(points.device).float()

        if points.dim() == 4:
            if source.dim() == 2:
                source = source.unsqueeze(0)
            if target.dim() == 2:
                target = target.unsqueeze(0)
            transform = torch.linalg.inv(target) @ source
            rotation = transform[..., :3, :3]
            translation = transform[..., :3, 3]
            aligned = (
                torch.matmul(points, rotation.transpose(-1, -2)[:, None]) +
                translation[:, None, None, :])
        elif points.dim() == 5:
            if source.dim() == 3:
                source = source.unsqueeze(0)
            if target.dim() == 2:
                target = target.unsqueeze(0)
            transform = torch.linalg.inv(target)[:, None] @ source
            rotation = transform[..., :3, :3]
            translation = transform[..., :3, 3]
            aligned = (
                torch.matmul(
                    points, rotation.transpose(-1, -2)[:, :, None]) +
                translation[:, :, None, None, :])
        else:
            raise ValueError(
                'points_metric must have shape [B, M, R, 3] or '
                f'[B, K, M, R, 3], got {tuple(points_metric.shape)}')
        return aligned.to(orig_dtype)

    def align_normalized(self, points_normalized, source_ego2global,
                         target_ego2global):
        metric = decode_points_metric(points_normalized, self.pc_range)
        aligned = self.forward(metric, source_ego2global, target_ego2global)
        return encode_points_normalized(aligned, self.pc_range)


class CausalQueryMemoryAttention(nn.Module):
    """Spatio-temporal, confidence-aware causal multi-head memory read."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.0,
                 lambda_position=1.0,
                 lambda_time=1.0,
                 lambda_confidence=1.0,
                 lambda_pos=None,
                 lambda_conf=None,
                 spatial_radius=12.0,
                 topk=32,
                 max_age=3.0,
                 pc_range=None):
        super().__init__()
        if lambda_pos is not None:
            lambda_position = lambda_pos
        if lambda_conf is not None:
            lambda_confidence = lambda_conf
        if embed_dims % num_heads != 0:
            raise ValueError(
                'STAC-QM requires embed_dims % num_heads == 0, got '
                f'embed_dims={embed_dims}, num_heads={num_heads}')
        if spatial_radius is None or float(spatial_radius) <= 0:
            raise ValueError('spatial_radius must be a positive float')
        if int(topk) <= 0:
            raise ValueError('topk must be a positive integer')
        if float(max_age) <= 0:
            raise ValueError('max_age must be a positive float')
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dims // self.num_heads
        self.spatial_radius = float(spatial_radius)
        self.topk = int(topk)
        self.max_age = float(max_age)
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(float(dropout))
        self.raw_lambda_position = nn.Parameter(
            _inverse_softplus(lambda_position).repeat(self.num_heads))
        self.raw_lambda_time = nn.Parameter(
            _inverse_softplus(lambda_time).repeat(self.num_heads))
        self.raw_lambda_confidence = nn.Parameter(
            _inverse_softplus(lambda_confidence).repeat(self.num_heads))
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.register_buffer(
            'pc_range', torch.as_tensor(pc_range, dtype=torch.float32))
        nn.init.zeros_(self.out_proj.bias)

    def _project_heads(self, layer, tensor, seq_len):
        proj_dtype = layer.weight.dtype
        projected = layer(tensor.to(proj_dtype)).float()
        return projected.view(
            tensor.shape[0], seq_len, self.num_heads,
            self.head_dim).permute(0, 2, 1, 3)

    def forward(self,
                query_feat,
                query_points_metric,
                memory_query_feat,
                memory_points_metric,
                memory_conf,
                memory_age,
                memory_valid):
        B, Q, C = query_feat.shape
        if C != self.embed_dims:
            raise ValueError(
                f'query feature dim {C} does not match embed_dims '
                f'{self.embed_dims}')
        if memory_query_feat is None or memory_query_feat.numel() == 0:
            zeros = query_feat.new_zeros(B, Q, C)
            return zeros, self._empty_diagnostics(B, Q, query_feat.device)

        if memory_query_feat.dim() == 3:
            memory_query_feat = memory_query_feat.unsqueeze(0)
            memory_points_metric = memory_points_metric.unsqueeze(0)
            memory_conf = memory_conf.unsqueeze(0)
            memory_age = memory_age.unsqueeze(0)
            memory_valid = memory_valid.unsqueeze(0)
        if memory_query_feat.shape[0] != B:
            raise ValueError(
                'memory batch size must match query batch size, got '
                f'{memory_query_feat.shape[0]} and {B}')

        K, M = memory_query_feat.shape[1:3]
        N = K * M
        if N == 0:
            zeros = query_feat.new_zeros(B, Q, C)
            return zeros, self._empty_diagnostics(B, Q, query_feat.device)

        mem_feat = memory_query_feat.reshape(B, N, C).to(query_feat.device)
        mem_points = memory_points_metric.reshape(
            B, N, memory_points_metric.shape[-2], 3).to(query_feat.device)
        mem_conf = memory_conf.reshape(B, N).to(query_feat.device).float()
        mem_age = memory_age.reshape(B, N).to(query_feat.device).float()
        mem_valid = memory_valid.reshape(B, N).to(query_feat.device).bool()

        q = self._project_heads(self.q_proj, query_feat, Q)
        k = self._project_heads(self.k_proj, mem_feat, N)
        v = self._project_heads(self.v_proj, mem_feat, N)

        semantic = torch.matmul(q, k.transpose(-1, -2))
        semantic = semantic / math.sqrt(float(self.head_dim))

        q_center = query_points_metric.to(query_feat.device).float().mean(dim=-2)
        m_center = mem_points.float().mean(dim=-2)
        dist_sq = ((q_center[:, :, None] - m_center[:, None])**2).sum(dim=-1)
        dist = torch.sqrt(dist_sq.clamp_min(0.0))

        age_valid = (mem_age > 0.0) & (mem_age <= self.max_age)
        conf_valid = torch.isfinite(mem_conf) & (mem_conf > 0.0)
        base_valid = mem_valid & age_valid & conf_valid
        spatial_valid = dist <= self.spatial_radius
        candidate = base_valid[:, None, :] & spatial_valid

        lambda_position = F.softplus(self.raw_lambda_position).view(
            1, self.num_heads, 1, 1)
        lambda_time = F.softplus(self.raw_lambda_time).view(
            1, self.num_heads, 1, 1)
        lambda_confidence = F.softplus(self.raw_lambda_confidence).view(
            1, self.num_heads, 1, 1)
        radius_norm = self.spatial_radius**2 + _EPS
        age_norm = self.max_age + _EPS
        scores = semantic
        scores = scores - lambda_position * dist_sq[:, None] / radius_norm
        scores = scores - lambda_time * mem_age[:, None, None] / age_norm
        scores = scores + lambda_confidence * torch.log(
            mem_conf.clamp_min(_EPS))[:, None, None]

        topk = min(self.topk, N)
        expanded_candidate = candidate[:, None].expand(
            B, self.num_heads, Q, N)
        topk_scores, topk_indices = torch.topk(
            scores.masked_fill(~expanded_candidate, torch.finfo(scores.dtype).min),
            k=topk,
            dim=-1)
        topk_valid = torch.gather(expanded_candidate, -1, topk_indices)
        weights = safe_masked_softmax(topk_scores, topk_valid, dim=-1)
        weights = self.dropout(weights)

        gather_index = topk_indices[..., None].expand(
            B, self.num_heads, Q, topk, self.head_dim)
        selected_v = torch.gather(
            v[:, :, None].expand(B, self.num_heads, Q, N, self.head_dim),
            3,
            gather_index)
        read_heads = (weights[..., None] * selected_v).sum(dim=-2)
        readout = read_heads.permute(0, 2, 1, 3).reshape(B, Q, C)
        output = self.out_proj(readout.to(self.out_proj.weight.dtype))
        output = output.to(query_feat.dtype)

        has_candidate_head = topk_valid.any(dim=-1)
        has_candidate = has_candidate_head.any(dim=1)
        output = output * has_candidate.unsqueeze(-1).to(output.dtype)

        selected_conf = torch.gather(
            mem_conf[:, None, None].expand(B, self.num_heads, Q, N),
            -1,
            topk_indices)
        selected_dist = torch.gather(
            dist[:, None].expand(B, self.num_heads, Q, N), -1,
            topk_indices)
        selected_age = torch.gather(
            mem_age[:, None, None].expand(B, self.num_heads, Q, N),
            -1,
            topk_indices)
        support_conf_head = (weights * selected_conf).sum(dim=-1)
        avg_dist_head = (weights * selected_dist).sum(dim=-1)
        avg_age_head = (weights * selected_age).sum(dim=-1)
        head_den = has_candidate_head.float().sum(dim=1).clamp_min(1.0)
        support_conf = (
            support_conf_head * has_candidate_head.float()).sum(dim=1) / head_den
        avg_dist = (
            avg_dist_head * has_candidate_head.float()).sum(dim=1) / head_den
        avg_age = (
            avg_age_head * has_candidate_head.float()).sum(dim=1) / head_den
        diagnostics = dict(
            has_candidate=has_candidate,
            support_conf=support_conf,
            candidate_count=candidate.sum(dim=-1).detach(),
            topk_candidate_count=topk_valid.sum(dim=-1).max(dim=1).values.detach(),
            avg_distance=avg_dist.detach(),
            avg_age=avg_age.detach(),
            attention_shape=(B, self.num_heads, Q, topk))
        return output, diagnostics

    def _empty_diagnostics(self, B, Q, device):
        return dict(
            has_candidate=torch.zeros(B, Q, device=device, dtype=torch.bool),
            support_conf=torch.zeros(B, Q, device=device),
            candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            topk_candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            avg_distance=torch.zeros(B, Q, device=device),
            avg_age=torch.zeros(B, Q, device=device),
            attention_shape=(B, self.num_heads, Q, 0))


class ConfidenceGatedFusion(nn.Module):
    """Confidence-gated residual query fusion with exact identity fallback."""

    def __init__(self, embed_dims=256, ffn_dims=512, gate_bias=-4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dims)
        self.norm_h = nn.LayerNorm(embed_dims)
        self.norm_delta = nn.LayerNorm(embed_dims)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dims * 3 + 2, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_bias))

    def forward(self,
                query_feat,
                memory_output,
                current_confidence,
                support_confidence,
                has_candidate):
        orig_dtype = query_feat.dtype
        if current_confidence.dim() == 2:
            current_confidence = current_confidence.unsqueeze(-1)
        if support_confidence.dim() == 2:
            support_confidence = support_confidence.unsqueeze(-1)
        has = has_candidate.to(device=query_feat.device).bool()
        h = memory_output.to(query_feat.device)
        gate_input = torch.cat([
            self.norm_q(query_feat.float()),
            self.norm_h(h.float()),
            self.norm_delta((query_feat - h).float()),
            current_confidence.to(query_feat.device).float(),
            support_confidence.to(query_feat.device).float(),
        ], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        residual = self.out_proj(h.to(self.out_proj.weight.dtype)).to(orig_dtype)
        fused = query_feat + has.unsqueeze(-1).to(orig_dtype) * gate.to(
            orig_dtype) * residual
        diagnostics = dict(avg_gate=(
            gate.detach() * has.unsqueeze(-1).float()).mean(dim=-1))
        return fused, diagnostics


class STACQueryMemory(nn.Module):
    """STAC-QM wrapper: align memory, read it, then gate-fuse residuals."""

    def __init__(self,
                 enabled=True,
                 embed_dims=256,
                 num_heads=8,
                 spatial_radius=12.0,
                 topk=32,
                 max_age=3.0,
                 lambda_position=1.0,
                 lambda_time=1.0,
                 lambda_confidence=1.0,
                 dropout=0.0,
                 pc_range=None,
                 **kwargs):
        super().__init__()
        del kwargs
        self.enabled = bool(enabled)
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.aligner = EgoPoseAligner(pc_range)
        self.attention = CausalQueryMemoryAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            lambda_position=lambda_position,
            lambda_time=lambda_time,
            lambda_confidence=lambda_confidence,
            spatial_radius=spatial_radius,
            topk=topk,
            max_age=max_age,
            pc_range=pc_range)
        self.fusion = ConfidenceGatedFusion(
            embed_dims=embed_dims, ffn_dims=embed_dims * 2)

    def forward(self,
                query_feat,
                query_points_metric,
                current_confidence,
                memory=None,
                target_ego2global=None,
                **memory_kwargs):
        if not self.enabled:
            return query_feat, dict(enabled=False)
        if memory is None:
            memory = memory_kwargs
        if not memory:
            return query_feat, self._identity_diagnostics(query_feat)
        memory_query_feat = memory.get('memory_query_feat')
        memory_points_metric = memory.get('memory_points_metric')
        memory_conf = memory.get('memory_conf')
        memory_valid = memory.get('memory_valid')
        memory_age = memory.get('memory_age')
        source_ego = memory.get('memory_source_ego2global')
        required = [
            memory_query_feat, memory_points_metric, memory_conf, memory_valid,
            memory_age
        ]
        if any(x is None for x in required):
            raise KeyError(
                'STAC-QM memory requires memory_query_feat, '
                'memory_points_metric, memory_conf, memory_valid, and '
                'memory_age')
        if memory_valid.numel() == 0 or not memory_valid.to(
                query_feat.device).bool().any():
            return query_feat, self._identity_diagnostics(query_feat)

        if memory_query_feat.dim() == 3:
            memory_query_feat = memory_query_feat.unsqueeze(0)
            memory_points_metric = memory_points_metric.unsqueeze(0)
            memory_conf = memory_conf.unsqueeze(0)
            memory_valid = memory_valid.unsqueeze(0)
            memory_age = memory_age.unsqueeze(0)
            if source_ego is not None and source_ego.dim() == 3:
                source_ego = source_ego.unsqueeze(0)
        if memory_query_feat.shape[0] != query_feat.shape[0]:
            raise ValueError(
                'STAC-QM does not share memory across batch samples: '
                f'query B={query_feat.shape[0]}, memory B='
                f'{memory_query_feat.shape[0]}')
        if source_ego is None or target_ego2global is None:
            raise ValueError(
                'memory_source_ego2global and target_ego2global are required '
                'to align valid STAC-QM memory')
        aligned_points = self.aligner(
            memory_points_metric.to(query_feat.device),
            source_ego.to(query_feat.device),
            target_ego2global.to(query_feat.device))
        memory_output, attn_diag = self.attention(
            query_feat=query_feat,
            query_points_metric=query_points_metric,
            memory_query_feat=memory_query_feat.to(query_feat.device),
            memory_points_metric=aligned_points,
            memory_conf=memory_conf.to(query_feat.device),
            memory_age=memory_age.to(query_feat.device),
            memory_valid=memory_valid.to(query_feat.device))
        fused, gate_diag = self.fusion(
            query_feat,
            memory_output,
            current_confidence,
            attn_diag['support_conf'],
            attn_diag['has_candidate'])
        diagnostics = dict(attn_diag)
        diagnostics.update(gate_diag)
        return fused, diagnostics

    def _identity_diagnostics(self, query_feat):
        B, Q = query_feat.shape[:2]
        device = query_feat.device
        return dict(
            has_candidate=torch.zeros(B, Q, device=device, dtype=torch.bool),
            support_conf=torch.zeros(B, Q, device=device),
            candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            topk_candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            avg_distance=torch.zeros(B, Q, device=device),
            avg_age=torch.zeros(B, Q, device=device),
            avg_gate=torch.zeros(B, Q, device=device),
            attention_shape=(B, self.attention.num_heads, Q, 0))
