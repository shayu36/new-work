import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points


class MemoryEntry:
    __slots__ = ['query_feat', 'query_pos', 'confidence',
                 'ego2global', 'timestamp', 'source_type', 'valid_mask']

    def __init__(self, query_feat, query_pos, confidence, ego2global,
                 timestamp, source_type, valid_mask):
        self.query_feat = query_feat
        self.query_pos = query_pos
        self.confidence = confidence
        self.ego2global = ego2global
        self.timestamp = timestamp
        self.source_type = source_type
        self.valid_mask = valid_mask


class QueryMemoryBank:
    """Ring buffer storing recent observation query memories."""

    def __init__(self, bank_size=5, confidence_threshold=0.3):
        self.bank_size = bank_size
        self.confidence_threshold = confidence_threshold
        self.entries: List[MemoryEntry] = []

    def write(self, query_feat, query_pos, cls_scores, ego2global, timestamp,
              source_type='observation'):
        confidence = cls_scores.detach().sigmoid().max(dim=-1)[0].mean(dim=-1)
        valid_mask = confidence > self.confidence_threshold
        entry = MemoryEntry(
            query_feat=query_feat.detach(),
            query_pos=query_pos.detach(),
            confidence=confidence.detach(),
            ego2global=ego2global.detach(),
            timestamp=timestamp,
            source_type=source_type,
            valid_mask=valid_mask.detach(),
        )
        if len(self.entries) >= self.bank_size:
            self.entries.pop(0)
        self.entries.append(entry)

    def read_all(self, current_timestamp=0.0):
        if not self.entries:
            return None
        all_feat, all_pos, all_conf = [], [], []
        all_ego2global, all_time_delta, all_valid = [], [], []
        sizes = []

        for entry in self.entries:
            B, Q = entry.query_feat.shape[:2]
            all_feat.append(entry.query_feat)
            all_pos.append(entry.query_pos)
            all_conf.append(entry.confidence)
            all_ego2global.append(entry.ego2global)
            dt = current_timestamp - entry.timestamp
            all_time_delta.append(
                entry.query_feat.new_full((B, Q), abs(dt)))
            all_valid.append(entry.valid_mask)
            sizes.append(Q)

        return dict(
            mem_feat=torch.cat(all_feat, dim=1),
            mem_pos=torch.cat(all_pos, dim=1),
            mem_confidence=torch.cat(all_conf, dim=1),
            mem_ego2global=all_ego2global,
            mem_time_delta=torch.cat(all_time_delta, dim=1),
            mem_valid_mask=torch.cat(all_valid, dim=1),
            mem_sizes=sizes,
        )

    def clear(self):
        self.entries.clear()

    def __len__(self):
        return len(self.entries)


class EgoPoseAligner(nn.Module):
    """Transform historical query positions to current ego frame."""

    def __init__(self, pc_range):
        super().__init__()
        self.register_buffer(
            'pc_range', torch.as_tensor(pc_range, dtype=torch.float32))

    def forward(self, mem_pos, mem_ego2global_list, curr_ego2global, mem_sizes):
        pc_range = self.pc_range
        pos_physical = decode_points(mem_pos, pc_range)
        B = pos_physical.shape[0]
        inv_curr = torch.linalg.inv(curr_ego2global)

        aligned_parts = []
        offset = 0
        for i, size in enumerate(mem_sizes):
            part = pos_physical[:, offset:offset + size]
            T = torch.matmul(inv_curr, mem_ego2global_list[i])
            flat = part.reshape(B, -1, 3)
            transformed = (torch.matmul(flat, T[:, :3, :3].transpose(1, 2))
                           + T[:, None, :3, 3])
            aligned_parts.append(transformed.reshape(part.shape))
            offset += size

        aligned = torch.cat(aligned_parts, dim=1)
        return encode_points(aligned, pc_range)


class CausalQueryMemoryAttention(nn.Module):
    """Cross-attention from current queries to aligned historical memory."""

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1,
                 lambda_pos=0.01, lambda_time=0.1, lambda_conf=0.5,
                 spatial_radius=None, pc_range=None):
        super().__init__()
        self.head_dim = embed_dims // num_heads
        self.lambda_pos = lambda_pos
        self.lambda_time = lambda_time
        self.lambda_conf = lambda_conf
        self.spatial_radius = spatial_radius
        self.register_buffer(
            'pc_range', torch.as_tensor(pc_range, dtype=torch.float32))

        self.W_q = nn.Linear(embed_dims, embed_dims)
        self.W_k = nn.Linear(embed_dims, embed_dims)
        self.W_v = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feat, query_pos, mem_feat, mem_pos_aligned,
                mem_confidence, mem_time_delta, mem_valid_mask):
        B, Q, C = query_feat.shape

        q = self.W_q(query_feat)
        k = self.W_k(mem_feat)
        semantic_score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        q_center = decode_points(query_pos, self.pc_range).mean(dim=2)
        m_center = decode_points(mem_pos_aligned, self.pc_range).mean(dim=2)
        dist_sq = ((q_center.unsqueeze(2) - m_center.unsqueeze(1)) ** 2).sum(-1)
        spatial_penalty = -self.lambda_pos * dist_sq

        time_penalty = -self.lambda_time * mem_time_delta.unsqueeze(1)
        conf_bonus = self.lambda_conf * torch.log(mem_confidence + 1e-6).unsqueeze(1)

        attn = semantic_score + spatial_penalty + time_penalty + conf_bonus

        if self.spatial_radius is not None:
            attn = attn.masked_fill(dist_sq > self.spatial_radius ** 2, -1e5)

        attn = attn.masked_fill(~mem_valid_mask.unsqueeze(1), -1e5)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v = self.W_v(mem_feat)
        h = torch.matmul(attn_weights, v)
        return self.out_proj(h)


class ConfidenceGatedFusion(nn.Module):
    """Gate-controlled residual fusion of memory retrieval into query features."""

    def __init__(self, embed_dims=256, ffn_dims=512):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2 + 1, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm = nn.LayerNorm(embed_dims)
        nn.init.constant_(self.gate_mlp[-2].bias, -2.0)

    def forward(self, query_feat, memory_output, query_confidence):
        gate_input = torch.cat(
            [query_feat, memory_output, query_confidence], dim=-1)
        gate = self.gate_mlp(gate_input)
        enhanced = query_feat + gate * self.ffn(memory_output)
        return self.norm(enhanced)
