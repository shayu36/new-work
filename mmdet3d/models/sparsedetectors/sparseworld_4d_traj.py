# Copyright (c) Phigent Robotics. All rights reserved.
from mmdet3d.models.detectors.bevdet_occ import BEVStereo4DOCC
from .opus import OPUS
import torch.nn.functional as F
import torch
import time
import warnings
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import MultiheadAttention
from torch import nn
import numpy as np
from mmdet3d.models import builder
from .opus_transformer import OPUSSelfAttention, OPUSCrossAttention
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import get_dist_info
from mmdet3d.models.detectors.loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss, l1_loss, l2_loss
from mmdet3d.models.detectors.lovasz_softmax import lovasz_softmax
from IPython import embed
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points, trans_coords,get_matched_inds
from mmdet3d.models.sparsedetectors.query_memory import (
    STACQueryMemory, QueryMemoryBank, decode_points_metric,
    logits_to_query_confidence
)
from mmdet3d.models.heads import DownScaleModule3DCustom
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
device = torch.device('cuda')
# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947,
                                   497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,
                                   141625221, 2307405309])
import time
# from ptflops import get_model_complexity_info
from thop import profile

def Scatter(src_dict):
    for key, value in src_dict.items():
        if isinstance(value, torch.Tensor):
            src_dict[key] = value.cuda()
        if isinstance(value, dict):
            src_dict[key] = Scatter(value)
        if isinstance(value, list):
            if isinstance(value[0], dict):
                src_dict[key] = [Scatter(v) for v in value]
            if isinstance(value[0], torch.Tensor):
                src_dict[key] = [v.cuda() for v in value]
    return src_dict


@DETECTORS.register_module()
class SparseWorld4DTraj(OPUS):
    def __init__(self,
                 out_dim=32,
                 dataset_type='Nuscenes',
                 num_classes=18,
                 test_threshold=8.5,
                 drop_out=0.1,
                 use_3d_loss=True,
                 if_pretrain=False,
                 if_render=True,
                 if_post_finetune=False,
                 finetune_epoch = 0,
                 num_out_query=600,
                 empty_idx=17,
                 use_focal_loss=True,
                 balance_cls_weight=True,
                 final_softplus=True,
                 memory_enabled=False,
                 memory_bank_size=5,
                 memory_embed_dims=256,
                 memory_num_heads=8,
                 memory_dropout=0.1,
                 memory_confidence_threshold=0.3,
                 memory_lambda_pos=0.01,
                 memory_lambda_time=0.1,
                 memory_lambda_conf=0.5,
                 query_memory_cfg=None,
                 **kwargs):
        self.memory_self_noise = kwargs.pop('memory_self_noise', 0.0)
        super(SparseWorld4DTraj, self).__init__(**kwargs)
        self.dataset_type = dataset_type
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.num_refines = self.pts_bbox_head.transformer.num_refines[-1]
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        # self.if_pretrain = if_pretrain
        self.if_render = if_render
        self.if_post_finetune = if_post_finetune
        self.empty_idx = empty_idx
        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction="mean"
            )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        self.velocity_dim = 3
        self.past_frame = 5
        self.pc_range = self.pts_bbox_head.pc_range

        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim)
        )
        self.ego_cross_attn = OPUSCrossAttention(self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)

        self.position_encoder = nn.Sequential(
            nn.Linear(4 * self.num_refines, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.reg_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 3)
        )

        self.vel_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 2)
        )

        self.cls_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 17)
        )
        self.points_scale_branch = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2),
        )
        self.l2_loss = l2_loss()

        self.box_mode_3d = Box3DMode.LIDAR
        self.planning_metric = None
        self.finetune_epoch = finetune_epoch

        self.pred_num = torch.zeros(18).cuda()

        self.gt_traj = list()
        self.tau = list()

        legacy_memory_cfg = dict(
            enabled=memory_enabled,
            source='online',
            history_frames=memory_bank_size,
            max_queries_per_frame=256,
            write_threshold=memory_confidence_threshold,
            embed_dims=memory_embed_dims,
            num_heads=memory_num_heads,
            spatial_radius=12.0,
            topk=32,
            max_age=3.0,
            lambda_position=memory_lambda_pos,
            lambda_time=memory_lambda_time,
            lambda_confidence=memory_lambda_conf,
            dropout=memory_dropout,
            max_time_gap=2.0,
            log_diagnostics=False,
            freeze_base_model=False,
        )
        self.query_memory_cfg = self._build_query_memory_cfg(
            query_memory_cfg, legacy_memory_cfg)
        self.query_memory_enabled = bool(self.query_memory_cfg.get('enabled', False))
        self.memory_enabled = self.query_memory_enabled
        self.query_memory_source = self.query_memory_cfg.get('source', 'cache')
        self.query_memory_frame_interval = float(
            self.query_memory_cfg.get('frame_interval', 0.5))
        self.query_memory_log_diagnostics = bool(
            self.query_memory_cfg.get('log_diagnostics', False))
        self.query_memory_diagnostics = []
        self.query_memory = None
        self.query_memory_bank = None
        if self.query_memory_enabled:
            if self.query_memory_source not in ('cache', 'online'):
                raise ValueError(
                    'query_memory_cfg.source must be "cache" or "online", '
                    f'got {self.query_memory_source!r}')
            self.query_memory = STACQueryMemory(
                enabled=True,
                embed_dims=self.query_memory_cfg['embed_dims'],
                num_heads=self.query_memory_cfg['num_heads'],
                spatial_radius=self.query_memory_cfg['spatial_radius'],
                topk=self.query_memory_cfg['topk'],
                max_age=self.query_memory_cfg['max_age'],
                lambda_position=self.query_memory_cfg['lambda_position'],
                lambda_time=self.query_memory_cfg['lambda_time'],
                lambda_reliability=self.query_memory_cfg.get(
                    'lambda_reliability'),
                lambda_confidence=self.query_memory_cfg['lambda_confidence'],
                dropout=self.query_memory_cfg['dropout'],
                motion_compensation=self.query_memory_cfg.get(
                    'motion_compensation', True),
                max_velocity=self.query_memory_cfg.get('max_velocity', 20.0),
                pc_range=self.pc_range)
            if self.query_memory_source == 'online':
                self.query_memory_bank = QueryMemoryBank(
                    history_frames=self.query_memory_cfg['history_frames'],
                    max_queries_per_frame=self.query_memory_cfg[
                        'max_queries_per_frame'],
                    write_threshold=self.query_memory_cfg['write_threshold'],
                    max_time_gap=self.query_memory_cfg.get('max_time_gap'),
                    history_selection_mode=self.query_memory_cfg.get(
                        'history_selection_mode', 'recent'),
                    history_target_ages=self.query_memory_cfg.get(
                        'history_target_ages', [2.5, 3.5, 4.5]),
                    history_age_tolerance=self.query_memory_cfg.get(
                        'history_age_tolerance', 0.35),
                    visual_history_window=self.query_memory_cfg.get(
                        'visual_history_window', 2.0),
                    retention_seconds=self.query_memory_cfg.get(
                        'retention_seconds', 5.0),
                    max_bank_entries=self.query_memory_cfg.get(
                        'max_bank_entries', 16),
                    min_reliability=self.query_memory_cfg.get(
                        'min_reliability', 0.0),
                    spatial_cell_size=self.query_memory_cfg.get(
                        'spatial_cell_size', 4.0),
                    max_per_spatial_cell=self.query_memory_cfg.get(
                        'max_per_spatial_cell', 16),
                    max_per_class=self.query_memory_cfg.get(
                        'max_per_class', 64))
            # Freeze STAC-QM params during training (not trained yet)
            for param in self.query_memory.parameters():
                param.requires_grad = False
        if self.query_memory_cfg.get('freeze_base_model', False):
            self._freeze_base_for_query_memory()

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if epoch<self.finetune_epoch:
            self.pretrain = True
            self.pts_bbox_head.pretrain = True
            if getattr(self.pts_bbox_head, 'num_stamps_all', None) is not None:
                self.pts_bbox_head.num_stamps_all[:] = 1  # avoid diving 0
        else:
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            num_stamps = self.pts_bbox_head.num_stamps_all / torch.sum(self.pts_bbox_head.num_stamps_all, dim=-1,
                                                                       keepdim=True)
            self.pts_bbox_head.ind_stamps_all = get_matched_inds(num_stamps, [self.num_query] + self.num_fu_query)

            self.pts_bbox_head.reset_mask()


    def trans_points(self, points_proposal, points_delta, trans_matrix):

        inv_trans_matrix = torch.linalg.inv(trans_matrix.cpu()).cuda()
        points_proposal = decode_points(points_proposal, self.pc_range)
        # points_proposal = points_proposal.mean(dim=2, keepdim=True) fengze
        new_points = torch.matmul(points_proposal, trans_matrix[..., :3, :3].transpose(1, 2)) + trans_matrix[..., None,
                                                                                                :3, 3]
        new_points = new_points + points_delta
        new_points = torch.matmul(new_points, inv_trans_matrix[..., :3, :3].transpose(1, 2)) + inv_trans_matrix[...,
                                                                                               None, :3, 3]

        return encode_points(new_points, self.pc_range)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def loss_traj(self, pred_traj, gt_traj, ego_interval):
        loss_dict = dict()
        loss_dict[f'loss_traj_{str(ego_interval)}s'] = self.l2_loss(pred_traj, gt_traj)

        return loss_dict

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img = [img] if img is None else img

        result = self.simple_test(img_metas[0], img[0], **kwargs)

        

        return result


    # legacy query_memory_cfg keys -> canonical STAC-QM keys. Mapping is applied
    # with a single explicit warning and NEVER silently changes meaning.
    _QUERY_MEMORY_LEGACY_KEYS = {
        'lambda_confidence': 'lambda_reliability',
        'memory_conf': 'min_reliability',
        'confidence_threshold': 'write_threshold',
        'bank_size': 'history_frames',
    }

    def _build_query_memory_cfg(self, query_memory_cfg, legacy_cfg):
        cfg = dict(
            enabled=False,
            source='cache',
            # --- history selection (Problem 6) ---
            history_selection_mode='target_age',
            history_frames=3,
            history_target_ages=[2.5, 3.5, 4.5],
            history_age_tolerance=0.35,
            visual_history_window=2.0,
            retention_seconds=5.0,
            max_bank_entries=16,
            frame_interval=0.5,
            # --- diversity selection (Problem 5) ---
            max_queries_per_frame=256,
            min_reliability=0.0,
            spatial_cell_size=4.0,
            max_per_spatial_cell=16,
            max_per_class=64,
            write_threshold=0.35,
            # --- attention / fusion ---
            embed_dims=256,
            num_heads=8,
            spatial_radius=12.0,
            topk=32,
            max_age=8.0,
            lambda_position=1.0,
            lambda_time=1.0,
            lambda_reliability=1.0,
            # kept for the CausalQueryMemoryAttention legacy alias
            lambda_confidence=1.0,
            dropout=0.0,
            # --- motion compensation (Problem 3) ---
            motion_compensation=True,
            max_velocity=20.0,
            # --- misc ---
            max_time_gap=None,
            schema_version=2,
            log_diagnostics=False,
            freeze_base_model=False,
        )
        user_cfg = query_memory_cfg
        if user_cfg is None:
            # legacy constructor-arg path (memory_* kwargs)
            if legacy_cfg.get('enabled', False):
                user_cfg = legacy_cfg
            else:
                return cfg
        cfg.update(self._map_legacy_query_memory_keys(dict(user_cfg)))
        return cfg

    def _map_legacy_query_memory_keys(self, user_cfg):
        legacy_present = [
            k for k in self._QUERY_MEMORY_LEGACY_KEYS if k in user_cfg]
        # lambda_confidence is still a live kwarg of CausalQueryMemoryAttention;
        # only *remap* it when the canonical lambda_reliability is absent.
        for legacy_key in legacy_present:
            canonical = self._QUERY_MEMORY_LEGACY_KEYS[legacy_key]
            if canonical in user_cfg:
                continue
            user_cfg[canonical] = user_cfg[legacy_key]
        if legacy_present:
            warnings.warn(
                'STAC-QM query_memory_cfg received legacy keys '
                f'{legacy_present}; mapped to '
                f'{[self._QUERY_MEMORY_LEGACY_KEYS[k] for k in legacy_present]}'
                '. These legacy keys are deprecated and will be removed; '
                'update the config to the canonical keys.',
                DeprecationWarning)
        return user_cfg

    def _freeze_base_for_query_memory(self):
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith('query_memory.')

    def validate_query_memory_training_setup(self):
        if not self.query_memory_enabled:
            return
        if self.query_memory_source == 'online' and self.training:
            return  # allowed: STAC-QM params frozen, memory skipped during training
        if self.query_memory_cfg.get('freeze_base_model', False):
            trainable = [
                name for name, param in self.named_parameters()
                if param.requires_grad
            ]
            bad = [name for name in trainable if not name.startswith('query_memory.')]
            if bad:
                raise RuntimeError(
                    'freeze_base_model=True allows only query_memory.* to be '
                    f'trainable, but found: {bad[:20]}')
            if not trainable:
                raise RuntimeError(
                    'freeze_base_model=True left no trainable parameters.')

    def _meta_scene_id(self, meta):
        scene_id = meta.get('scene_token', None)
        if scene_id is None:
            scene_id = meta.get('scene_name', meta.get('scene_id', None))
        return None if scene_id is None else str(scene_id)

    def _meta_sample_idx(self, meta):
        sample_idx = meta.get('sample_idx', meta.get('sample_token', None))
        return None if sample_idx is None else str(sample_idx)

    def _meta_frame_idx(self, meta):
        frame_idx = meta.get('frame_idx', None)
        if frame_idx is None and 'curr' in meta:
            frame_idx = meta['curr'].get('frame_idx', None)
        return None if frame_idx is None else int(frame_idx)

    def _meta_timestamp(self, meta):
        timestamp = meta.get('timestamp', None)
        if timestamp is None:
            timestamp = meta.get('img_timestamp', None)
        if isinstance(timestamp, (list, tuple)):
            timestamp = timestamp[0]
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.detach().cpu().reshape(-1)[0].item()
        if timestamp is None:
            raise KeyError('timestamp or img_timestamp is required for STAC-QM')
        return float(timestamp)

    def _current_ego2global_tensor(self, img_metas, device):
        ego2globals = []
        for meta in img_metas:
            if 'ego2global' not in meta:
                raise KeyError('ego2global is required for STAC-QM alignment')
            value = meta['ego2global']
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            ego2globals.append(np.asarray(value, dtype=np.float32))
        return torch.tensor(
            np.stack(ego2globals), device=device, dtype=torch.float32)

    def _tensor_from_memory_kwargs(self, kwargs, key, device, dtype=None):
        value = kwargs[key]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(device=device)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def _query_memory_context(self, kwargs, img_metas, device, dtype):
        if not self.query_memory_enabled:
            return None
        if self.query_memory_source == 'cache':
            required = [
                'memory_query_feat', 'memory_points_metric', 'memory_conf',
                'memory_valid', 'memory_source_ego2global', 'memory_age'
            ]
            missing = [key for key in required if key not in kwargs]
            if missing:
                return None
            context = dict(
                memory_query_feat=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_query_feat', device, dtype),
                memory_points_metric=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_points_metric', device, dtype),
                memory_conf=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_conf', device, torch.float32),
                memory_valid=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_valid', device).bool(),
                memory_source_ego2global=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_source_ego2global', device, torch.float32),
                memory_age=self._tensor_from_memory_kwargs(
                    kwargs, 'memory_age', device, torch.float32))
            # schema-v2 optional fields (Problem 4). Absent for v1 caches, in
            # which case STAC-QM falls back to memory_conf as reliability.
            if 'memory_reliability' in kwargs:
                context['memory_reliability'] = self._tensor_from_memory_kwargs(
                    kwargs, 'memory_reliability', device, torch.float32)
            if 'memory_label' in kwargs:
                context['memory_label'] = self._tensor_from_memory_kwargs(
                    kwargs, 'memory_label', device, torch.long)
            return context

        if self.training:
            return None  # STAC-QM params frozen, memory skipped during training
        _, world_size = get_dist_info()
        if world_size != 1:
            raise RuntimeError(
                'STAC-QM online bank requires single-GPU sequential eval; '
                f'got world_size={world_size}')
        if len(img_metas) != 1:
            raise RuntimeError(
                'STAC-QM online bank only supports batch_size=1; got '
                f'{len(img_metas)} samples')
        meta = img_metas[0]
        return self.query_memory_bank.read(
            scene_id=self._meta_scene_id(meta),
            sample_idx=self._meta_sample_idx(meta),
            frame_idx=self._meta_frame_idx(meta),
            timestamp=self._meta_timestamp(meta),
            device=device,
            dtype=dtype)

    def _apply_query_memory_once(self, query_feat, query_pos, query_cls,
                                 img_metas, memory_context, future_offset=0.0):
        if not self.query_memory_enabled:
            return query_feat
        if memory_context is None:
            return query_feat
        query_points_metric = decode_points_metric(query_pos, self.pc_range)
        query_conf = logits_to_query_confidence(query_cls.detach())
        target_ego2global = self._current_ego2global_tensor(
            img_metas, query_feat.device)
        fused, diagnostics = self.query_memory(
            query_feat=query_feat,
            query_points_metric=query_points_metric,
            current_confidence=query_conf,
            memory=memory_context,
            target_ego2global=target_ego2global,
            future_offset=future_offset)
        if self.query_memory_log_diagnostics:
            self.query_memory_diagnostics.append({
                key: value.detach().cpu() if isinstance(value, torch.Tensor)
                else value
                for key, value in diagnostics.items()
            })
        return fused

    def _prepare_online_query_memory(self, img_metas):
        if not self.query_memory_enabled or self.query_memory_source != 'online':
            return
        if len(img_metas) != 1:
            raise RuntimeError(
                'STAC-QM online bank only supports batch_size=1; got '
                f'{len(img_metas)} samples')
        meta = img_metas[0]
        scene_id = self._meta_scene_id(meta)
        frame_idx = self._meta_frame_idx(meta)
        timestamp = self._meta_timestamp(meta)
        bank = self.query_memory_bank
        if bank._last_scene_id is not None and scene_id != bank._last_scene_id:
            bank.clear()
            return
        if frame_idx is not None and bank._last_frame_idx is not None:
            if frame_idx < bank._last_frame_idx:
                bank.clear()
                return
        if timestamp is not None and bank._last_timestamp is not None:
            if timestamp < bank._last_timestamp:
                bank.clear()
                return
            max_time_gap = self.query_memory_cfg.get('max_time_gap', None)
            if max_time_gap is not None:
                if timestamp - bank._last_timestamp > float(max_time_gap):
                    bank.clear()

    def _memory_write(self, curr_query_feat, curr_query_pos, curr_query_cls,
                      img_metas):
        if not self.query_memory_enabled or self.query_memory_source != 'online':
            return
        if curr_query_feat.shape[0] != 1:
            raise RuntimeError(
                'STAC-QM online bank only supports batch_size=1; got '
                f'B={curr_query_feat.shape[0]}')
        meta = img_metas[0]
        ego2global = self._current_ego2global_tensor(
            img_metas, curr_query_feat.device)
        points_metric = decode_points_metric(curr_query_pos, self.pc_range)
        self.query_memory_bank.write(
            curr_query_feat,
            points_metric,
            cls_scores=curr_query_cls,
            ego2global=ego2global,
            timestamp=self._meta_timestamp(meta),
            scene_id=self._meta_scene_id(meta),
            sample_idx=self._meta_sample_idx(meta),
            frame_idx=self._meta_frame_idx(meta))

    def _check_scene_change(self, img_metas):
        self._prepare_online_query_memory(img_metas)
        return False

    def forward_backbone(self,img,img_metas,**kwargs):

        B = img.shape[0]
        ego_states = kwargs['temporal_ego_states'][0]
        bs, _, dim_ = ego_states.shape
        ego_states = ego_states.view((bs, 1, dim_))
        ego_feat = self.plan_head(ego_states)
        points_scale = self.points_scale_branch(ego_feat)
        points_scale = torch.tanh(points_scale)
        self.pts_bbox_head.points_scale = (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8

        if self.training:
            img_feats = self.extract_feat(img, img_metas)
            outs = self.pts_bbox_head(img_feats, img_metas)
        else:
            outs = self.simple_test_online(img_metas,img)

        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        query_feat = outs['query_feat']
        query_pos = outs['all_refine_pts'][-1]
        query_cls = outs['all_cls_scores'][-1]

        curr_query_feat = query_feat[:, ind_stamps_all == 0]
        curr_query_pos = query_pos[:, ind_stamps_all == 0].detach()
        curr_query_timestamp = query_pos.new_zeros(
            B, curr_query_feat.shape[1], self.num_refines, 1)
        curr_query_cls = query_cls[:, ind_stamps_all == 0]
        curr_query_cls_for_memory = curr_query_cls
        outputs = dict(cls_score=curr_query_cls,
                       refine_pts=curr_query_pos,
                       outs=outs)

        if self.query_memory_enabled:
            outputs['_raw_query_feat'] = curr_query_feat.clone()
            outputs['_raw_query_pos'] = curr_query_pos.clone()
            outputs['_raw_query_cls'] = curr_query_cls.clone()
            memory_context = self._query_memory_context(
                kwargs, img_metas, curr_query_feat.device,
                curr_query_feat.dtype)
        else:
            memory_context = None

        # Problem 1 (single-read) + Problem 2 (future-aware age):
        # the observation queries read history memory EXACTLY ONCE, before the
        # SCF recursion, at effective_age = base_age + 0. Already-fused active
        # queries are never re-read inside the loop.
        curr_query_feat = self._apply_query_memory_once(
            curr_query_feat, curr_query_pos, curr_query_cls_for_memory,
            img_metas, memory_context, future_offset=0.0)

        forecast_points_list = list()
        forecast_semantics_list = list()
        pred_trajs_list = list()
        forecast_points_mask_list = list()
        if self.training:
            num_fu_frames = max(1,min(self.curr_epoch - self.finetune_epoch+1, self.num_fu_frames))
        else:
            num_fu_frames = self.num_fu_frames

        for interval in range(num_fu_frames):
            # NOTE: the observation queries were already fused ONCE before this
            # loop and are NOT re-read here (Problem 1). Only the scheduled
            # future-query group added this step reads memory, and it does so
            # exactly once (below), with its own future_offset (Problem 2).
            fused_ego_feat,_ = self.ego_cross_attn(ego_feat.new_ones(B, 1, 3)*0.5, ego_feat, curr_query_pos.detach(),
                                                    curr_query_feat.detach(), )
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

            scheduled_mask = ind_stamps_all == interval + 1
            scheduled_feat = query_feat[:, scheduled_mask]
            scheduled_pos = query_pos[:, scheduled_mask].detach()
            scheduled_cls = query_cls[:, scheduled_mask]
            if scheduled_feat.shape[1] > 0:
                # scheduled future-query group `interval` reads memory ONCE, at
                # effective_age = base_age + (interval + 1) * frame_interval.
                scheduled_future_offset = (
                    interval + 1) * self.query_memory_frame_interval
                scheduled_feat = self._apply_query_memory_once(
                    scheduled_feat, scheduled_pos, scheduled_cls, img_metas,
                    memory_context, future_offset=scheduled_future_offset)
                curr_query_feat = torch.cat(
                    [curr_query_feat, scheduled_feat], dim=1)
                curr_query_pos = torch.cat(
                    [curr_query_pos, scheduled_pos], dim=1).detach()
                curr_query_cls_for_memory = torch.cat(
                    [curr_query_cls_for_memory, scheduled_cls], dim=1)
                curr_query_timestamp = torch.cat([
                    curr_query_timestamp,
                    curr_query_pos.new_ones(
                        B, scheduled_feat.shape[1], self.num_refines, 1) * 0.5
                ], dim=1)

            pos_embedding = self.position_encoder(torch.cat([curr_query_pos,curr_query_timestamp],dim=-1).flatten(2,3))
            curr_query_feat = curr_query_feat + fused_ego_feat + pos_embedding

            reg_offset = self.reg_branch(curr_query_feat).unflatten(-1, (-1, 3)) * 0.5
            cls_score = self.cls_branch(curr_query_feat).unflatten(-1, (-1, 17))
            curr_query_cls_for_memory = cls_score
            vel_offset = self.vel_branch(curr_query_feat).unflatten(-1, (-1, 2))
            #
            pred_labels = cls_score.argmax(-1)
            pred_moving_mask = torch.logical_and(pred_labels >= 2, pred_labels <= 10).unsqueeze(-1)
            reg_offset = torch.cat(
                [reg_offset[..., :2] + vel_offset * pred_moving_mask, reg_offset[..., 2:]], dim=-1)
            reg_offset = reg_offset.flatten(2, 3)

            # reg_offset = self.reg_branch(curr_query_feat) * 0.5
            curr_query_pos = self.refine_points(curr_query_pos, reg_offset)
            forecast_semantics_list.append(cls_score)
            forecast_points_list.append(curr_query_pos)
            if self.training:
                ego2lidar =torch.tensor(np.stack([meta['ego2lidar'] for meta in img_metas]) ,device=device,dtype=torch.float32)
                gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1, :]
                pred_traj_expand = torch.cat([-gt_traj, torch.zeros_like(pred_traj[:, :, :1])], dim=-1)

                gt_points = self.trans_points(curr_query_pos.flatten(1, 2), pred_traj_expand, ego2lidar).reshape(
                    curr_query_pos.shape)
                forecast_points_mask_list.append(gt_points[..., 0] >= 0)

        if not self.pretrain and len(pred_trajs_list)<self.num_fu_frames :
            fused_ego_feat,_ = self.ego_cross_attn(ego_feat.new_zeros(B, 1, 3), ego_feat, curr_query_pos,
                                                 curr_query_feat)
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

        outputs.update(
                        dict(forecast_semantics_list = forecast_semantics_list,
                       forecast_points_list = forecast_points_list,
                       pred_trajs_list = pred_trajs_list,
                       forecast_points_mask_list = forecast_points_mask_list))
        return outputs

    def simple_test(self,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        for key in kwargs.keys():
            kwargs[key] = kwargs[key][0]

        if self.query_memory_enabled and self.query_memory_source == 'online':
            self._prepare_online_query_memory(img_metas)

        outputs = self.forward_backbone(img, img_metas, **kwargs)
        cls_score, curr_query_pos, outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        if self.query_memory_enabled and self.query_memory_source == 'online':
            raw_feat = outputs.get('_raw_query_feat')
            raw_pos = outputs.get('_raw_query_pos')
            raw_cls = outputs.get('_raw_query_cls')
            if raw_feat is None or raw_pos is None or raw_cls is None:
                raw_cls = outs['all_cls_scores'][-1][:, self.pts_bbox_head.ind_stamps_all == 0]
                raw_pos = outs['all_refine_pts'][-1][:, self.pts_bbox_head.ind_stamps_all == 0]
                raw_feat = outs['query_feat'][:, self.pts_bbox_head.ind_stamps_all == 0]
            self._memory_write(raw_feat, raw_pos, raw_cls, img_metas)

        pred_dict = dict(cls_scores=outs['all_cls_scores'][-1][:,self.pts_bbox_head.ind_stamps_all==0], refine_pts=outs['all_refine_pts'][-1][:,self.pts_bbox_head.ind_stamps_all==0])
        occ_pred = self.pts_bbox_head.get_occ(pred_dict)[0]
        # self.pred_num += torch.bincount(occ_pred.flatten())
        geo_pred = torch.ones_like(occ_pred) * 17
        geo_pred[occ_pred != 17] = 0
        res_dict = {f'semantic_occ_0s': [occ_pred.cpu().numpy()],
                    f'geo_occ_0s': [geo_pred.cpu().numpy()]}

        forecast_points_list, forecast_semantics_list, pred_trajs_list = \
            outputs['forecast_points_list'],outputs['forecast_semantics_list'],outputs['pred_trajs_list']

        for interval in range(self.num_fu_frames):
            input_dict = dict(cls_scores=forecast_semantics_list[interval],
                              refine_pts=forecast_points_list[interval])
            occ_forecast = self.pts_bbox_head.get_occ(input_dict)[0]  # eval for single batch
            geo_forecast = torch.ones_like(occ_forecast) * 17
            geo_forecast[occ_forecast != 17] = 0
            # pred_traj_list.append(pred_traj)
            res_dict.update({
                f'semantic_occ_{int(interval + 1)}s': [occ_forecast.cpu().numpy()],
                f'geo_occ_{int(interval + 1)}s': [geo_forecast.cpu().numpy()],
            })

        res_dict['pred_traj'] = torch.cat(pred_trajs_list, 1)
        return res_dict

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):

        temporal_semantics = kwargs['temporal_semantics']
        B = img.shape[0]
        temporal2ego = kwargs['temporal2ego']
        outputs = self.forward_backbone(img,img_metas,**kwargs)
        cls_score,refine_pts,outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        losses = dict()
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        if self.pretrain:
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
        else:
            # outs_inits = dict(init_points = outs['init_points'],all_cls_scores = [], all_refine_pts = [])
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
            outs['init_points'] = None
            for i in range(len(outs['all_cls_scores'])):
                outs['all_cls_scores'][i] = outs['all_cls_scores'][i][:,ind_stamps_all==0]
                outs['all_refine_pts'][i] = outs['all_refine_pts'][i][:,ind_stamps_all==0]
            loss_inputs = [voxel_semantics,outs,]
            losses.update(self.pts_bbox_head.loss(*loss_inputs))

        forecast_points_list = outputs['forecast_points_list']
        forecast_semantics_list = outputs['forecast_semantics_list']
        pred_trajs_list = outputs['pred_trajs_list']
        forecast_points_mask_list = outputs['forecast_points_mask_list']

        voxel_semantics_temporal = [sem['voxel_semantics'] for sem in kwargs['temporal_semantics'].values()]

        num_fu_frames = len(forecast_semantics_list)
        losses.update(
            self.pts_bbox_head.loss_future(voxel_semantics_temporal[:num_fu_frames],
                                           forecast_points_list,forecast_semantics_list,forecast_points_mask_list))
        for interval,pred_traj in enumerate(pred_trajs_list):

            loss_traj = self.loss_traj(pred_traj.squeeze(1), kwargs['temporal_trajs'][:, interval, :], interval + 1)
            losses.update(loss_traj)

        return losses