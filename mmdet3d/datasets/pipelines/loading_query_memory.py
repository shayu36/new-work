import os
from pathlib import Path

import torch

try:
    from ..builder import PIPELINES
except Exception:
    class _FallbackRegistry:
        def register_module(self):
            def _decorator(cls):
                return cls
            return _decorator
    PIPELINES = _FallbackRegistry()


def _first_scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _first_scalar(value[0], default)
    return float(value)


def _scene_id(results):
    return results.get(
        'scene_token', results.get('scene_id', results.get('scene_name', None)))


def _sample_idx(results):
    return results.get(
        'sample_idx', results.get('sample_token', results.get('token', None)))


def _frame_idx(results):
    frame_idx = results.get('frame_idx', None)
    if frame_idx is None and 'curr' in results:
        frame_idx = results['curr'].get('frame_idx', None)
    return None if frame_idx is None else int(frame_idx)


def _timestamp(results):
    timestamp = results.get('timestamp', None)
    if timestamp is None:
        timestamp = results.get('img_timestamp', None)
    if timestamp is None and 'curr' in results:
        curr = results['curr']
        if 'timestamp' in curr:
            timestamp = float(curr['timestamp']) / 1e6
    return _first_scalar(timestamp, None)


@PIPELINES.register_module()
class LoadQueryMemoryFromFiles(object):
    """Load causal same-scene observation query cache for STAC-QM."""

    def __init__(self,
                 cache_root=None,
                 history_frames=3,
                 max_queries_per_frame=256,
                 strict=False,
                 embed_dims=256,
                 num_points=48,
                 file_suffix='.pt'):
        self.cache_root = cache_root
        self.history_frames = int(history_frames)
        self.max_queries_per_frame = int(max_queries_per_frame)
        self.strict = bool(strict)
        self.embed_dims = int(embed_dims)
        self.num_points = int(num_points)
        self.file_suffix = file_suffix

    def _candidate_paths(self, root, hist):
        sample = str(hist.get('sample_idx', hist.get('sample_token')))
        scene_values = []
        for key in ('scene_id', 'scene_token', 'scene_name'):
            value = hist.get(key, None)
            if value is not None and str(value) not in scene_values:
                scene_values.append(str(value))
        paths = []
        for scene in scene_values:
            paths.append(Path(root) / scene / f'{sample}{self.file_suffix}')
        paths.append(Path(root) / f'{sample}{self.file_suffix}')
        return paths

    def _find_cache_path(self, root, hist):
        paths = self._candidate_paths(root, hist)
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    def _validate_cache(self, cache, path):
        required = [
            'schema_version', 'sample_idx', 'scene_id', 'frame_idx',
            'timestamp', 'ego2global', 'query_feat', 'query_points_metric',
            'query_conf', 'valid_mask'
        ]
        missing = [key for key in required if key not in cache]
        if missing:
            raise KeyError(f'{path} missing query memory keys: {missing}')
        if int(cache['schema_version']) != 1:
            raise ValueError(
                f'{path} has unsupported schema_version '
                f'{cache["schema_version"]}')
        query_feat = cache['query_feat']
        query_points = cache['query_points_metric']
        query_conf = cache['query_conf']
        valid_mask = cache['valid_mask']
        ego2global = cache['ego2global']
        if query_feat.dim() != 2:
            raise ValueError(f'{path} query_feat must be [M, C]')
        if query_points.dim() != 3 or query_points.shape[-1] != 3:
            raise ValueError(f'{path} query_points_metric must be [M, R, 3]')
        if query_conf.dim() != 1 or valid_mask.dim() != 1:
            raise ValueError(f'{path} query_conf and valid_mask must be [M]')
        if query_feat.shape[0] != query_points.shape[0]:
            raise ValueError(f'{path} query_feat/query_points M mismatch')
        if query_feat.shape[0] != query_conf.shape[0]:
            raise ValueError(f'{path} query_feat/query_conf M mismatch')
        if query_feat.shape[0] != valid_mask.shape[0]:
            raise ValueError(f'{path} query_feat/valid_mask M mismatch')
        if tuple(ego2global.shape) != (4, 4):
            raise ValueError(f'{path} ego2global must be [4, 4]')

    def _select_top_queries(self, cache):
        feat = cache['query_feat'].detach().cpu()
        points = cache['query_points_metric'].detach().cpu()
        conf = cache['query_conf'].detach().cpu().float()
        valid = cache['valid_mask'].detach().cpu().bool()
        valid = valid & torch.isfinite(conf) & (conf > 0)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if indices.numel() > self.max_queries_per_frame:
            _, order = torch.topk(conf[indices], self.max_queries_per_frame)
            indices = indices[order]
        return feat[indices], points[indices], conf[indices], valid[indices]

    def _history_candidates(self, results):
        current_scene = _scene_id(results)
        current_sample = _sample_idx(results)
        current_frame = _frame_idx(results)
        current_ts = _timestamp(results)
        histories = results.get('query_memory_history_infos', [])
        filtered = []
        for hist in histories:
            hist_scene = hist.get(
                'scene_id', hist.get('scene_token', hist.get('scene_name')))
            if current_scene is not None and hist_scene != current_scene:
                continue
            hist_sample = hist.get('sample_idx', hist.get('sample_token'))
            if current_sample is not None and hist_sample == current_sample:
                continue
            hist_frame = hist.get('frame_idx', None)
            if current_frame is not None and hist_frame is not None:
                if int(hist_frame) >= current_frame:
                    continue
            hist_ts = hist.get('timestamp', None)
            if current_ts is not None and hist_ts is not None:
                if float(hist_ts) >= current_ts:
                    continue
            filtered.append(hist)
        return filtered[-self.history_frames:]

    def _empty_outputs(self, embed_dims=None, num_points=None):
        embed_dims = self.embed_dims if embed_dims is None else int(embed_dims)
        num_points = self.num_points if num_points is None else int(num_points)
        K = self.history_frames
        M = self.max_queries_per_frame
        return dict(
            memory_query_feat=torch.zeros(K, M, embed_dims),
            memory_points_metric=torch.zeros(K, M, num_points, 3),
            memory_conf=torch.zeros(K, M),
            memory_valid=torch.zeros(K, M, dtype=torch.bool),
            memory_source_ego2global=torch.eye(4).repeat(K, 1, 1),
            memory_age=torch.zeros(K, M))

    def __call__(self, results):
        root = self.cache_root or results.get('query_memory_cache_root', None)
        histories = self._history_candidates(results)
        current_ts = _timestamp(results)
        loaded = []
        if root is None:
            if self.strict and histories:
                raise FileNotFoundError(
                    'query memory cache_root is required in strict mode')
        else:
            for hist in histories:
                path = self._find_cache_path(root, hist)
                if not path.exists():
                    if self.strict:
                        raise FileNotFoundError(
                            'missing query memory cache for sample '
                            f'{hist.get("sample_idx")}: {path}')
                    loaded.append((hist, path, None))
                    continue
                cache = torch.load(path, map_location='cpu')
                self._validate_cache(cache, path)
                loaded.append((hist, path, cache))

        first_cache = next((cache for _, _, cache in loaded if cache is not None), None)
        embed_dims = self.embed_dims
        num_points = self.num_points
        if first_cache is not None:
            embed_dims = int(first_cache['query_feat'].shape[-1])
            num_points = int(first_cache['query_points_metric'].shape[-2])
        memory = self._empty_outputs(embed_dims, num_points)
        offset = self.history_frames - len(loaded)
        for out_index, (_, _, cache) in enumerate(loaded, start=offset):
            if cache is None:
                continue
            feat, points, conf, valid = self._select_top_queries(cache)
            n = min(feat.shape[0], self.max_queries_per_frame)
            age = float(current_ts) - float(cache['timestamp'])
            memory['memory_query_feat'][out_index, :n] = feat[:n]
            memory['memory_points_metric'][out_index, :n] = points[:n]
            memory['memory_conf'][out_index, :n] = conf[:n]
            memory['memory_valid'][out_index, :n] = valid[:n] & (age > 0)
            memory['memory_source_ego2global'][out_index] = \
                cache['ego2global'].detach().cpu().float()
            memory['memory_age'][out_index, :n] = age
        results.update(memory)
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(cache_root={self.cache_root}, '
                f'history_frames={self.history_frames}, '
                f'max_queries_per_frame={self.max_queries_per_frame}, '
                f'strict={self.strict})')
