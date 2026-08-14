# Copyright (c) Phigent Robotics. All rights reserved.
import hashlib

import torch
from mmcv.runner import HOOKS, OptimizerHook


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@HOOKS.register_module()
class QueryMemoryConnectivityOptimizerHook(OptimizerHook):
    """Optimizer hook for guarded STAC-QM Memory-only smoke training.

    It performs the normal zero-grad/backward/clip/step sequence while checking
    that only query-memory parameters receive gradients, the restored TASS state
    never changes, expected memory reads actually occur, and the zero-initialized
    fusion path becomes connected after a short warm-up.
    """

    _GRAD_GROUPS = {
        'fusion_out': ('query_memory.fusion.out_proj.weight',),
        'fusion_gate': ('query_memory.fusion.gate_mlp.',),
        'attention_q': ('query_memory.attention.q_proj.',),
        'attention_k': ('query_memory.attention.k_proj.',),
        'attention_v': ('query_memory.attention.v_proj.',),
        'motion_last': ('query_memory.motion_compensator.mlp.2.',),
    }

    def __init__(self,
                 grad_clip=None,
                 connectivity_check_iter=100,
                 log_interval=20,
                 expected_memory_reads=7,
                 expected_fused_queries=1040):
        super().__init__(grad_clip=grad_clip)
        self.connectivity_check_iter = int(connectivity_check_iter)
        self.log_interval = int(log_interval)
        self.expected_memory_reads = int(expected_memory_reads)
        self.expected_fused_queries = int(expected_fused_queries)
        self._ever_nonzero = {key: False for key in self._GRAD_GROUPS}
        self._base_parameter_digest = None
        self._base_buffer_digest = None
        self._connectivity_checked = False

    def before_run(self, runner):
        model = _unwrap_model(runner.model)
        if not getattr(model, 'memory_finetune_mode', False):
            raise RuntimeError(
                'QueryMemoryConnectivityOptimizerHook requires '
                'memory_finetune_mode=True')
        model.validate_query_memory_training_setup(
            optimizer=runner.optimizer, logger=runner.logger)
        self._base_parameter_digest = _tensor_digest([
            (name, param) for name, param in model.named_parameters()
            if not name.startswith('query_memory.')
        ])
        self._base_buffer_digest = _tensor_digest([
            (name, buffer) for name, buffer in model.named_buffers()
            if not name.startswith('query_memory.')
        ])

    def _gradient_metrics(self, model):
        metrics = {key: 0.0 for key in self._GRAD_GROUPS}
        for name, param in model.named_parameters():
            grad = param.grad
            if not name.startswith('query_memory.'):
                if grad is not None and torch.count_nonzero(grad).item() != 0:
                    raise RuntimeError(
                        f'Frozen base parameter received gradient: {name}')
                continue
            if grad is None:
                continue
            grad_norm = float(grad.detach().float().norm().item())
            for key, prefixes in self._GRAD_GROUPS.items():
                if any(name == prefix or name.startswith(prefix)
                       for prefix in prefixes):
                    metrics[key] += grad_norm
        for key, value in metrics.items():
            if value > 0.0:
                self._ever_nonzero[key] = True
        return metrics

    def _memory_metrics(self, model):
        diagnostics = list(getattr(model, 'query_memory_diagnostics', []))
        metrics = dict(
            memory_read_count=float(len(diagnostics)),
            memory_fused_queries=float(sum(
                int(item.get('query_count', 0)) for item in diagnostics)),
            memory_empty_forward=float(not diagnostics))
        tensor_keys = {
            'has_candidate': 'memory_has_candidate_ratio',
            'candidate_count': 'memory_candidate_count',
            'avg_gate': 'memory_gate_mean',
            'residual_norm': 'memory_residual_norm',
            'effective_age': 'memory_effective_age',
            'motion_residual_mean': 'memory_motion_residual',
        }
        for source_key, metric_key in tensor_keys.items():
            values = []
            for item in diagnostics:
                value = item.get(source_key)
                if isinstance(value, torch.Tensor) and value.numel():
                    values.append(value.detach().float().reshape(-1))
                elif isinstance(value, (float, int)):
                    values.append(torch.tensor([float(value)]))
            if values:
                metrics[metric_key] = float(torch.cat(values).mean().item())
        valid_slots = getattr(model, '_last_query_memory_valid_slots', None)
        if valid_slots is not None:
            metrics['memory_valid_slots'] = float(valid_slots)
        return metrics

    def _check_connectivity(self, runner):
        missing = [
            key for key in (
                'fusion_out', 'fusion_gate', 'attention_q', 'attention_k',
                'attention_v')
            if not self._ever_nonzero[key]
        ]
        if missing:
            raise RuntimeError(
                'STAC-QM connectivity check found no nonzero gradient for: '
                f'{missing}')
        if not self._ever_nonzero['motion_last']:
            runner.logger.warning(
                'STAC-QM motion compensator final layer still has zero gradient '
                'at connectivity check; inspect motion candidates before the '
                'formal run.')
        self._connectivity_checked = True

    def after_train_iter(self, runner):
        model = _unwrap_model(runner.model)
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        model._assert_memory_finetune_temporal_state()
        gradient_metrics = self._gradient_metrics(model)
        memory_metrics = self._memory_metrics(model)
        if int(memory_metrics['memory_read_count']) != self.expected_memory_reads:
            raise RuntimeError(
                'Unexpected STAC-QM read count: '
                f'{int(memory_metrics["memory_read_count"])} != '
                f'{self.expected_memory_reads}')
        if int(memory_metrics['memory_fused_queries']) != \
                self.expected_fused_queries:
            raise RuntimeError(
                'Unexpected STAC-QM fused-query count: '
                f'{int(memory_metrics["memory_fused_queries"])} != '
                f'{self.expected_fused_queries}')

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                gradient_metrics['total'] = float(grad_norm)
        runner.optimizer.step()

        iteration = int(runner.iter) + 1
        if iteration % self.log_interval == 0 or iteration == 1:
            log_values = {
                f'qm_grad_{key}': value
                for key, value in gradient_metrics.items()
            }
            log_values.update(memory_metrics)
            runner.log_buffer.update(
                log_values, runner.outputs.get('num_samples', 1))
        if not self._connectivity_checked and \
                iteration >= self.connectivity_check_iter:
            self._check_connectivity(runner)

    def after_run(self, runner):
        model = _unwrap_model(runner.model)
        model._assert_memory_finetune_temporal_state()
        parameter_digest = _tensor_digest([
            (name, param) for name, param in model.named_parameters()
            if not name.startswith('query_memory.')
        ])
        buffer_digest = _tensor_digest([
            (name, buffer) for name, buffer in model.named_buffers()
            if not name.startswith('query_memory.')
        ])
        if parameter_digest != self._base_parameter_digest:
            raise RuntimeError('Frozen base parameters changed during smoke run')
        if buffer_digest != self._base_buffer_digest:
            raise RuntimeError(
                'Frozen base buffers/BN statistics changed during smoke run')
        if not self._connectivity_checked:
            self._check_connectivity(runner)
