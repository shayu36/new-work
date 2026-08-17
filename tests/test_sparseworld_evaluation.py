from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook

from mmdet3d.apis.sparseworld_test import (
    _interleave_result_parts, _parse_sparseworld_result)
from mmdet3d.apis.train import _select_detector_eval_hook
from mmdet3d.core.evaluation import (
    SparseWorldDistEvalHook, SparseWorldEvalHook)
from mmdet3d.core.evaluation.sparseworld_eval_hooks import (
    _flatten_eval_results)


def _temporal_result():
    return {
        'semantic_occ_0s': [np.full((2, 3), 1, dtype=np.int64)],
        'semantic_occ_2s': [np.full((2, 3), 2, dtype=np.int64)],
        'semantic_occ_4s': [np.full((2, 3), 3, dtype=np.int64)],
        'semantic_occ_6s': [np.full((2, 3), 4, dtype=np.int64)],
        'pred_traj': torch.ones(1, 6, 2),
    }


def test_sparseworld_dictionary_result_is_parsed_without_sequence_indexing():
    occ, pred_traj = _parse_sparseworld_result(_temporal_result())

    assert occ.shape == (4, 2, 3)
    assert occ.dtype == np.uint8
    assert occ[:, 0, 0].tolist() == [1, 2, 3, 4]
    assert pred_traj.device.type == 'cpu'
    assert pred_traj.shape == (1, 6, 2)


@pytest.mark.parametrize('invalid_result', [[], tuple(), np.zeros(1)])
def test_sparseworld_result_rejects_generic_sequence_outputs(invalid_result):
    with pytest.raises(TypeError, match='expects a dictionary'):
        _parse_sparseworld_result(invalid_result)


def test_distributed_result_interleave_trims_4219_sampler_padding():
    dataset_size = 4219
    rank_zero = list(range(0, dataset_size, 2))
    rank_one = list(range(1, dataset_size, 2)) + [0]

    ordered = _interleave_result_parts(
        [rank_zero, rank_one], size=dataset_size)

    assert ordered == list(range(dataset_size))


def test_temporal_metric_lists_are_flattened_for_tensorboard():
    flattened = _flatten_eval_results({
        'IoU': [25.68, 23.15, 22.27, 21.21],
        'mIoU': np.array([18.20, 14.96, 13.18, 11.53]),
        'classes': 17,
    })

    assert flattened['IoU_0s'] == pytest.approx(25.68)
    assert flattened['IoU_3s'] == pytest.approx(21.21)
    assert flattened['mIoU_future_mean'] == pytest.approx(
        (14.96 + 13.18 + 11.53) / 3)
    assert flattened['classes'] == 17
    assert all(not isinstance(value, (list, tuple, np.ndarray))
               for value in flattened.values())


def test_eval_hook_places_only_scalars_in_log_buffer():
    class Dataset:
        def evaluate(self, results, **kwargs):
            del results, kwargs
            return {
                'IoU': [25.68, 23.15, 22.27, 21.21],
                'mIoU': [18.20, 14.96, 13.18, 11.53],
                'classes': 17,
            }

    hook = SparseWorldEvalHook.__new__(SparseWorldEvalHook)
    hook.dataloader = SimpleNamespace(dataset=Dataset())
    hook.eval_kwargs = {}
    hook.save_best = None
    runner = SimpleNamespace(
        logger=None,
        log_buffer=SimpleNamespace(output={}, ready=False))

    assert hook.evaluate(runner, results=[]) is None
    assert runner.log_buffer.ready is True
    assert 'IoU' not in runner.log_buffer.output
    assert 'mIoU' not in runner.log_buffer.output
    assert runner.log_buffer.output['mIoU_future_mean'] == pytest.approx(
        (14.96 + 13.18 + 11.53) / 3)
    assert all(not isinstance(value, list)
               for value in runner.log_buffer.output.values())


def test_sparseworld_model_selects_dictionary_aware_eval_hooks():
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import (
        SparseWorld4DTraj)

    assert SparseWorld4DTraj.uses_sparseworld_eval_api is True

    class SparseWorldModel:
        uses_sparseworld_eval_api = True

    model = SparseWorldModel()
    assert _select_detector_eval_hook(model, distributed=False) is \
        SparseWorldEvalHook
    assert _select_detector_eval_hook(model, distributed=True) is \
        SparseWorldDistEvalHook


def test_standard_model_keeps_generic_mmdet_eval_hooks():
    model = object()
    assert _select_detector_eval_hook(model, distributed=False) is \
        MMDET_EvalHook
    assert _select_detector_eval_hook(model, distributed=True) is \
        MMDET_DistEvalHook
