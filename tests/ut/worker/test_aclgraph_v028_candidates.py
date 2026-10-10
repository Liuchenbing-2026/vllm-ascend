# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed batches must retain PIECEWISE candidates beside padded FULL graphs."""

from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


def make_manager(mode, lora_cases=(0,)):
    manager = ModelAclGraphManager.__new__(ModelAclGraphManager)
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[72, 80, 88, 96], max_cudagraph_capture_size=96
    )
    manager.max_num_reqs = 100
    manager.decode_query_len = 9
    manager.varlen_decode = False
    manager.cudagraph_mode = mode
    manager.lora_capture_cases = list(lora_cases)
    manager._candidates = {}
    manager._capture_descs = {}
    manager._graphs_captured = True
    manager._lora_dispatch_map, manager._max_lora_case = manager._build_lora_dispatch_map()
    manager._init_candidates()
    return manager


@pytest.mark.parametrize("tokens,padded", [(73, 80), (81, 88), (89, 96)])
@pytest.mark.parametrize("loras", [0, 2])
def test_mixed_batch_falls_back_to_piecewise(tokens, padded, loras):
    manager = make_manager(CUDAGraphMode.FULL_AND_PIECEWISE, (0, 2))
    desc = manager.dispatch(1, tokens, None, loras, max_query_len=tokens)
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.num_tokens == padded
    assert desc.num_active_loras == loras


def test_uniform_decode_keeps_full_priority():
    manager = make_manager(CUDAGraphMode.FULL_AND_PIECEWISE)
    desc = manager.dispatch(9, 81, 9, 0, max_query_len=9)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens == 81


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE])
def test_without_piecewise_does_not_invent_capture(mode):
    manager = make_manager(mode)
    assert manager.dispatch(1, 81, None, 0, max_query_len=81).cg_mode == CUDAGraphMode.NONE


def test_above_capture_limit_remains_eager():
    manager = make_manager(CUDAGraphMode.FULL_AND_PIECEWISE)
    assert manager.dispatch(1, 97, None, 0, max_query_len=97).cg_mode == CUDAGraphMode.NONE


def test_before_capture_does_not_replay():
    manager = make_manager(CUDAGraphMode.FULL_AND_PIECEWISE)
    manager._graphs_captured = False
    assert manager.dispatch(1, 81, None, 0, max_query_len=81).cg_mode == CUDAGraphMode.NONE
