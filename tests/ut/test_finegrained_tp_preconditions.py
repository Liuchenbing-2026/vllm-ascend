# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields
from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.ascend_config import FinegrainedTPConfig


@pytest.fixture
def vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=8,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=512,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=256),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        model_config=SimpleNamespace(is_moe=True),
    )


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("field_name", [field.name for field in fields(FinegrainedTPConfig)])
def test_enabled_finegrained_tp_rejects_pipeline_parallelism(vllm_config, field_name, size):
    config = FinegrainedTPConfig(**{field_name: size})
    vllm_config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(AssertionError, match="not supported with pipeline_parallel_size > 1"):
        config._validate_preconditions(vllm_config)


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 4])
def test_disabled_finegrained_tp_allows_pipeline_parallelism(vllm_config, pipeline_parallel_size):
    vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    FinegrainedTPConfig()._validate_preconditions(vllm_config)


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("field_name", [field.name for field in fields(FinegrainedTPConfig)])
def test_enabled_finegrained_tp_allows_single_pipeline_stage(vllm_config, field_name, size):
    config = FinegrainedTPConfig(**{field_name: size})
    config._validate_preconditions(vllm_config)
    assert getattr(config, field_name) == size


def test_capture_bound_disables_finegrained_tp_before_remaining_checks(vllm_config):
    config = FinegrainedTPConfig(oproj_tensor_parallel_size=2, mlp_tensor_parallel_size=2)
    vllm_config.compilation_config.max_cudagraph_capture_size = 16
    vllm_config.model_config.is_moe = False
    config._validate_preconditions(vllm_config)
    assert config.oproj_tensor_parallel_size == 0
    assert config.mlp_tensor_parallel_size == 0
