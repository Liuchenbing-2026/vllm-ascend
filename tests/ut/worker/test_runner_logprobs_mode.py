# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.worker import model_runner_v1


@pytest.mark.parametrize("mode", ["raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"])
def test_runner_preserves_configured_logprobs_mode(mode):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), logprobs_mode=mode),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=2),
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
    )

    def parent_init(runner, vllm_config, device):
        runner.offload_config = None
        runner.max_num_reqs = 2
        runner.scheduler_config = vllm_config.scheduler_config

    class SamplerReached(Exception):
        pass

    # Exercise the real constructor through sampler creation; isolate device
    # buffers and stop before unrelated attention/backend initialization.
    with (
        patch.object(model_runner_v1.GPUModelRunner, "__init__", parent_init),
        patch.object(model_runner_v1, "_torch_cuda_wrapper", return_value=nullcontext()),
        patch.object(model_runner_v1, "set_offloader"),
        patch.object(model_runner_v1, "create_offloader"),
        patch.object(model_runner_v1, "check_gdn_layer", return_value=False),
        patch.object(model_runner_v1.NPUModelRunner, "_make_buffer"),
        patch.object(model_runner_v1, "AscendSampler", side_effect=SamplerReached) as sampler,
        pytest.raises(SamplerReached),
    ):
        model_runner_v1.NPUModelRunner(config, torch.device("cpu"))
    sampler.assert_called_once_with(logprobs_mode=mode)
