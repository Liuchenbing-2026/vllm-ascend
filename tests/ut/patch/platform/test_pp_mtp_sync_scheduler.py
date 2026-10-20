# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

import vllm_ascend.patch.platform.patch_pp_mtp as pp_mtp_patch


@pytest.mark.parametrize(
    "pp,v2,spec,async_mode,expected",
    [
        (True, True, True, False, True),
        (False, True, True, False, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
        (True, True, True, True, False),
    ],
)
def test_sync_pp_spec_fence_scope(pp, v2, spec, async_mode, expected):
    config = SimpleNamespace(
        use_v2_model_runner=v2,
        speculative_config=object() if spec else None,
        scheduler_config=SimpleNamespace(async_scheduling=async_mode),
    )
    assert pp_mtp_patch._use_sync_pp_spec_scheduler_fence(config, pp) is expected


def _scheduler(tmp_path, monkeypatch):
    # CPU metadata only: no model weights or model execution.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = AutoConfig.for_model(
        "qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        mtp_num_hidden_layers=1,
    )
    config.architectures = ["Qwen3_5ForCausalLM"]
    config.to_json_file(tmp_path / "config.json")
    model = ModelConfig(model=str(tmp_path), dtype="float16", max_model_len=128, seed=42, skip_tokenizer_init=True)
    parallel = ParallelConfig(pipeline_parallel_size=2)
    spec = SpeculativeConfig(
        method="mtp", num_speculative_tokens=1, target_model_config=model, target_parallel_config=parallel
    )
    cache = CacheConfig(block_size=16, enable_prefix_caching=False)
    cache.num_gpu_blocks = 128
    vllm_config = VllmConfig(
        model_config=model,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=parallel,
        cache_config=cache,
        speculative_config=spec,
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            max_model_len=128,
            enable_chunked_prefill=True,
            async_scheduling=False,
            is_encoder_decoder=model.is_encoder_decoder,
            watermark=0.0,
        ),
    )
    register_all_kvcache_specs(vllm_config)
    kv_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
    )
    assert scheduler.vllm_config.use_v2_model_runner
    assert scheduler.use_v2_model_runner
    request = Request("r", [10, 11, 12, 13, 14, 15], SamplingParams(max_tokens=16, ignore_eos=True), None)
    scheduler.add_request(request)
    return scheduler, request


def _output(request, tokens):
    return ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_sync_pp_mtp_waits_for_final_prefill_output(tmp_path, monkeypatch):
    scheduler, request = _scheduler(tmp_path, monkeypatch)
    try:
        first = scheduler.schedule()
        assert first.num_scheduled_tokens == {request.request_id: 4}
        final = scheduler.schedule()
        assert final.num_scheduled_tokens == {request.request_id: 2}
        assert request.num_output_tokens == 0

        # Draft metadata from the earlier in-flight prefill is supplied through
        # the real scheduler API, rather than injecting cached-data arguments.
        scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[-1]]))
        assert scheduler.schedule().num_scheduled_tokens == {}

        scheduler.update_from_output(first, _output(request, []))
        assert request.num_in_flight_tokens == 2
        assert scheduler.schedule().num_scheduled_tokens == {}

        scheduler.update_from_output(final, _output(request, [42]))
        assert request.num_in_flight_tokens == 0
        resumed = scheduler.schedule()
        assert resumed.num_scheduled_tokens == {request.request_id: 2}
        assert resumed.scheduled_cached_reqs.new_token_ids == [[42]]
        assert resumed.scheduled_spec_decode_tokens == {request.request_id: [-1]}
    finally:
        scheduler.shutdown()


def test_sync_pp_mtp_fence_keeps_other_prefills_schedulable(tmp_path, monkeypatch):
    scheduler, request = _scheduler(tmp_path, monkeypatch)
    try:
        scheduler.schedule()
        other = Request("other", [20, 21, 22, 23, 24], SamplingParams(max_tokens=16, ignore_eos=True), None)
        scheduler.add_request(other)
        final = scheduler.schedule()
        assert final.num_scheduled_tokens[request.request_id] == 2
        assert final.num_scheduled_tokens[other.request_id] > 0

        scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[-1]]))
        next_step = scheduler.schedule()
        assert request.request_id not in next_step.num_scheduled_tokens
        assert next_step.num_scheduled_tokens[other.request_id] > 0
    finally:
        scheduler.shutdown()
