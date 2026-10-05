from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitBiasLogitsProcessor, MinTokensLogitsProcessor
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.sample.sampler import AscendSampler


def metadata(**overrides):
    values = dict(
        all_greedy=True,
        all_random=False,
        no_penalties=True,
        max_num_logprobs=None,
        logprob_token_ids=None,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        thinking_budget_state_holder=None,
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_greedy_exact_tokens(dtype):
    x = torch.tensor(
        [[1, 3, 3], [-float("inf")] * 3, [float("inf"), 1, float("inf")], [0, float("nan"), float("nan")]], dtype=dtype
    )
    sampler = AscendSampler()
    with patch(
        "vllm_ascend.sample.sampler.get_ascend_config", return_value=SimpleNamespace(enable_native_greedy_sampling=True)
    ):
        out = sampler(x, metadata())
    assert torch.equal(out.sampled_token_ids, x.float().argmax(-1).int().unsqueeze(-1))
    assert out.logprobs_tensors is None
    assert out.sampled_token_ids.dtype == torch.int32


@pytest.mark.parametrize(
    "override",
    [
        {"all_greedy": False},
        {"all_random": True},
        {"no_penalties": False},
        {"max_num_logprobs": 0},
        {"logprob_token_ids": {0: [1]}},
        {"allowed_token_ids_mask": torch.ones(1, 3, dtype=torch.bool)},
        {"bad_words_token_ids": {0: [[1]]}},
        {"thinking_budget_state_holder": object()},
        {"logitsprocs": SimpleNamespace(non_argmax_invariant=[object()])},
    ],
)
def test_processing_falls_back(override):
    sampler = AscendSampler()
    x, meta = torch.zeros(1, 3, dtype=torch.bfloat16), metadata(**override)
    with (
        patch(
            "vllm_ascend.sample.sampler.get_ascend_config",
            return_value=SimpleNamespace(enable_native_greedy_sampling=True),
        ),
        patch.object(Sampler, "forward", return_value="fallback") as parent,
    ):
        assert sampler(x, meta) == "fallback"
        parent.assert_called_once_with(x, meta, False, None)


@pytest.mark.parametrize(
    "enabled,dtype,bonus",
    [
        (False, torch.bfloat16, False),
        (True, torch.float32, False),
        (True, torch.bfloat16, True),
    ],
)
def test_disabled_dtype_bonus_fallback(enabled, dtype, bonus):
    sampler = AscendSampler()
    with (
        patch(
            "vllm_ascend.sample.sampler.get_ascend_config",
            return_value=SimpleNamespace(enable_native_greedy_sampling=enabled),
        ),
        patch.object(Sampler, "forward", return_value="fallback") as parent,
    ):
        assert sampler(torch.zeros(1, 3, dtype=dtype), metadata(), bonus, "raw_logits") == "fallback"
        assert parent.call_args.args[2:] == (bonus, "raw_logits")


@pytest.mark.parametrize("cls,field", [(LogitBiasLogitsProcessor, "biases"), (MinTokensLogitsProcessor, "min_toks")])
def test_inactive_builtin_and_reactivation(cls, field):
    processor = object.__new__(cls)
    setattr(processor, field, {})
    meta = metadata(logitsprocs=SimpleNamespace(non_argmax_invariant=[processor]))
    assert AscendSampler._has_no_active_greedy_processors(meta)
    setattr(processor, field, {0: object()})
    assert not AscendSampler._has_no_active_greedy_processors(meta)
    setattr(processor, field, {})
    assert AscendSampler._has_no_active_greedy_processors(meta)


def test_subclass_falls_back():
    class CustomBias(LogitBiasLogitsProcessor):
        pass

    processor = object.__new__(CustomBias)
    processor.biases = {}
    assert not AscendSampler._has_no_active_greedy_processors(
        metadata(logitsprocs=SimpleNamespace(non_argmax_invariant=[processor]))
    )
