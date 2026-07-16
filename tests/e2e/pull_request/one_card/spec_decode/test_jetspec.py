from __future__ import annotations

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import JETSPEC, calculate_acceptance_per_pos

# JetSpec block_size=16 -> 1 anchor + 15 mask slots per draft step.
NUM_SPECULATIVE_TOKENS = 15

# Sanity floor for the mean acceptance length (1 + accepted / drafts). The
# non-causal DFlash-b16 head scores ~2.9 on this prompt set, so the floor must
# sit clearly above it: if the causal draft wiring regresses (the head runs
# non-causal again), acceptance falls back to the ~2.9 band and this must
# fail, not ride on noise. JetSpec reports roughly 2x DFlash acceptance for
# causal chain drafting. TODO(jetspec): replace with a calibrated
# per-position baseline (like BASELINES["dflash"]) once measured on NPU.
MIN_ACCEPTANCE_LENGTH = 3.5


@pytest.mark.parametrize("method", JETSPEC.keys())
def test_jetspec_acceptance(method: str):
    main_model_name = JETSPEC[method]["main"]
    spec_model_name = JETSPEC[method]["spec"]

    # The whole adaptation rests on the official checkpoint spelling its
    # causality switch as `dflash_config.causal_head` (translated to `causal`
    # by the NPU platform hook). If the checkpoint changes shape, fail loudly
    # here instead of silently running non-causal and riding on the
    # acceptance floor below.
    from transformers import PretrainedConfig

    spec_config_dict, _ = PretrainedConfig.get_config_dict(spec_model_name)
    spec_dflash_config = spec_config_dict.get("dflash_config") or {}
    assert spec_dflash_config.get("causal_head") or spec_dflash_config.get("causal"), (
        f"{spec_model_name} no longer declares dflash_config.causal_head/causal; "
        "the JetSpec causality translation would silently no-op."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [{"role": "user", "content": "Hello, your name is"}]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
    }

    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=[NUM_SPECULATIVE_TOKENS + 1, 2 * (NUM_SPECULATIVE_TOKENS + 1)],
    )

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, NUM_SPECULATIVE_TOKENS, Counter, Vector)
    acceptance_length = 1 + sum(acceptance_per_pos)
    print(f"acceptance_per_pos: {acceptance_per_pos}")
    print(f"acceptance_length: {acceptance_length}")

    assert acceptance_length >= MIN_ACCEPTANCE_LENGTH
