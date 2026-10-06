"""Offline vLLM run with KvCacheTurboQuant shadow integration."""
import argparse
import os
import sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")
os.environ.setdefault("VLLM_ASCEND_KVTQ", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "torch_ext"))
sys.path.insert(0, HERE)

import torch  # noqa: F401, E402
import torch_npu  # noqa: F401, E402

from build_ext import build  # noqa: E402
build()

import kvtq_shadow  # noqa: E402
kvtq_shadow.install()

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/models/Qwen3-1.7B")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp,
              enforce_eager=args.enforce_eager, gpu_memory_utilization=0.85)
    prompts = [
        "介绍一下华为昇腾NPU的主要特点。",
        "The capital of France is",
        "Write a short poem about autumn leaves.",
    ]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    print("=" * 60)
    for out in outs:
        print(f"PROMPT: {out.prompt!r}")
        print(f"OUTPUT: {out.outputs[0].text!r}")
        print("-" * 60)
    kvtq_shadow.report()


if __name__ == "__main__":
    main()