"""One native invocation per process to isolate baseline sanitizer diagnostics."""

import argparse
import hashlib
import importlib.util
import json

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["original", "queued", "debug"], required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fixed_store", "/ws/c128_20261012/candidate/integration/kvtq_store.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rotation, qjl = reference._get_quant_matrices(torch.device("npu:0"))
    torch.manual_seed(20261014)
    vectors = torch.randn(1, 1, 128).to(torch.bfloat16).npu()
    if args.variant == "original":
        torch.ops.load_library("/ws/torch-extensions/turboquant_torch/turboquant_torch.so")
        output = torch.ops.turboquant.kv_cache_turbo_quant(vectors, rotation, qjl, 4)
    elif args.variant == "debug":
        torch.ops.load_library("/ws/quant_debug_20261014/torch-extensions/tq_quant_debug/tq_quant_debug.so")
        output = torch.ops.turboquant.kv_cache_turbo_quant_debug(vectors, rotation, qjl, 4)
    else:
        torch.ops.load_library("/ws/queued_codec_20261014/torch-extensions/tq_queued_quant/tq_queued_quant.so")
        output = torch.ops.turboquant.kv_cache_turbo_quant_queued(vectors, rotation, qjl, 4)
    print(json.dumps({"variant": args.variant, "output_sha256": [hashlib.sha256(x.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in output]}), flush=True)


if __name__ == "__main__":
    main()
