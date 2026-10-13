"""Compare all four quantizer outputs against the original native operator."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path("/ws/queued_codec_20261014")
    sys.path.insert(0, str(root / "code/integration"))
    from kvtq_queued_quant import QueuedQuantizer

    torch.ops.load_library("/ws/torch-extensions/turboquant_torch/turboquant_torch.so")
    spec = importlib.util.spec_from_file_location(
        "fixed_store", "/ws/c128_20261012/candidate/integration/kvtq_store.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rotation, qjl = reference._get_quant_matrices(torch.device("npu:0"))
    quantizer = QueuedQuantizer()
    torch.manual_seed(20261014)
    records = []
    async_records = []

    def compare(actual, expected):
        counts = [
            (
                x.cpu().contiguous().view(torch.uint8)
                != y.cpu().contiguous().view(torch.uint8)
            )
            .sum()
            .item()
            for x, y in zip(actual, expected)
        ]
        assert counts == [0, 0, 0, 0], counts
        return counts

    for heads in [1, 2, 4]:
        for tokens in [1, 17, 60, 128, 4096]:
            vectors = (
                (torch.randn(tokens, heads, 128) * (heads + 0.5))
                .to(torch.bfloat16)
                .npu()
            )
            expected = torch.ops.turboquant.kv_cache_turbo_quant(
                vectors, rotation, qjl, 4
            )
            actual = quantizer(vectors, rotation, qjl, 4)
            row = {
                "heads": heads,
                "tokens": tokens,
                "mismatched_output_bytes": compare(actual, expected),
            }
            records.append(row)
            print(json.dumps(row), flush=True)
    for nondefault in [False, True]:
        for tokens in [17, 60]:
            stream = torch.npu.Stream() if nondefault else torch.npu.current_stream()
            with torch.npu.stream(stream):
                payload = (torch.randn(tokens, 2, 128) * 3).to(torch.bfloat16).npu()
                expected = tuple(
                    x.cpu()
                    for x in torch.ops.turboquant.kv_cache_turbo_quant(
                        payload, rotation, qjl, 4
                    )
                )
                vectors = torch.empty_like(payload)
                for repeat in range(5):
                    vectors.zero_()
                    vectors.copy_(payload)
                    # Temporary input clones are immediately released by Python;
                    # the wrapper must preserve their storage until submission.
                    actual = quantizer(
                        vectors.clone(), rotation.clone(), qjl.clone(), 4
                    )
                    consumer = tuple(x.clone() for x in actual)
                    row = {
                        "nondefault_stream": nondefault,
                        "tokens": tokens,
                        "repeat": repeat,
                        "mismatched_output_bytes": compare(consumer, expected),
                    }
                    async_records.append(row)
                    print(json.dumps(row), flush=True)
    assert len(records) == 15 and len(async_records) == 20
    (args.output_dir / "verify-queued-quant.json").write_text(
        json.dumps(
            {
                "passed": len(records),
                "async_passed": len(async_records),
                "records": records,
                "async_records": async_records,
                "candidate_quantizer_invocations": 35,
                "original_reference_invocations": 19,
                "expected_filtered_kernel_invocations": 54,
                "scope": "Exact idx/QJL/norm/gamma bytes; BF16 vectors, FP32 matrices, bits4; original device kernel unchanged",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
