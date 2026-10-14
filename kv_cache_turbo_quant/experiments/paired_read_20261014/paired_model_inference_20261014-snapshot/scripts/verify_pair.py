"""Independent full-state oracle for paired K/V packing and real quantization."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path("/ws/paired_model_inference_20261014")
    sys.path.insert(0, str(root / "candidate_write/integration"))
    from kvtq_write import PackedCacheWriter

    torch.ops.load_library("/ws/torch-extensions/turboquant_torch/turboquant_torch.so")
    torch.manual_seed(20261013)
    records = []
    writer = PackedCacheWriter()
    for heads in [1, 2, 4]:
        for tokens in [1, 16, 17, 63, 64, 129, 4096, 8000]:
            capacity = max(1024, tokens * 2)
            initial = [
                torch.randint(0, 256, (capacity, heads, 66), dtype=torch.uint8)
                for _ in range(2)
            ]
            idx = torch.randint(0, 256, (tokens, 2 * heads, 64), dtype=torch.uint8)
            norm_bytes = torch.randint(
                0, 256, (tokens, 2 * heads, 2), dtype=torch.uint8
            )
            norms = norm_bytes.view(torch.bfloat16).squeeze(-1)
            packed = torch.cat([idx, norm_bytes], dim=-1)
            for order in ["adjacent", "reverse", "random"]:
                slots = (
                    torch.arange(tokens)
                    if order == "adjacent"
                    else torch.arange(tokens - 1, -1, -1)
                    if order == "reverse"
                    else torch.randperm(capacity)[:tokens]
                )
                expected = [
                    initial[i]
                    .clone()
                    .index_copy_(0, slots, packed[:, i * heads : (i + 1) * heads])
                    for i in range(2)
                ]
                actual = [x.npu().view(torch.bfloat16) for x in initial]
                idx_npu, norms_npu, slots_npu = idx.npu(), norms.npu(), slots.npu()
                for _ in range(10):
                    writer.write_pair(idx_npu, norms_npu, slots_npu, *actual)
                torch.npu.synchronize()
                mismatches = [
                    (a.view(torch.uint8).cpu() != b).sum().item()
                    for a, b in zip(actual, expected)
                ]
                record = {
                    "heads": heads,
                    "tokens": tokens,
                    "order": order,
                    "mismatched_bytes": mismatches,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
                assert mismatches == [0, 0], record
    old = load("old_store", "/ws/c128_20261012/candidate/integration/kvtq_store.py")
    new = load("paired_store", root / "candidate_write/integration/kvtq_store.py")
    for heads in [1, 2, 4]:
        for tokens in [1, 17, 60, 128, 256, 4096]:
            key = torch.randn(tokens, heads, 128).to(torch.bfloat16).npu()
            value = (torch.randn(tokens, heads, 128) * 3).to(torch.bfloat16).npu()
            initial = [
                torch.randint(0, 256, (128, 128, heads, 66), dtype=torch.uint8)
                for _ in range(2)
            ]
            slots = torch.randperm(16384)[:tokens].npu()
            baseline, actual = [
                [x.npu().view(torch.bfloat16) for x in initial] for _ in range(2)
            ]
            old._write_cache(SimpleNamespace(), key, baseline[0], slots)
            old._write_cache(SimpleNamespace(), value, baseline[1], slots)
            new._write_cache_pair(
                SimpleNamespace(key_cache=actual[0], value_cache=actual[1]),
                key,
                value,
                slots,
            )
            torch.npu.synchronize()
            mismatch = [
                (a.view(torch.uint8) != b.view(torch.uint8)).sum().item()
                for a, b in zip(actual, baseline)
            ]
            record = {
                "kind": "paired_real_quantizer",
                "tokens": tokens,
                "heads": heads,
                "mismatched_bytes": mismatch,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            assert mismatch == [0, 0], record
    (root / "verify-pair.json").write_text(
        json.dumps(
            {
                "passed": len(records),
                "records": records,
                "scope": "90 cases, full K/V cache bytes including untouched slots; exact independent pack oracle and original two quantizer calls",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
