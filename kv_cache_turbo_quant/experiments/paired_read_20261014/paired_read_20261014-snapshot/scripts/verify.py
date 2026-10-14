"""Check both outputs, physical-page changes, LUT invalidation and asynchronous producers."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/ws/paired_read_20261014/code/integration")
    from kvtq_native_pair import NativePagedPairReader

    reference = load("reference", "/ws/c128_20261012/candidate/integration/kvtq_store.py")
    old = load("old", "/ws/native_combined_20261013/candidate_write/integration/kvtq_native_read.py").NativePagedReader()
    reader = NativePagedPairReader()
    groups = load("groups", "/ws/opt_20261012/verify/paged_read_torch.py").get_input_groups()
    torch.manual_seed(20261014)
    records, async_records, lookup_records, timings = [], [], [], []
    invocations = 0

    def call(key, value, table, lengths, lookup, metadata):
        nonlocal invocations
        if sum(lengths):
            invocations += 1
        return reader(key, value, table, lengths, lookup, metadata)

    def compare(actual, expected):
        torch.npu.synchronize()
        counts = [(a.view(torch.int16) != b.view(torch.int16)).sum().item() for a, b in zip(actual, expected)]
        assert counts == [0, 0], counts
        return counts

    def benchmark(key, value, table, lengths, lookup, metadata):
        if args.precision_only or not sum(lengths):
            return
        functions = [("native_old", lambda: (old(key, table, lengths, lookup, metadata), old(value, table, lengths, lookup, metadata))),
                     ("native", lambda: reader(key, value, table, lengths, lookup, metadata))]
        for name, function in functions:
            for _ in range(5):
                function()
            torch.npu.synchronize()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                function()
            end.record()
            end.synchronize()
            record = {"tokens": sum(lengths), "reader": name, "device_event_mean_us": start.elapsed_time(end) * 50,
                      "iterations": 20, "scope": "two K/V outputs; event interval includes host submission gaps"}
            timings.append(record)
            print(json.dumps(record), flush=True)

    def expected_pair(key, value, table, lengths):
        return tuple(reference._dequant_dense_reference(None, c.view(torch.bfloat16), table, lengths) for c in [key, value])

    for number, group in enumerate(groups):
        key, table, _, _, total, _, lengths = group
        value = key.clone()
        value[..., :64].bitwise_xor_(0x3C)
        lookup = reference._get_byte_lut(key.device)
        metadata = SimpleNamespace()
        count = compare(call(key, value, table, lengths, lookup, metadata), expected_pair(key, value, table, lengths))
        records.append({"case": number, "tokens": total, "mismatched_bf16_bits": count})
        if total:
            original_table = table.clone()
            table[:, 0] = (table[:, 0] + 1) % key.shape[0]
            count = compare(call(key, value, table, lengths, lookup, metadata), expected_pair(key, value, table, lengths))
            records.append({"case": number, "variant": "changed_physical_pages", "mismatched_bf16_bits": count})
            table.copy_(original_table)
            benchmark(key, value, table, lengths, lookup, metadata)
    for batch in [60, 128]:
        length, heads, block = 2000, 2, 128
        pages = (length + block - 1) // block
        key = torch.randint(0, 256, (batch * pages, block, heads, 66), dtype=torch.uint8)
        key[..., 64:] = (torch.rand(batch * pages, block, heads, 1) + 0.01).to(torch.bfloat16).view(torch.uint8)
        key = key.npu()
        value = key.clone(); value[..., :64].bitwise_xor_(0x3C)
        table = torch.randperm(batch * pages).view(batch, pages).to(torch.int32).npu()
        lengths, metadata = [length] * batch, SimpleNamespace()
        lookup = reference._get_byte_lut(key.device)
        count = compare(call(key, value, table, lengths, lookup, metadata), expected_pair(key, value, table, lengths))
        records.append({"variant": f"long_c{batch}", "mismatched_bf16_bits": count})
        benchmark(key, value, table, lengths, lookup, metadata)
    changed = lookup.clone()
    for mutation in [False, True]:
        if mutation:
            changed.mul_(2)
        expected = (old(key, table, lengths, changed, metadata), old(value, table, lengths, changed, metadata))
        count = compare(call(key, value, table, lengths, changed, metadata), expected)
        lookup_records.append({"inplace_mutation": mutation, "mismatched_bf16_bits": count})
    for lengths, heads in [([20] * 16, 2), ([0, 33, 0, 257], 3)]:
        block, batch = 128, len(lengths)
        pages = max(1, (max(lengths) + block - 1) // block)
        key = torch.randint(0, 256, (batch * pages, block, heads, 66), dtype=torch.uint8)
        key[..., 64:] = (torch.rand(batch * pages, block, heads, 1) + 0.01).to(torch.bfloat16).view(torch.uint8)
        key = key.npu(); value = key.clone(); value[..., :64].bitwise_xor_(0x3C)
        table = torch.randperm(batch * pages).view(batch, pages).to(torch.int32).npu()
        lookup = reference._get_byte_lut(key.device)
        expected = expected_pair(key, value, table, lengths)
        torch.npu.synchronize()
        for nondefault in [False, True]:
            stream = torch.npu.Stream() if nondefault else torch.npu.current_stream()
            for repeat in range(5):
                with torch.npu.stream(stream):
                    k, v = torch.empty_like(key), torch.empty_like(value)
                    k.zero_(); k.copy_(key); v.zero_(); v.copy_(value)
                    actual = call(k.clone(), v.clone(), table.clone(), lengths, lookup, SimpleNamespace())
                    actual = tuple(x.clone() for x in actual)
                count = compare(actual, expected)
                async_records.append({"nondefault_stream": nondefault, "repeat": repeat, "mismatched_bf16_bits": count})
    assert len(records) == 15 and len(async_records) == 20 and len(lookup_records) == 2
    assert invocations == 36, invocations
    result = {"passed": len(records), "async_passed": len(async_records), "lookup_records": lookup_records,
              "records": records, "async_records": async_records, "timings": timings,
              "expected_filtered_kernel_invocations": invocations, "scope": "both K/V outputs exactly equal original BF16 decode; no semantic-model or throughput acceptance"}
    (args.output_dir / "verify-native-read.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
