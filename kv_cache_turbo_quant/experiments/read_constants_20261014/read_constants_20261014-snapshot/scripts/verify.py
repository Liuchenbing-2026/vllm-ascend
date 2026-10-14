"""Strict read equality and isolated latency, outside formal model cohorts."""

import argparse
import importlib.util
import json
import sys
from functools import partial
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/ws/read_constants_20261014")
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    root = Path("/ws/read_constants_20261014")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "code/integration"))
    sys.path.insert(0, "/ws/opt_20261012/verify")
    sys.path.append("/ws/c128_20261012/candidate/integration")
    from kvtq_native_read import NativePagedReader

    reference = load(
        "fixed_store", "/ws/c128_20261012/candidate/integration/kvtq_store.py"
    )
    groups = load(
        "input_groups", "/ws/opt_20261012/verify/paged_read_torch.py"
    ).get_input_groups()
    reader = NativePagedReader()
    old_reader = load("fixed_native_reader", "/ws/native_combined_20261013/candidate_write/integration/kvtq_native_read.py").NativePagedReader()
    records = []
    timings = []
    for number, group in enumerate(groups):
        cache, table, _cumulative, _centroid, total, _maximum, lengths = group
        metadata = SimpleNamespace()
        expected = reference._dequant_dense_reference(
            None, cache.view(torch.bfloat16), table, lengths
        )
        lookup = reference._get_byte_lut(cache.device)
        actual = reader(cache, table, lengths, lookup, metadata)
        torch.npu.synchronize()
        mismatch = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        record = {
            "case": number,
            "total_tokens": total,
            "lengths": lengths,
            "shape": list(actual.shape),
            "mismatched_bf16_bits": mismatch,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        assert mismatch == 0, record
        if total:
            old_table = table.clone()
            table[:, 0] = (table[:, 0] + 1) % cache.shape[0]
            expected = reference._dequant_dense_reference(
                None, cache.view(torch.bfloat16), table, lengths
            )
            actual = reader(cache, table, lengths, lookup, metadata)
            torch.npu.synchronize()
            mismatch = (
                (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
            )
            record = {
                "case": number,
                "variant": "same_lengths_changed_physical_pages",
                "mismatched_bf16_bits": mismatch,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            assert mismatch == 0, record
            table.copy_(old_table)
            for name, function in (
                []
                if args.precision_only
                else [
                    (
                        "native_old",
                        partial(
                            old_reader, cache, table, lengths, lookup, metadata,
                        ),
                    ),
                    (
                        "native",
                        partial(reader, cache, table, lengths, lookup, metadata),
                    ),
                ]
            ):
                for _ in range(5):
                    function()
                torch.npu.synchronize()
                start, end = (
                    torch.npu.Event(enable_timing=True),
                    torch.npu.Event(enable_timing=True),
                )
                start.record()
                for _ in range(20):
                    function()
                end.record()
                end.synchronize()
                row = {
                    "case": number,
                    "tokens": total,
                    "reader": name,
                    "device_event_mean_us": start.elapsed_time(end) * 50,
                    "iterations": 20,
                }
                timings.append(row)
                print(json.dumps(row), flush=True)
    # Additional scale points use the same original dequant reference.
    for batch in [60, 128]:
        heads, block, length = 2, 128, 2000
        pages = (length + block - 1) // block
        cache = torch.randint(
            0, 256, (batch * pages, block, heads, 66), dtype=torch.uint8
        )
        norms = (torch.rand(batch * pages, block, heads, 1) + 0.01).to(torch.bfloat16)
        cache[..., 64:] = norms.view(torch.uint8)
        cache = cache.npu()
        table = torch.randperm(batch * pages).view(batch, pages).to(torch.int32).npu()
        lengths = [length] * batch
        metadata = SimpleNamespace()
        lookup = reference._get_byte_lut(cache.device)
        expected = reference._dequant_dense_reference(
            None, cache.view(torch.bfloat16), table, lengths
        )
        actual = reader(cache, table, lengths, lookup, metadata)
        torch.npu.synchronize()
        mismatch = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        record = {
            "variant": f"long_c{batch}",
            "total_tokens": batch * length,
            "mismatched_bf16_bits": mismatch,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        assert mismatch == 0, record
        for name, function in (
            []
            if args.precision_only
            else [
                (
                    "native_old",
                    partial(
                        old_reader,
                        cache,
                        table,
                        lengths,
                        lookup,
                        metadata,
                    ),
                ),
                ("native", partial(reader, cache, table, lengths, lookup, metadata)),
            ]
        ):
            for _ in range(5):
                function()
            torch.npu.synchronize()
            start, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                function()
            end.record()
            end.synchronize()
            row = {
                "tokens": batch * length,
                "reader": name,
                "device_event_mean_us": start.elapsed_time(end) * 50,
                "iterations": 20,
            }
            timings.append(row)
            print(json.dumps(row), flush=True)
    assert len(records) == 15
    lookup_records = []
    changed = lookup.clone()
    for mutation in [False, True]:
        if mutation:
            changed.mul_(2)
        expected = old_reader(cache, table, lengths, changed, metadata)
        actual = reader(cache, table, lengths, changed, metadata)
        torch.npu.synchronize()
        mismatch = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        assert mismatch == 0, mismatch
        lookup_records.append({"inplace_mutation": mutation, "mismatched_bf16_bits": mismatch})
    order_records = []
    for lengths, heads in [([20] * 16, 2), ([0, 33, 0, 257], 3)]:
        import math
        block = 128
        pages = math.ceil(max(lengths) / block)
        cpu_cache = torch.randint(0, 256, (pages * len(lengths) + 2, block, heads, 66), dtype=torch.uint8)
        cpu_norm = (torch.rand(cpu_cache.shape[:-1]) + 0.01).to(torch.bfloat16)
        cpu_cache[..., 64:] = cpu_norm.unsqueeze(-1).view(torch.uint8)
        cpu_table = torch.arange(pages * len(lengths), dtype=torch.int32).view(len(lengths), pages)
        expected_cpu = reference._dequant_dense_reference(None, cpu_cache.view(torch.bfloat16), cpu_table, lengths)
        for use_other_stream in [False, True]:
            stream = torch.npu.Stream() if use_other_stream else torch.npu.current_stream()
            with torch.npu.stream(stream):
                target = cpu_cache.npu()
                table_target = cpu_table.npu()
                cache = target.clone()
                table = table_target.clone()
                lookup = reference._get_byte_lut(cache.device)
                metadata = SimpleNamespace()
                # Prime logical jobs outside the dependency test; otherwise
                # their first H2D allocation can hide a missing queue edge.
                reader(cache, table, lengths, lookup, metadata)
                torch.npu.synchronize()
                for repeat in range(5):
                    cache.fill_(repeat + 1)
                    cache.copy_(target)
                    table.zero_()
                    table.copy_(table_target)
                    # Release temporary inputs immediately. The queued handler
                    # must retain them and execute after their GPU producers,
                    # before the standard PyTorch clone consumer.
                    consumed = reader(cache.clone(), table.clone(), lengths, lookup.clone(), metadata).clone()
                    actual_cpu = consumed.cpu()
                    mismatch = (actual_cpu.view(torch.int16) != expected_cpu.view(torch.int16)).sum().item()
                    record = {"variant": "async_producer_reader_consumer", "lengths": lengths,
                              "heads": heads, "nondefault_stream": use_other_stream,
                              "repeat": repeat, "mismatched_bf16_bits": mismatch}
                    order_records.append(record)
                    print(json.dumps(record), flush=True)
                    assert mismatch == 0, record
    assert len(order_records) == 20
    (output_dir / "verify-native-read.json").write_text(
        json.dumps(
            {
                "passed": len(records),
                "lookup_records": lookup_records,
                "async_passed": len(order_records),
                "async_records": order_records,
                "records": records,
                "timings": timings,
                "scope": "strict original BF16 read bits; micro latency is not model throughput",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
