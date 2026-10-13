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
    parser.add_argument("--output-dir", default="/ws/read_tiles_20261014")
    parser.add_argument("--precision-only", action="store_true")
    args = parser.parse_args()
    root = Path("/ws/read_tiles_20261014")
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
    (output_dir / "verify-native-read.json").write_text(
        json.dumps(
            {
                "passed": len(records),
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
