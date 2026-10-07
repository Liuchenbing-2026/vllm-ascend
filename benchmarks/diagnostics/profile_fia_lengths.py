# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture a short operator-only trace after warming both FIA v2 call variants."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu
from fia_tnd_probe import invoke_fia


def run(args):
    torch.npu.set_device(args.device)
    torch.manual_seed(16465)
    batch, query_len, heads, kv_heads, head_dim, block_size = 8, 9, 4, 2, 128, 128
    q = torch.randn(batch * query_len, heads, head_dim, device="npu", dtype=torch.bfloat16)
    k = torch.randn(batch * 2, block_size, kv_heads * head_dim, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    lengths = [53 + (i % 2) * 106 for i in range(batch)]
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        table=torch.arange(batch * 2, device="npu", dtype=torch.int32).view(batch, 2),
        block_size=block_size,
        qlens=[(i + 1) * query_len for i in range(batch)],
        lengths=lengths,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        device_lengths=torch.tensor(lengths, device="npu", dtype=torch.int64),
    )
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    summary = []
    for variant in ("v2_list", "v2_tensor"):
        for _ in range(20):
            invoke_fia(variant, **kwargs)
        torch.npu.synchronize()
        target = root / variant
        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        calls = 0
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            experimental_config=config,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            invoke_fia(variant, **kwargs)
            torch.npu.synchronize()
            profiler.step()
            start = time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                with torch.profiler.record_function("fia_length_probe"):
                    invoke_fia(variant, **kwargs)
                calls += 1
            torch.npu.synchronize()
            profiler.step()
        outputs = list(target.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one trace output, got {outputs}")
        output = outputs[0]
        data = json.loads((output / "trace_view.json").read_text())
        events = data["traceEvents"] if isinstance(data, dict) else data
        counts = Counter(e.get("name", "") for e in events if e.get("ph") == "X")
        with (output / "kernel_details.csv").open() as f:
            columns = len(next(csv.reader(f)))
        if columns < 40 or not (output / "api_statistic.csv").exists():
            raise RuntimeError("Incomplete profiler detail output")
        row = dict(
            variant=variant,
            calls=calls,
            requested_seconds=args.seconds,
            kernel_columns=columns,
            marker_count=counts["fia_length_probe"],
            scalar_count=counts["aten::_local_scalar_dense"],
            related_events={
                n: c for n, c in counts.items() if any(w in n.lower() for w in ("synchron", "memcpy", "scalar", "item"))
            },
            trace_path=str(output),
        )
        summary.append(row)
        print(json.dumps(row), flush=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
