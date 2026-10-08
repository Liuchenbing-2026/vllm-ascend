# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile one warmed device-tiling draft layer, including metadata construction."""

import argparse
import csv
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(args.device)
    torch.ops.load_library(args.library)
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/draft_tnd.py"
    spec = importlib.util.spec_from_file_location("draft_tnd_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(16465)
    batch, qlen, heads, kvheads, dim, block = 8, 9, 4, 2, 128, 128
    query = torch.randn(batch * qlen, heads, dim, dtype=torch.bfloat16, device="npu")
    key = torch.randn(batch * 2, block, kvheads * dim, dtype=query.dtype, device="npu")
    value = torch.randn_like(key)
    lengths = torch.tensor([53 + (i % 2) * 106 for i in range(batch)], device="npu", dtype=torch.int32)
    starts = torch.arange(0, (batch + 1) * qlen, qlen, dtype=torch.int32, device="npu")
    table = torch.arange(batch * 2, device="npu", dtype=torch.int32).view(batch, 2)

    def invoke():
        output = module.draft_tnd_attention(
            query,
            key,
            value,
            seq_lens=lengths,
            query_start_loc=starts,
            block_table=table,
            block_size=block,
            num_heads=heads,
            num_kv_heads=kvheads,
            scale=dim**-0.5,
            causal=False,
            sliding_window=None,
            attn_mask=None,
            cache={},
        )
        if output is None:
            raise RuntimeError("candidate fell back")
        return output

    for _ in range(20):
        invoke()
    torch.npu.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        invoke()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - begin) * 10
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    calls = 0
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
        experimental_config=config,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        invoke()
        torch.npu.synchronize()
        prof.step()
        begin = time.perf_counter()
        while time.perf_counter() - begin < 3:
            with torch.profiler.record_function("draft_tnd_with_metadata"):
                invoke()
            calls += 1
        torch.npu.synchronize()
        prof.step()
    outputs = list(args.output.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one parsed trace, got {outputs}")
    output = outputs[0]
    data = json.loads((output / "trace_view.json").read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data
    counts = Counter(e.get("name", "") for e in events if e.get("ph") == "X")
    with (output / "kernel_details.csv").open() as f:
        columns = len(next(csv.reader(f)))
    assert columns >= 40 and (output / "api_statistic.csv").exists() and (output / "op_statistic.csv").exists()
    result = dict(
        scope="operator only, one layer plus metadata per call; not serving performance",
        synchronized_wall_ms=wall_ms,
        calls=calls,
        requested_seconds=3,
        kernel_columns=columns,
        markers=counts["draft_tnd_with_metadata"],
        related_events={
            n: c for n, c in counts.items() if any(w in n.lower() for w in ("synchron", "memcpy", "scalar", "item"))
        },
        trace_path=str(output),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
