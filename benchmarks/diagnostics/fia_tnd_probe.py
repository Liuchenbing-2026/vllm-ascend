# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare installed TND FIA APIs on an idle, explicitly selected NPU.

This measures synchronized operator wall time, not serving throughput. The
caller must check physical-card ownership before launching. Native v2 with a
Tensor length is a hypothesis under test, not a guaranteed device-length path.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu
from torch.utils._python_dispatch import TorchDispatchMode


class CountScalarReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.scalar_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads += 1
        return func(*args, **(kwargs or {}))


def invoke_fia(variant, *, q, k, v, table, block_size, qlens, lengths, heads, kv_heads, head_dim, device_lengths):
    common = dict(block_table=table, input_layout="TND", block_size=block_size, sparse_mode=0)
    if variant == "v1_list":
        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            actual_seq_lengths=qlens,
            actual_seq_lengths_kv=lengths,
            num_heads=heads,
            num_key_value_heads=kv_heads,
            scale=head_dim**-0.5,
            **common,
        )[0]
    return torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        actual_seq_qlen=qlens,
        actual_seq_kvlen=device_lengths if variant == "v2_tensor" else lengths,
        num_query_heads=heads,
        num_key_value_heads=kv_heads,
        softmax_scale=head_dim**-0.5,
        **common,
    )[0]


def run(args):
    torch.npu.set_device(args.device)
    torch.manual_seed(16465)
    rows = []
    for batch in (1, 8):
        for query_len in (1, 9):
            block_size, heads, kv_heads, head_dim = 128, 4, 2, 128
            q = torch.randn(batch * query_len, heads, head_dim, device="npu", dtype=torch.bfloat16)
            k = torch.randn(batch * 2, block_size, kv_heads * head_dim, device="npu", dtype=q.dtype)
            v = torch.randn_like(k)
            table = torch.arange(batch * 2, device="npu", dtype=torch.int32).view(batch, 2)
            qlens = [(i + 1) * query_len for i in range(batch)]
            # Reuse identical shapes while changing live lengths to catch stale metadata.
            for step in (0, 1):
                lengths = [53 + step + (i % 2) * 106 for i in range(batch)]
                device_lengths = torch.tensor(lengths, device="npu", dtype=torch.int64)

                invoke = partial(
                    invoke_fia,
                    q=q,
                    k=k,
                    v=v,
                    table=table,
                    block_size=block_size,
                    qlens=qlens,
                    lengths=lengths,
                    heads=heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    device_lengths=device_lengths,
                )

                expected = invoke("v1_list")
                torch.npu.synchronize()
                for variant in ("v1_list", "v2_list", "v2_tensor"):
                    row = dict(batch=batch, query_len=query_len, step=step, variant=variant)
                    try:
                        monitor = CountScalarReads()
                        with monitor:
                            actual = invoke(variant)
                        torch.npu.synchronize()
                        torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
                        row["scalar_reads_requested"] = monitor.scalar_reads
                        for _ in range(3):
                            invoke(variant)
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            invoke(variant)
                        torch.npu.synchronize()
                        row.update(
                            status="pass", synchronized_wall_ms=(time.perf_counter() - start) * 1000 / args.iterations
                        )
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    report = dict(scope="operator-only, TND, noncausal, paged BF16; not TP4 serving", rows=rows)
    Path(args.output).write_text(json.dumps(report, indent=2))
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
