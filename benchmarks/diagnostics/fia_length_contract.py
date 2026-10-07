# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit the installed FIA length contract without allocating NPU tensors.

This is an interface diagnostic, not a device benchmark or a fix. It reports
whether passing a CPU Tensor to the list-valued API requests scalar extraction.
A positive result identifies an interface conversion, not measured D2H latency.
"""

import importlib.metadata
import json

import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode


class RejectScalarExtraction(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.scalar_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads += 1
            raise RuntimeError("DIAGNOSTIC: tensor scalar extraction requested")
        return func(*args, **(kwargs or {}))


def audit():
    q = torch.empty(1, 1, 16)
    k = torch.empty(1, 1, 16)
    v = torch.empty_like(k)
    lengths = torch.tensor([1], dtype=torch.int64)
    rows = []
    variants = [
        ("npu_fused_infer_attention_score", "actual_seq_lengths_kv", "num_heads"),
        ("npu_fused_infer_attention_score_v2", "actual_seq_kvlen", "num_query_heads"),
    ]
    for name, parameter, heads in variants:
        op = getattr(torch.ops.npu, name).default
        schema = str(op._schema)
        for kind, value in [("host_list", [1]), ("cpu_tensor", lengths)]:
            monitor = RejectScalarExtraction()
            error = None
            try:
                with monitor:
                    op(q, k, v, **{parameter: value, heads: 1, "input_layout": "TND"})
            except Exception as exc:
                # CPU dispatch failure is expected: no NPU operation is requested.
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            rows.append(
                {
                    "op": name,
                    "length_parameter": parameter,
                    "schema": schema,
                    "length_input": kind,
                    "scalar_reads_requested": monitor.scalar_reads,
                    "outcome": error or "returned",
                }
            )
    return {
        "scope": "CPU argument-binding diagnostic; no NPU tensors or performance measurements",
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "torch-npu"]},
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
