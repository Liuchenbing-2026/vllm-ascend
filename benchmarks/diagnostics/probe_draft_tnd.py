# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2 device-tiling candidate: FP32 reference, stale-tail and live-length checks.

Uses random tensors, not model weights. Run only on an idle allocated device.
The 2% relative L2 threshold is inherited from the existing draft-tail ST.
"""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("device length converted to a host scalar")
        return func(*args, **(kwargs or {}))


def check_case(attention, batch, ragged, causal, window, invalid_value, dim):
    heads, kv_heads, page_size, pages = 4, 2, 128, 2
    capacity = page_size * pages
    qcounts = [8 + (i % 2 if ragged else 0) for i in range(batch)]
    starts = [0]
    for count in qcounts:
        starts.append(starts[-1] + count)
    query = torch.randn(starts[-1], heads, dim, dtype=torch.bfloat16)
    keys = torch.randn(batch, capacity, kv_heads, dim, dtype=query.dtype)
    values = torch.randn_like(keys)
    order = torch.randperm(batch * pages)
    table = order.reshape(batch, pages).to(device="npu", dtype=torch.int32)
    device_query = query.npu()
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device="npu")
    seq_lens = torch.empty(batch, dtype=torch.int32, device="npu")
    attn_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu() if causal else None
    errors = []
    for step in (0, 1):
        lengths = [127 + step + 52 * (i % 2) for i in range(batch)]
        seq_lens.copy_(torch.tensor(lengths, device="npu", dtype=torch.int32))
        step_key, step_value = keys.clone(), values.clone()
        for req, length in enumerate(lengths):
            step_key[req, length:] = invalid_value
            step_value[req, length:] = invalid_value
        key_pages = torch.empty(batch * pages, page_size, kv_heads, dim, dtype=query.dtype)
        value_pages = torch.empty_like(key_pages)
        key_pages[order] = step_key.reshape_as(key_pages)
        value_pages[order] = step_value.reshape_as(value_pages)
        # Reproduce physical interleaved K/V storage and expanded block IDs.
        backing = torch.stack((key_pages, value_pages), dim=1).npu()
        flat_storage = backing.flatten(0, 1).flatten(2)
        device_key = flat_storage[:-1]
        device_value = flat_storage[1:]
        physical_table = table * 2
        before = backing.clone()
        cache = {}
        with NoHostScalar():
            actual = attention(
                device_query,
                device_key,
                device_value,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                block_table=physical_table,
                block_size=page_size,
                num_heads=heads,
                num_kv_heads=kv_heads,
                scale=dim**-0.5,
                causal=causal,
                sliding_window=window,
                attn_mask=attn_mask,
                cache=cache,
            )
        if actual is None:
            raise AssertionError("candidate fell back instead of testing device tiling")
        torch.npu.synchronize()
        references = []
        for req, length in enumerate(lengths):
            q = query[starts[req] : starts[req + 1]].float().permute(1, 0, 2)
            k = keys[req, :length].float().repeat_interleave(heads // kv_heads, dim=1).permute(1, 0, 2)
            v = values[req, :length].float().repeat_interleave(heads // kv_heads, dim=1).permute(1, 0, 2)
            scores = q @ k.transpose(1, 2) * dim**-0.5
            last = (
                torch.arange(qcounts[req]) + length - qcounts[req]
                if causal
                else torch.full((qcounts[req],), length - 1)
            )
            visible = torch.arange(length).view(1, -1) <= last.view(-1, 1)
            if window is not None:
                visible &= torch.arange(length).view(1, -1) >= last.view(-1, 1) - window
            scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
            references.append((scores.softmax(-1) @ v).permute(1, 0, 2))
        reference = torch.cat(references)
        result = actual.cpu().float()
        assert torch.isfinite(result).all(), "nonfinite tail contaminated valid attention"
        relative_l2 = float((result - reference).norm() / reference.norm())
        assert relative_l2 < 0.02, f"relative L2 {relative_l2} >= 0.02"
        torch.testing.assert_close(backing, before, rtol=0, atol=0, equal_nan=True)
        errors.append(relative_l2)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.ops.load_library(args.library)
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/draft_tnd.py"
    spec = importlib.util.spec_from_file_location("draft_tnd_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(16465)
    rows = []
    for batch, ragged, mode, invalid, dim in itertools.product(
        (2, 8),
        (False, True),
        ((False, None), (True, None), (True, 16)),
        (100.0, float("nan"), float("inf")),
        args.head_dims,
    ):
        causal, window = mode
        row = dict(batch=batch, ragged=ragged, causal=causal, window=window, invalid=str(invalid), head_dim=dim)
        try:
            row["relative_l2"] = check_case(module.draft_tnd_attention, batch, ragged, causal, window, invalid, dim)
            row["status"] = "pass"
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        args.output.write_text(
            json.dumps(dict(scope="A2 operator only; no model or graph validation", rows=rows), indent=2)
        )
        print(json.dumps(row), flush=True)
        if row["status"] != "pass":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
