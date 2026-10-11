"""Standalone glue test: torch.ops.turboquant vs golden.py reference on NPU."""
import os
import sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")

import numpy as np
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ext import build  # noqa: E402

build()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
from golden import calc_expect_func  # noqa: E402

torch.npu.set_device(0)
gen = torch.Generator().manual_seed(20260929)

CASES = [
    (1, 8, 2), (16, 8, 3), (256, 8, 4), (7, 4, 3), (1024, 2, 2),
]

for num_tokens, num_kv_heads, mse_bits in CASES:
    kv = torch.randn(num_tokens, num_kv_heads, 128, generator=gen, dtype=torch.float32)
    kv = (kv * 2.0).to(torch.bfloat16).npu().contiguous()
    q, _ = torch.linalg.qr(torch.randn(128, 128, generator=gen, dtype=torch.float64))
    rot = q.to(torch.float32).npu().contiguous()
    qjl = (torch.randn(128, 128, generator=gen, dtype=torch.float64).to(torch.float32)
           / (128 ** 0.5)).npu().contiguous()

    idx, qjlb, norm, gamma = torch.ops.turboquant.kv_cache_turbo_quant(kv, rot, qjl, mse_bits)
    torch.npu.synchronize()

    exp_idx, exp_qjl, exp_norm, exp_gamma = calc_expect_func(
        kv.float().cpu().numpy(), rot.cpu().numpy(), qjl.cpu().numpy(), mse_bits)

    ok_idx = np.array_equal(idx.cpu().numpy(), exp_idx)
    ok_qjl = np.array_equal(qjlb.cpu().numpy(), exp_qjl)
    got_norm = np.asarray(norm.float().cpu().numpy(), dtype=np.float32)
    got_gamma = np.asarray(gamma.float().cpu().numpy(), dtype=np.float32)
    ok_norm = np.array_equal(got_norm, exp_norm.astype(np.float32))
    ok_gamma = np.array_equal(got_gamma, exp_gamma.astype(np.float32))
    mism = int((idx.cpu().numpy() != exp_idx).sum())
    print(f"case N={num_tokens} H={num_kv_heads} bits={mse_bits}: "
          f"idx={ok_idx} qjl={ok_qjl} norm={ok_norm} gamma={ok_gamma} idx_mismatch={mism}")
    assert ok_idx and ok_qjl and ok_norm and ok_gamma, "BITWISE MISMATCH"

print("ALL GLUE TESTS PASS")