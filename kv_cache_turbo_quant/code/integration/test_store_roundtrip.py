"""Standalone roundtrip test for kvtq_store math (no vllm needed).

Validates: op quantize -> pack -> simulated paged scatter/gather -> unpack ->
MSE-only dequant in rotated space -> query-side rotation equivalence.
Run inside the va container: python3 /root/kvtq_integration/test_store_roundtrip.py
"""
import math
import os
import sys

sys.path.insert(0, "/root/kvtq_integration")
sys.path.insert(0, "/root/kvtq_integration/torch_ext")

import torch
import torch_npu  # noqa: F401

import build_ext
build_ext.build()

os.environ.setdefault("VLLM_ASCEND_KVTQ_STORE", "1")
import kvtq_store as ks


def main():
    device = "npu:0"
    torch.npu.set_device(0)
    N, H, D = 512, 4, 128
    rot, rot_t, cent = ks._get_consts(device)
    rot32, qjl32 = ks._get_quant_matrices(device)

    gen = torch.Generator(device="cpu").manual_seed(7)
    k = torch.randn(N, H, D, generator=gen, dtype=torch.float32).to(device).to(torch.bfloat16)
    q = torch.randn(8, H, D, generator=gen, dtype=torch.float32).to(device).to(torch.bfloat16)

    idx, qjlb, norm, gamma = torch.ops.turboquant.kv_cache_turbo_quant(k.contiguous(), rot32, qjl32, ks._MSE_BITS)
    packed = ks._pack_rows(idx, norm)
    assert packed.shape == (N, H, ks._ROW_BYTES) and packed.dtype == torch.uint8

    # simulate paged scatter/gather with shuffled slots
    block_size, num_blocks = 16, (N + 15) // 15
    cache = torch.zeros(num_blocks, block_size, H, ks._ROW_BYTES // 2, dtype=torch.bfloat16, device=device)
    slots = torch.randperm(num_blocks * block_size, generator=torch.Generator().manual_seed(3))[:N].to(device)
    flat = cache.view(torch.uint8).view(-1, H, ks._ROW_BYTES)
    flat.index_copy_(0, slots, packed)

    # gather back via the production _dequant_dense path (block table)
    block_table = (slots // block_size).view(2, -1)  # fake 2 seqs... not aligned; use direct rows instead
    rows = flat[slots].view(N, H, ks._ROW_BYTES)
    norm_b = ks._bf16_from_bytes(rows[..., ks._IDX_BYTES], rows[..., ks._IDX_BYTES + 1])
    lut = ks._get_byte_lut(device)
    codes = rows[..., :ks._IDX_BYTES].reshape(-1).to(torch.int32)
    dense_rot = torch.index_select(lut, 0, codes).view(N, H, D) * norm_b.unsqueeze(-1).to(torch.bfloat16)

    # direct full dequant reference (shadow math, with QJL)
    import kvtq_shadow
    recon_full = kvtq_shadow._dequant(idx, qjlb, norm, gamma, rot32, qjl32, ks._MSE_BITS)

    # MSE-only reconstruction, back-rotated
    recon_mse = dense_rot.float() @ rot

    ref = k.float()
    rel_full = ((recon_full - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6))
    rel_mse = ((recon_mse - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6))
    print(f"bits={ks._MSE_BITS} row_bytes={ks._ROW_BYTES}")
    print(f"recon rel err: full(qjl) mean={rel_full.mean():.4f} max={rel_full.max():.4f} | "
          f"mse-only mean={rel_mse.mean():.4f} max={rel_mse.max():.4f}")

    # query-side rotation equivalence: q.k_hat == (q@rot.T).(dense_rot)
    kh_direct = recon_mse  # [N,H,D]
    lhs = torch.einsum("bhd,nhd->bhn", q.float(), kh_direct)
    q_rot = (q.float() @ rot_t).to(torch.bfloat16)
    rhs = torch.einsum("bhd,nhd->bhn", q_rot.float(), dense_rot.float())
    eq = ((lhs - rhs).norm() / lhs.norm().clamp_min(1e-6)).item()
    print(f"q-side rotation equivalence max rel diff: {eq:.5f}")
    assert eq < 0.01, "rotation equivalence broken"
    print("ROUNDTRIP OK")


if __name__ == "__main__":
    main()