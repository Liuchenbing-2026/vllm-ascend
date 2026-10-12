import os
import sys

sys.path.insert(0, "/root/kvtq_torch_ext")
import torch
import torch_npu  # noqa: F401
import build_ext

build_ext.build(verbose=False)

torch.manual_seed(0)
dev = "npu:0"
fails = 0
for (N, H, D, bs, nb) in [(1, 4, 128, 128, 8), (37, 4, 128, 128, 8), (1000, 4, 128, 128, 64),
                          (64, 2, 256, 128, 16), (5, 8, 64, 128, 4), (8192, 4, 128, 128, 70)]:
    x = (torch.randn(N, H, D) * 2).bfloat16().to(dev)
    scale = (x.float().abs().amax(dim=0) / 127.0).clamp_min(1e-8)
    cache_ref = torch.zeros(nb, bs, H, D, dtype=torch.int8, device=dev)
    cache_new = cache_ref.clone()
    slots = torch.randperm(nb * bs, device=dev)[:N].to(torch.int32)
    # reference: pytorch op chain
    q = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    view5 = cache_ref.view(nb, H, D // 32, bs, 32)
    sl = slots.long()
    view5[sl // bs, :, :, sl % bs, :] = q.view(-1, H, D // 32, 32)
    # fused kernel
    torch.ops.turboquant.kvtq_int8_store(x, scale, slots, cache_new)
    torch.npu.synchronize()
    mism = (cache_ref != cache_new).sum().item()
    status = "PASS" if mism == 0 else "FAIL"
    if mism:
        fails += 1
        idx = (cache_ref != cache_new).nonzero()[0].tolist()
        print(f"  first mismatch at {idx}: ref={cache_ref[tuple(idx)].item()} new={cache_new[tuple(idx)].item()}")
    print(f"N={N} H={H} D={D}: {status} mismatches={mism}")
print("ALL PASS" if fails == 0 else f"{fails} CASES FAILED")