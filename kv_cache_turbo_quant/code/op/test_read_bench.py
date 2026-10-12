import os
import sys
import time

sys.path.insert(0, "/root/kvtq_torch_ext")
import torch
import torch_npu  # noqa: F401
import build_ext

build_ext.build(verbose=False)

dev = "npu:0"
PACKED_ROW = 96   # 64B codes + 2B norm padded to 32B*3
DENSE_ROW = 256   # 128 x bf16

# c8 single-layer K+V rows; c1 for scale reference
CASES = {
    "c1_kv": 1 * 131072 * 4 * 2,
    "c8_kv": 8 * 131072 * 4 * 2,
}

for name, rows in CASES.items():
    rows = (rows + 127) // 128 * 128
    packed = torch.randint(0, 256, (rows * PACKED_ROW + 12288,), dtype=torch.uint8, device=dev)
    dense = torch.randn(rows * DENSE_ROW // 2 + 16384, dtype=torch.float16, device=dev).bfloat16()
    print(f"=== {name}: rows={rows} packed={rows*PACKED_ROW/1e6:.0f}MB dense={rows*DENSE_ROW/1e6:.0f}MB ===")
    for mode, label in ((0, "packed read floor"), (1, "packed read + dequant"), (2, "bf16 read floor")):
        for _ in range(3):
            torch.ops.turboquant.kvtq_read_bench(packed, dense, mode, rows)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            torch.ops.turboquant.kvtq_read_bench(packed, dense, mode, rows)
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / iters
        nbytes = rows * (DENSE_ROW if mode == 2 else PACKED_ROW)
        print(f"  mode{mode} {label:22s}: {dt*1e3:8.3f} ms  {nbytes/dt/1e12:6.2f} TB/s")
    del packed, dense
    torch.npu.empty_cache()
print("BENCH DONE")