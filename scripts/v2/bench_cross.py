"""Find the exact per-family crossover B where v2 stops beating AscendC.
Per-layer device time (7 shrink + 7 expand modules), NR=1, C++ launcher."""
import os, shutil, time
import torch, torch_npu  # noqa
import vllm_ascend, vllm_ascend.vllm_ascend_C  # noqa

VA = os.path.dirname(vllm_ascend.__file__)
SRC = os.environ.get("V2_SRC", "/work/v2new")
for f in ("lora_ops_triton.py", "lora_ops_triton_kernels.py",
          "lora_cpp_launcher.cpp",
          "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so"):
    shutil.copy(os.path.join(SRC, f), os.path.join(VA, "lora", f))
os.environ["TRITON_LORA_CPP"] = "1"
os.environ["TRITON_LORA_EXACT"] = "1"
from vllm_ascend.lora import lora_ops_triton as T   # noqa: E402

DEV, DT, R, L = "npu", torch.bfloat16, 16, 2
SHRINK_H = (5120, 5120, 5120, 6144, 5120, 5120, 17408)
EXPAND = [(6144, 8192, 0), (1024, 8192, 6144), (1024, 8192, 7168),
          (5120, 5120, 0), (17408, 34816, 0), (17408, 34816, 17408),
          (5120, 5120, 0)]


def dev_us(fn, ncap=20, nrep=10):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(ncap):
            fn()
    torch.npu.synchronize()
    g.replay(); torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(nrep):
        g.replay()
    torch.npu.synchronize()
    dt = time.perf_counter() - t0
    del g
    return dt * 1e6 / (ncap * nrep)


print("B    | shrink asc / v2  ratio | expand asc / v2  ratio", flush=True)
for B in (1, 2, 3, 4, 6, 8, 16):
    idx = torch.zeros(1, device=DEV, dtype=torch.int64)
    seq = torch.full((1,), B, device=DEV, dtype=torch.int64)
    torch.manual_seed(7)
    sa = sv = ea = ev = 0.0
    for H in SHRINK_H:
        x = torch.randn(B, H, device=DEV, dtype=DT) * 0.05
        w = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * 0.05
        y = torch.zeros(B, R, device=DEV, dtype=torch.float32)
        wv = w.view(L, R, H)
        sa += dev_us(lambda: torch.ops._C_ascend.sgmv_shrink(x, wv, idx, seq, y, 0.5))
        sv += dev_us(lambda: T.sgmv_shrink(x, w, y, None, seq, idx, B, H, 1, 0.5))
    for Ho, YHO, OFF in EXPAND:
        xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * 0.05
        we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * 0.05
        y = torch.zeros(B, YHO, device=DEV, dtype=DT)
        wev = we.view(L, Ho, R)
        ea += dev_us(lambda: torch.ops._C_ascend.sgmv_expand(xe, wev, idx, seq, y, OFF, Ho))
        ev += dev_us(lambda: T.sgmv_expand_slice(xe, we, y, None, seq, idx, B, Ho, 1, OFF, Ho, True))
    print("%-4d | %7.1f %7.1f  %.2fx | %7.1f %7.1f  %.2fx  %s"
          % (B, sa, sv, sv / sa, ea, ev, ev / ea,
             "shrink:v2" if sv < sa else "shrink:ASC"), flush=True)
print("DONE", flush=True)
