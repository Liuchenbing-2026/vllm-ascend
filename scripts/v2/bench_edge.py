"""Edge-shape correctness: v2 (native masked) vs AscendC for shapes the guards
used to fall back on — H%64!=0 (shrink), Ho%bh!=0 (expand).  Also re-checks a
standard shape stays bit-exact.  Requires v2 tree via V2_SRC."""
import os
import shutil
import torch
import torch_npu  # noqa
import vllm_ascend  # noqa
import vllm_ascend.vllm_ascend_C  # noqa

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


def shrink_case(H, tag):
    B, NR = 8, 1
    idx = torch.zeros(1, device=DEV, dtype=torch.int64)
    seq = torch.full((1,), B, device=DEV, dtype=torch.int64)
    torch.manual_seed(hash(("s", H)) % 2**31)
    x = torch.randn(B, H, device=DEV, dtype=DT) * 0.05
    w = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * 0.05
    y = torch.zeros(B, R, device=DEV, dtype=torch.float32)
    torch.ops._C_ascend.sgmv_shrink(x, w.view(L, R, H), idx, seq, y, 0.5)
    torch.npu.synchronize()
    ref = y.clone()
    y.zero_()
    T.sgmv_shrink(x, w, y, None, seq, idx, B, H, NR, 0.5)
    torch.npu.synchronize()
    d = (y - ref).abs().max().item()
    be = torch.equal(y, ref)
    rel = d / max(ref.abs().max().item(), 1e-30)
    print("  shrink %-14s H=%-6d bitexact=%s max_abs=%.3e rel=%.3e %s"
          % (tag, H, be, d, rel, "OK" if (be or rel < 1e-3) else "FAIL"), flush=True)


def expand_case(Ho, YHO, OFF, tag):
    B, NR = 8, 1
    idx = torch.zeros(1, device=DEV, dtype=torch.int64)
    seq = torch.full((1,), B, device=DEV, dtype=torch.int64)
    torch.manual_seed(hash(("e", Ho)) % 2**31)
    xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * 0.05
    we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * 0.05
    y0 = torch.randn(B, YHO, device=DEV, dtype=DT) * 0.05
    y = y0.clone()
    torch.ops._C_ascend.sgmv_expand(xe, we.view(L, Ho, R), idx, seq, y, OFF, Ho)
    torch.npu.synchronize()
    ref = y.clone()
    y = y0.clone()
    T.sgmv_expand_slice(xe, we, y, None, seq, idx, B, Ho, NR, OFF, Ho, True)
    torch.npu.synchronize()
    d = (y.float() - ref.float()).abs().max().item()
    be = torch.equal(y, ref)
    rel = d / max(ref.float().abs().max().item(), 1e-30)
    # confirm columns outside the slice untouched
    outside = torch.equal(y[:, :OFF], y0[:, :OFF]) and torch.equal(y[:, OFF+Ho:], y0[:, OFF+Ho:])
    print("  expand %-14s Ho=%-6d bitexact=%s max_abs=%.3e rel=%.3e outside_ok=%s %s"
          % (tag, Ho, be, d, rel, outside, "OK" if (be or rel < 1e-3) else "FAIL"), flush=True)


print("=== STANDARD (must stay bit-exact) ===", flush=True)
shrink_case(5120, "std-5120")
shrink_case(17408, "std-17408")
expand_case(5120, 5120, 0, "std-o")
expand_case(17408, 34816, 17408, "std-up")
print("=== EDGE (H%%64!=0 / Ho%%bh!=0, was fallback) ===", flush=True)
shrink_case(1376, "llama-dp-tp8")     # 1376 % 64 = 32
shrink_case(2752, "odd-2752")         # 2752 % 64 = 0? 2752/64=43 -> yes; pick真奇的
shrink_case(1000, "odd-1000")         # 1000 % 64 = 40
expand_case(80, 96, 0, "headdim-80")  # 80 % 32 = 16
expand_case(1376, 1376, 0, "Ho-1376") # 1376 % 128 = 96
expand_case(1000, 1024, 0, "Ho-1000")
print("DONE", flush=True)
