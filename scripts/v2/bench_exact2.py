"""Round 2: bitwise sweep vs AscendC with the derived-B expand kernel and the
tl.split EXACT shrink, plus per-family timing.  On any cpp mismatch the same
config is re-run through the eager triton launcher to separate launcher bugs
from math differences, and small-case mismatch positions are dumped.
"""
import os
import shutil
import time

import torch
import torch_npu  # noqa
import vllm_ascend  # noqa
import vllm_ascend.vllm_ascend_C  # noqa

VA = os.path.dirname(vllm_ascend.__file__)
for f in ("lora_ops_triton.py", "lora_ops_triton_kernels.py",
          "lora_cpp_launcher.cpp",
          "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so"):
    shutil.copy(os.path.join(os.environ.get("V2_SRC", "/work/v2"), f), os.path.join(VA, "lora", f))
os.environ["TRITON_LORA_CPP"] = "1"
os.environ["TRITON_LORA_EXACT"] = "1"

from vllm_ascend.lora import lora_ops_triton as T   # noqa: E402

DEV = "npu"
DT = torch.bfloat16
R = 16
L = 2
SHRINK_H = (5120, 5120, 5120, 6144, 5120, 5120, 17408)
EXPAND = [(6144, 8192, 0, "q"), (1024, 8192, 6144, "k"), (1024, 8192, 7168, "v"),
          (5120, 5120, 0, "o"), (17408, 34816, 0, "gate"),
          (17408, 34816, 17408, "up"), (5120, 5120, 0, "down")]
BS = (1, 2, 3, 4, 7, 8, 16, 64, 256, 1024)
FAILS = []


def segs(B, NR):
    idx = torch.arange(NR, device=DEV, dtype=torch.int64) % L
    seq = torch.full((NR,), B // NR, device=DEV, dtype=torch.int64)
    seq[-1] += B - (B // NR) * NR
    return idx, seq


def report(tag, got, ref, eag=None):
    if torch.equal(got, ref):
        return
    n = int((got != ref).sum())
    d = (got.float() - ref.float()).abs().max()
    en = "-" if eag is None else int((eag != ref).sum())
    FAILS.append(tag)
    print("  MISMATCH %-34s cpp_ndiff=%d/%d max=%.3e eager_ndiff=%s"
          % (tag, n, ref.numel(), float(d), en), flush=True)
    if n and n <= 200:
        pos = (got != ref).nonzero()[:3]
        for p in pos:
            i = tuple(int(v) for v in p)
            print("      at %s cpp=%.6f ref=%.6f eager=%s"
                  % (i, float(got[i]), float(ref[i]),
                     "-" if eag is None else "%.6f" % float(eag[i])), flush=True)


def sweep(scale, seed):
    print("=== bitwise sweep scale=%s seed=%d (cpp, EXACT=1) ===" % (scale, seed),
          flush=True)
    for B in BS:
        for NR in (1, 2, 3):
            if NR > B:
                continue
            idx, seq = segs(B, NR)
            torch.manual_seed(seed * 100 + B)
            for H in (5120, 6144, 17408):
                x = torch.randn(B, H, device=DEV, dtype=DT) * scale
                w = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * scale
                y = torch.zeros(B, R, device=DEV, dtype=torch.float32)
                torch.ops._C_ascend.sgmv_shrink(x, w.view(L, R, H), idx, seq, y, 0.5)
                torch.npu.synchronize()
                ref = y.clone()
                y.zero_()
                T.sgmv_shrink(x, w, y, None, seq, idx, B, H, NR, 0.5)
                torch.npu.synchronize()
                eag = None
                if not torch.equal(y, ref):
                    got = y.clone()
                    y.zero_()
                    os.environ["TRITON_LORA_CPP"] = "0"
                    T.sgmv_shrink(x, w, y, None, seq, idx, B, H, NR, 0.5)
                    torch.npu.synchronize()
                    os.environ["TRITON_LORA_CPP"] = "1"
                    eag = y.clone()
                    y = got
                report("shrink H=%d B=%d NR=%d" % (H, B, NR), y, ref, eag)
            for Ho, YHO, OFF, lab in EXPAND:
                xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * scale
                we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * scale
                y0 = torch.randn(B, YHO, device=DEV, dtype=DT) * scale
                y = y0.clone()
                torch.ops._C_ascend.sgmv_expand(xe, we.view(L, Ho, R), idx, seq, y, OFF, Ho)
                torch.npu.synchronize()
                ref = y.clone()
                y = y0.clone()
                T.sgmv_expand_slice(xe, we, y, None, seq, idx, B, Ho, NR, OFF, Ho, True)
                torch.npu.synchronize()
                eag = None
                if not torch.equal(y, ref):
                    got = y
                    y = y0.clone()
                    os.environ["TRITON_LORA_CPP"] = "0"
                    T.sgmv_expand_slice(xe, we, y, None, seq, idx, B, Ho, NR, OFF, Ho, True)
                    torch.npu.synchronize()
                    os.environ["TRITON_LORA_CPP"] = "1"
                    eag = y
                    y = got
                report("expand %s B=%d NR=%d" % (lab, B, NR), y, ref, eag)
        print("  B=%-5d done, fails so far: %d" % (B, len(FAILS)), flush=True)


def dev_us(fn, ncap=20, nrep=10):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(ncap):
            fn()
    torch.npu.synchronize()
    g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(nrep):
        g.replay()
    torch.npu.synchronize()
    dt = time.perf_counter() - t0
    del g
    return dt * 1e6 / (ncap * nrep)


def timing():
    print("", flush=True)
    print("=== per-LAYER device us by family (7 shrink + 7 expand modules, NR=1) ===",
          flush=True)
    for B in (1, 4, 8, 64, 256, 1024):
        idx, seq = segs(B, 1)
        sa = se1 = se0 = ea = ev = 0.0
        torch.manual_seed(7)
        for H in SHRINK_H:
            x = torch.randn(B, H, device=DEV, dtype=DT) * 0.05
            w = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * 0.05
            y = torch.zeros(B, R, device=DEV, dtype=torch.float32)
            wv = w.view(L, R, H)
            sa += dev_us(lambda: torch.ops._C_ascend.sgmv_shrink(x, wv, idx, seq, y, 0.5))
            T._V2_EXACT = 1
            se1 += dev_us(lambda: T.sgmv_shrink(x, w, y, None, seq, idx, B, H, 1, 0.5))
            T._V2_EXACT = 0
            se0 += dev_us(lambda: T.sgmv_shrink(x, w, y, None, seq, idx, B, H, 1, 0.5))
            T._V2_EXACT = 1
        for Ho, YHO, OFF, lab in EXPAND:
            xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * 0.05
            we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * 0.05
            y = torch.zeros(B, YHO, device=DEV, dtype=DT)
            wev = we.view(L, Ho, R)
            ea += dev_us(lambda: torch.ops._C_ascend.sgmv_expand(xe, wev, idx, seq, y, OFF, Ho))
            ev += dev_us(lambda: T.sgmv_expand_slice(xe, we, y, None, seq, idx,
                                                     B, Ho, 1, OFF, Ho, True))
        print("  B=%-5d shrink: asc %8.1f  v2ex1 %8.1f (%.2fx)  v2ex0 %8.1f (%.2fx) | "
              "expand: asc %8.1f  v2 %8.1f (%.2fx) | SHIP ex1+v2 %8.1f vs asc %8.1f = %.2fx"
              % (B, sa, se1, se1 / sa, se0, se0 / sa, ea, ev, ev / ea,
                 se1 + ev, sa + ea, (se1 + ev) / (sa + ea)), flush=True)



def negseg():
    print("", flush=True)
    print("=== no-lora segment (idx=-1) skip semantics, planted -0.0 ===", flush=True)
    B, NR = 8, 2
    idx = torch.tensor([0, -1], device=DEV, dtype=torch.int64)
    seq = torch.tensor([4, 4], device=DEV, dtype=torch.int64)
    torch.manual_seed(99)
    for H in (5120, 17408):
        x = torch.randn(B, H, device=DEV, dtype=DT) * 0.05
        w = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * 0.05
        y_init = torch.randn(B, R, device=DEV, dtype=torch.float32)
        y_init[5] = -0.0
        y = y_init.clone()
        torch.ops._C_ascend.sgmv_shrink(x, w.view(L, R, H), idx, seq, y, 0.5)
        torch.npu.synchronize()
        ref = y.clone()
        y = y_init.clone()
        T.sgmv_shrink(x, w, y, None, seq, idx, B, H, NR, 0.5)
        torch.npu.synchronize()
        report("negseg shrink H=%d" % H, y, ref)
        print("  negseg shrink H=%d checked (row5 ref bits kept=%s)"
              % (H, bool((ref[5] == y_init[5]).all())), flush=True)
    for Ho, YHO, OFF, lab in ((5120, 5120, 0, "o"), (17408, 34816, 17408, "up")):
        xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * 0.05
        we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * 0.05
        y0 = torch.randn(B, YHO, device=DEV, dtype=DT) * 0.05
        y0[5] = -0.0
        y = y0.clone()
        torch.ops._C_ascend.sgmv_expand(xe, we.view(L, Ho, R), idx, seq, y, OFF, Ho)
        torch.npu.synchronize()
        ref = y.clone()
        y = y0.clone()
        T.sgmv_expand_slice(xe, we, y, None, seq, idx, B, Ho, NR, OFF, Ho, True)
        torch.npu.synchronize()
        report("negseg expand %s" % lab, y, ref)
        print("  negseg expand %s checked (row5 stays -0: %s)"
              % (lab, bool((ref[5].view(torch.int16) ==
                            y0[5].view(torch.int16)).all())), flush=True)


sweep(0.05, 0)
sweep(1.0, 1)
negseg()
timing()
print("", flush=True)
print("TOTAL BITWISE FAILS: %d" % len(FAILS), flush=True)
print("DONE", flush=True)
