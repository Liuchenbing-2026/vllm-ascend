#!/usr/bin/env python3
"""#16271 MRV2 ceiling probe, round 5 -- one arm per process.

Round 4's table was self-contradicting: a max of absolute differences came out
negative, denormals showed up as "agreement", the mask-equality check flapped
between runs of identical code, and FIA appeared to succeed at headDim 256 after
round 3 had proved the tiling function rejects it. The common cause is that ATB
and aclnn report errors asynchronously: once one op in a process fails, later ops
in that process can return without touching their output buffer, so every number
after the first failure is uninitialised memory dressed up as a result.

So: exactly one arm per process, selected on the command line, and no arm that
is expected to fail shares a process with one that is expected to work.
Correctness is checked against a pure-torch reference computed in the same
process rather than against the other operator, which removes the dependency on
getting a second picky operator's contract right.

  python3 sf_probe5.py --mode fia|sf --D 128 --n 8 [--kv 1024] [--check]
"""

import argparse
import os
import time

import torch
import torch_npu  # noqa: F401

DTYPE = torch.bfloat16
INT_MAX = 2147483647
DEV = "npu"
NEG = -10000.0


def build(n, kv, D, heads, kv_heads, block, Q, mask_m):
    torch.manual_seed(7)
    nt = n * Q
    bpr = (kv + block - 1) // block
    nb = n * bpr + 1
    kc4 = torch.randn(nb, block, kv_heads, D, dtype=DTYPE, device=DEV)
    vc4 = torch.randn(nb, block, kv_heads, D, dtype=DTYPE, device=DEV)
    return dict(
        n=n, kv=kv, D=D, nt=nt, bpr=bpr, nb=nb, heads=heads, kv_heads=kv_heads,
        block=block, Q=Q, mask_m=mask_m,
        q=torch.randn(nt, heads, D, dtype=DTYPE, device=DEV),
        kc4=kc4, vc4=vc4,
        kc3=kc4.view(nb, block, kv_heads * D),
        vc3=vc4.view(nb, block, kv_heads * D),
        bt=torch.arange(1, n * bpr + 1, dtype=torch.int32, device=DEV).view(n, bpr),
        qlen_host=torch.full((n,), Q, dtype=torch.int32),
        kvlen_dev=torch.full((n,), kv, dtype=torch.int32, device=DEV),
        cumq=[(i + 1) * Q for i in range(n)],
        kvlist=[kv] * n,
        scale=D**-0.5,
        arange_q=torch.arange(Q, dtype=torch.int32, device=DEV),
        cols=torch.arange(mask_m, dtype=torch.int32, device=DEV),
    )


def mask_compare(c, dtype=DTYPE):
    """Per-token mask rows built on device from the device KV lengths."""
    limit = ((c["kvlen_dev"] - c["Q"]).view(-1, 1) + c["arange_q"].view(1, -1)).reshape(-1, 1)
    return (c["cols"].view(1, -1) > limit).to(dtype).mul_(NEG)


def fia(c):
    mask = torch.triu(torch.ones(c["mask_m"], c["mask_m"]), diagonal=1).to(torch.int8).to(DEV)

    def run():
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=c["q"], key=c["kc3"], value=c["vc3"], atten_mask=mask,
            block_table=c["bt"], input_layout="TND", block_size=c["block"],
            actual_seq_lengths=c["cumq"], actual_seq_lengths_kv=c["kvlist"],
            num_key_value_heads=c["kv_heads"], num_heads=c["heads"], scale=c["scale"],
            pre_tokens=INT_MAX, next_tokens=0, sparse_mode=3,
        )
        return out

    return run


def sf(c, cache="4d", ctx="dev", qlen="cpu"):
    mask = mask_compare(c)
    buf = torch.empty(c["nt"], c["heads"], c["D"], dtype=DTYPE, device=DEV)
    k, v = (c["kc4"], c["vc4"]) if cache == "4d" else (c["kc3"], c["vc3"])
    # ATB reported "tensor.hostData is null": one of these length tensors is a
    # host-data param. Which one decides whether this route avoids #16271's D2H
    # or merely relocates it.
    ctx_t = c["kvlen_dev"] if ctx == "dev" else c["kvlen_dev"].cpu()
    qlen_t = c["qlen_host"] if qlen == "cpu" else c["qlen_host"].to(DEV)

    def run():
        torch_npu._npu_paged_attention_splitfuse(
            query=c["q"], key_cache=k, value_cache=v, mask=mask, block_table=c["bt"],
            seq_len=qlen_t, context_lens=ctx_t,
            num_kv_heads=c["kv_heads"], num_heads=c["heads"], scale_value=c["scale"], out=buf,
        )
        return buf

    return run


def reference(c):
    """Plain torch attention over the same paged cache, fp32."""
    n, Q, D, H, KVH = c["n"], c["Q"], c["D"], c["heads"], c["kv_heads"]
    group = H // KVH
    out = torch.empty(c["nt"], H, D, dtype=torch.float32, device=DEV)
    for r in range(n):
        blocks = c["bt"][r].tolist()
        k = torch.cat([c["kc4"][b] for b in blocks], dim=0)[: c["kv"]].float()  # [L, KVH, D]
        v = torch.cat([c["vc4"][b] for b in blocks], dim=0)[: c["kv"]].float()
        k = k.repeat_interleave(group, dim=1)  # [L, H, D]
        v = v.repeat_interleave(group, dim=1)
        for j in range(Q):
            limit = c["kv"] - Q + j + 1
            qv = c["q"][r * Q + j].float()  # [H, D]
            s = torch.einsum("hd,lhd->hl", qv, k[:limit]) * c["scale"]
            p = torch.softmax(s, dim=-1)
            out[r * Q + j] = torch.einsum("hl,lhd->hd", p, v[:limit])
    return out


def wall(fn, iters=50):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fia", "sf", "sf3d", "mask", "calib"])
    ap.add_argument("--ctx", default="dev", choices=["dev", "cpu"])
    ap.add_argument("--qlen", default="cpu", choices=["dev", "cpu"])
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--kv", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=int(os.environ.get("NT_HEADS", "4")))
    ap.add_argument("--kv-heads", type=int, default=int(os.environ.get("NT_KV_HEADS", "1")))
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--q", type=int, default=9)
    ap.add_argument("--mask-m", type=int, default=2048)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    fa = torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV)
    fb = torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV)

    def busy():
        for _ in range(6):
            torch.mm(fa, fb)

    if a.mode == "calib":
        p = torch.zeros(64, dtype=torch.int32, device=DEV)
        s = []
        for _ in range(15):
            busy()
            t0 = time.perf_counter()
            p.tolist()
            s.append((time.perf_counter() - t0) * 1e6)
        s.sort()
        tax = s[len(s) // 2]
        s = []
        for _ in range(15):
            busy()
            t0 = time.perf_counter()
            torch.mm(fa, fb)
            s.append((time.perf_counter() - t0) * 1e6)
            torch.npu.synchronize()
        s.sort()
        print("CALIB tolist=%.0f floor=%.0f" % (tax, s[len(s) // 2]))
        return

    c = build(a.n, a.kv, a.D, a.heads, a.kv_heads, a.block, a.q, a.mask_m)

    if a.mode == "mask":
        f = lambda: mask_compare(c)  # noqa: E731
        f()
        torch.npu.synchronize()
        print("OK mode=mask D=%d n=%d wall=%.1fus" % (a.D, a.n, wall(f)))
        return

    run = fia(c) if a.mode == "fia" else sf(c, "4d" if a.mode == "sf" else "3d", a.ctx, a.qlen)
    out = run()
    torch.npu.synchronize()

    chk = ""
    if a.check:
        ref = reference(c)
        d = (out.float() - ref).abs().max().item()
        scale = ref.abs().max().item()
        chk = "  maxabs=%.4g rel=%.3g" % (d, d / max(scale, 1e-9))

    t_wall = wall(run)
    s = []
    for _ in range(15):
        busy()
        t0 = time.perf_counter()
        run()
        s.append((time.perf_counter() - t0) * 1e6)
        torch.npu.synchronize()
    s.sort()
    print("OK mode=%-5s ctx=%-3s qlen=%-3s D=%-3d n=%-3d kv=%-5d wall=%7.1fus issue_busy=%7.1fus%s"
          % (a.mode, a.ctx, a.qlen, a.D, a.n, a.kv, t_wall, s[len(s) // 2], chk))


if __name__ == "__main__":
    main()
