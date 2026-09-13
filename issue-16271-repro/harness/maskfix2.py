#!/usr/bin/env python3
"""#16271: buy the accuracy back with a device-side per-request mask.

Where round 1 of this idea died: the drafter's call is TND, and
`IsUsingFAI(...)` starts with `inputLayoutStr == "TND"`, so it is routed to the
split-fuse (FAI) template, whose `CheckFAIMask` says

    "When attnMask is provided, sparseMode must be 3 or 4"

and `CheckFAIMaskShape` additionally forces the last two mask dims to 2048 with
all leading dims 1. A per-request mask is categorically impossible there.

But the drafter's batch is uniform -- every request contributes exactly
num_speculative_tokens+1 query tokens -- so TND [T, N, D] is bit-for-bit the same
buffer as BSND [B, S, N, D]. Leaving TND leaves the FAI template, and the general
template's MaskChecker::ValidateMaskDimAndShape accepts

    sparse_mode 0, atten_mask [B or 1, >=Q_S, >=KV_S]

which is exactly the per-request tail cut we need:

    actual_seq_kvlen = U      upper bound, host numpy, no sync
    mask[i, j, c] = 1 iff c > L_i - Q + j     built on device from seq_lens

One arm per process: aclnn errors are asynchronous, and after a failure later
ops return without writing their output buffer, which silently produces a full
table of plausible numbers.

  python3 maskfix2.py --arm tnd_ref|tnd_c0|bsnd_ref|bsnd_mask_L|bsnd_mask_U
"""

import argparse
import sys
import time

import torch
import torch_npu  # noqa: F401

DTYPE = torch.bfloat16
INT_MAX = 2147483647
DEV = "npu"


def err(exc):
    s = str(exc)
    cut = s.find("Exception raised from")
    if cut > 0:
        s = s[:cut]
    t = " | ".join(x.strip() for x in s.strip().splitlines() if x.strip())
    return t if len(t) <= 1200 else t[:600] + " ...[cut]... " + t[-600:]


def build(n, kv_base, D, heads, kv_heads, block, Q, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    L = torch.randint(kv_base, kv_base + 257, (n,), generator=g, dtype=torch.int32)
    d = torch.randint(0, Q, (n,), generator=g, dtype=torch.int32)
    U = L + d
    bpr = int((U.max().item() + block - 1) // block)
    nb = n * bpr + 1
    kc4 = torch.randn(nb, block, kv_heads, D, dtype=DTYPE, device=DEV)
    vc4 = torch.randn(nb, block, kv_heads, D, dtype=DTYPE, device=DEV)
    q_tnd = torch.randn(n * Q, heads, D, dtype=DTYPE, device=DEV)
    return dict(
        n=n, D=D, nt=n * Q, bpr=bpr, nb=nb, heads=heads, kv_heads=kv_heads, block=block, Q=Q,
        S2=bpr * block,
        q_tnd=q_tnd,
        q_bsnd=q_tnd.view(n, Q, heads, D),  # same bytes, different label
        kc4=kc4, vc4=vc4,
        kc3=kc4.view(nb, block, kv_heads * D),
        vc3=vc4.view(nb, block, kv_heads * D),
        bt=torch.arange(1, n * bpr + 1, dtype=torch.int32, device=DEV).view(n, bpr),
        L=L, U=U, d=d, L_dev=L.to(DEV),
        cumq=[(i + 1) * Q for i in range(n)],
        flatq=[Q] * n,
        Lh=L.tolist(), Uh=U.tolist(),
        scale=D**-0.5,
        arange_q=torch.arange(Q, dtype=torch.int32, device=DEV),
        cols=torch.arange(bpr * block, dtype=torch.int32, device=DEV),
    )


def tri2048():
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(DEV)


def tail_mask(c, lens_dev, window=0):
    """[n, Q, S2] int8, 1 = masked out. Pure device arithmetic.

    With ``window`` > 0 the sliding window is folded in as well: production runs
    the drafter with sliding_window=2048, i.e. sparse_mode 4, and moving to
    sparse_mode 0 means the mask has to carry the band too, not just the tail
    cut. One mask expresses all three rules (causal, window, rolled-back tail).
    """
    limit = (lens_dev.to(torch.int32) - c["Q"]).view(-1, 1) + c["arange_q"].view(1, -1)  # [n, Q]
    cols = c["cols"].view(1, 1, -1)
    m = cols > limit.unsqueeze(-1)
    if window:
        # band mode keeps (limit - window, limit]; pre_tokens=window, next_tokens=0
        m = m | (cols <= (limit.unsqueeze(-1) - window))
    return m.to(torch.int8)


def fia_tnd(c, kvlens, sparse_mode, mask):
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=c["q_tnd"], key=c["kc3"], value=c["vc3"], atten_mask=mask,
        block_table=c["bt"], input_layout="TND", block_size=c["block"],
        actual_seq_lengths=c["cumq"], actual_seq_lengths_kv=kvlens,
        num_key_value_heads=c["kv_heads"], num_heads=c["heads"], scale=c["scale"],
        pre_tokens=INT_MAX, next_tokens=0, sparse_mode=sparse_mode,
    )
    return out.view(c["nt"], c["heads"], c["D"])


def fia_tnd_swa(c, kvlens, mask, window):
    """Production's drafter call: sliding window => sparse_mode 4 (band)."""
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=c["q_tnd"], key=c["kc3"], value=c["vc3"], atten_mask=mask,
        block_table=c["bt"], input_layout="TND", block_size=c["block"],
        actual_seq_lengths=c["cumq"], actual_seq_lengths_kv=kvlens,
        num_key_value_heads=c["kv_heads"], num_heads=c["heads"], scale=c["scale"],
        pre_tokens=window, next_tokens=0, sparse_mode=4,
    )
    return out.view(c["nt"], c["heads"], c["D"])


def fia_bsnd(c, kvlens, sparse_mode, mask):
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=c["q_bsnd"], key=c["kc3"], value=c["vc3"], atten_mask=mask,
        block_table=c["bt"], input_layout="BSND", block_size=c["block"],
        actual_seq_lengths=c["flatq"], actual_seq_lengths_kv=kvlens,
        num_key_value_heads=c["kv_heads"], num_heads=c["heads"], scale=c["scale"],
        pre_tokens=INT_MAX, next_tokens=0, sparse_mode=sparse_mode,
    )
    return out.reshape(c["nt"], c["heads"], c["D"])


def reference(c, window=0):
    n, Q, D, H, KVH = c["n"], c["Q"], c["D"], c["heads"], c["kv_heads"]
    group = H // KVH
    out = torch.empty(c["nt"], H, D, dtype=torch.float32, device=DEV)
    for r in range(n):
        blocks = c["bt"][r].tolist()
        L = int(c["L"][r])
        k = torch.cat([c["kc4"][b] for b in blocks], dim=0)[:L].float().repeat_interleave(group, dim=1)
        v = torch.cat([c["vc4"][b] for b in blocks], dim=0)[:L].float().repeat_interleave(group, dim=1)
        for j in range(Q):
            hi = L - Q + j + 1
            lo = max(0, hi - window) if window else 0
            s = torch.einsum("hd,lhd->hl", c["q_tnd"][r * Q + j].float(), k[lo:hi]) * c["scale"]
            out[r * Q + j] = torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v[lo:hi])
    return out


def wall(fn, iters=30):
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
    ap.add_argument("--arm", required=True,
                    choices=["tnd_ref", "tnd_c0", "bsnd_ref", "bsnd_mask_L", "bsnd_mask_U", "maskbuild",
                             "tnd_swa_ref", "tnd_swa_c0", "bsnd_swa_U", "swa_cmp"])
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--kv", type=int, default=1024)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--q", type=int, default=9)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    c = build(a.n, a.kv, a.D, a.heads, a.kv_heads, a.block, a.q, a.seed)

    if a.arm == "swa_cmp":
        # My torch reference disagreed with production's band mode by 0.157,
        # which says my reading of pre_tokens' boundary is wrong -- not
        # necessarily that the candidate is. So drop the torch reference here and
        # make FIA-with-exact-lengths the reference, then sweep the window offset
        # until the device mask reproduces it. If no offset does, band mode means
        # something other than a fixed-width band and this needs a different fix.
        tri = tri2048()
        try:
            base = fia_tnd_swa(c, c["Lh"], tri, a.window)
            torch.npu.synchronize()
        except Exception as exc:  # noqa: BLE001
            print("FAIL swa_cmp base :: %s" % err(exc))
            return 2
        scale = base.float().abs().max().item()
        # control: no window at all, to show the comparison can tell them apart
        try:
            nowin = fia_bsnd(c, c["Uh"], 0, tail_mask(c, c["L_dev"], 0))
            torch.npu.synchronize()
            print("    %-18s rel_vs_tnd_sm4=%.4g   (control: window ignored)"
                  % ("mask w/o window", (nowin.float() - base.float()).abs().max().item() / scale))
        except Exception as exc:  # noqa: BLE001
            print("    control failed :: %s" % err(exc))
            return 2
        for shift in (-2, -1, 0, 1, 2):
            w = a.window + shift
            try:
                o = fia_bsnd(c, c["Uh"], 0, tail_mask(c, c["L_dev"], w))
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print("    shift=%+d FAIL :: %s" % (shift, err(exc)))
                return 2
            print("    window=%d (shift %+d)  rel_vs_tnd_sm4=%.4g"
                  % (w, shift, (o.float() - base.float()).abs().max().item() / scale))
        return 0

    if a.arm == "maskbuild":
        f = lambda: tail_mask(c, c["L_dev"])  # noqa: E731
        m = f()
        torch.npu.synchronize()
        print("OK  %-12s n=%-3d D=%-3d S2=%-5d shape=%s  wall=%7.1fus"
              % (a.arm, a.n, a.D, c["S2"], tuple(m.shape), wall(f)))
        return 0

    try:
        # Hoisted: building it inside the timed lambda allocates a 2048x2048
        # fp32 CPU tensor and copies it H2D on every iteration, which read as a
        # 38 ms "kernel" and buried the comparison.
        tri = tri2048() if a.arm in ("tnd_ref", "tnd_c0", "bsnd_ref") else None
        if a.arm == "tnd_ref":
            run = lambda: fia_tnd(c, c["Lh"], 3, tri)                  # noqa: E731
        elif a.arm == "tnd_c0":
            run = lambda: fia_tnd(c, c["Uh"], 3, tri)                  # noqa: E731
        elif a.arm == "bsnd_ref":
            run = lambda: fia_bsnd(c, c["Lh"], 3, tri)                 # noqa: E731
        elif a.arm == "bsnd_mask_L":
            m = tail_mask(c, c["L_dev"])
            run = lambda: fia_bsnd(c, c["Lh"], 0, m)                   # noqa: E731
        elif a.arm == "tnd_swa_ref":
            tri = tri2048()
            run = lambda: fia_tnd_swa(c, c["Lh"], tri, a.window)       # noqa: E731
        elif a.arm == "tnd_swa_c0":
            tri = tri2048()
            run = lambda: fia_tnd_swa(c, c["Uh"], tri, a.window)       # noqa: E731
        elif a.arm == "bsnd_swa_U":
            # Measured: band mode with pre_tokens=W keeps W+1 positions,
            # [limit-W, limit] closed at both ends. The +1 is not cosmetic --
            # dropping it costs 0.157 relative error, 27x the bf16 floor.
            m = tail_mask(c, c["L_dev"], a.window + 1)
            run = lambda: fia_bsnd(c, c["Uh"], 0, m)                   # noqa: E731
        else:  # bsnd_mask_U -- the candidate: op told U, mask built from L
            m = tail_mask(c, c["L_dev"])
            run = lambda: fia_bsnd(c, c["Uh"], 0, m)                   # noqa: E731
        out = run()
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print("FAIL %-12s n=%-3d D=%-3d :: %s" % (a.arm, a.n, a.D, err(exc)))
        return 2

    ref = reference(c, a.window + 1 if "swa" in a.arm else 0)
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    print("OK  %-12s n=%-3d D=%-3d S2=%-5d  rel_vs_torch=%-10.4g wall=%7.1fus  (mean rejected d=%.2f)"
          % (a.arm, a.n, a.D, c["S2"], rel, wall(run), float(c["d"].float().mean())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
