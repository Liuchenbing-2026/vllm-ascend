"""Harvest bf16-tie mismatch samples between AscendC sgmv_expand and the
current v2 order, with enough raw data (x 16-vec, w 16-col, y0, both outputs)
to replay ANY accumulation order on the host."""
import json

import torch
import torch_npu  # noqa
import vllm_ascend  # noqa
import vllm_ascend.vllm_ascend_C  # noqa

DEV, DT, R, L = "npu", torch.bfloat16, 16, 2
CFGS = [(17408, 34816, 0, "gate"), (17408, 34816, 17408, "up"),
        (6144, 8192, 0, "q"), (5120, 5120, 0, "o")]
samples = []
for scale in (0.05, 1.0):
    for Ho, YHO, OFF, lab in CFGS:
        B = 256
        idx = torch.zeros(1, device=DEV, dtype=torch.int64)
        seq = torch.full((1,), B, device=DEV, dtype=torch.int64)
        torch.manual_seed(hash((lab, scale)) % 2**31)
        xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * scale
        we = torch.randn(L, Ho, R, device=DEV, dtype=DT) * scale
        y0 = torch.randn(B, YHO, device=DEV, dtype=DT) * scale
        ya = y0.clone()
        torch.ops._C_ascend.sgmv_expand(xe, we, idx, seq, ya, OFF, Ho)
        torch.npu.synchronize()
        # v2 current order on host is irrelevant; find ties by comparing
        # ascendc against a HOST fp32 tree (tl.sum-like fold) rounded to bf16:
        # any element where multiple association orders straddle the boundary
        # is interesting, so just grab elements where ascendc != fold-order.
        prod = xe @ we[0].float().t()                     # [B, Ho] fp32 (host order)
        cand = (y0.float()[:, OFF:OFF + Ho] + prod).to(DT)
        sl = ya[:, OFF:OFF + Ho]
        bad = (sl != cand).nonzero()
        for p in bad[:40]:
            b_, h_ = int(p[0]), int(p[1])
            samples.append({
                "lab": lab, "scale": scale, "b": b_, "h": h_,
                "x": [float(v) for v in xe[b_]],
                "w": [float(v.float()) for v in we[0, h_]],
                "y0": float(y0[b_, OFF + h_].float()),
                "y0_bits": int(y0[b_, OFF + h_].view(torch.int16)),
                "ref_bits": int(sl[b_, h_].view(torch.int16)),
                "ref": float(sl[b_, h_].float()),
            })
        print("%s scale=%s: %d candidate ties" % (lab, scale, len(bad)), flush=True)
json.dump(samples, open("/work/tie_samples.json", "w"))
print("WROTE /work/tie_samples.json n=%d" % len(samples), flush=True)
