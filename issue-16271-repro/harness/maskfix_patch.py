#!/usr/bin/env python3
"""#16271 on MRV2: keep the free host upper bound, mask the rolled-back tail on
the device. Apply after ``nt_patch.py --fix`` and ``approx_patch.py``.

The approx switch (`c0`) already removes the blocking ``seq_lens.tolist()`` by
feeding FIA the optimistic bound U instead of the exact length L. Its cost is
that the kernel then attends to [L, U) -- the draft KV this step rolled back --
which dilutes the softmax and costs ~6% acceptance.

``atten_mask`` is a plain device tensor and NOT a ValueDepend parameter, so it
may depend on the device-resident seq_lens. Masking [L, U) restores the exact
answer while L never reaches the host. Measured in scripts/maskfix2.py: identical
relative error to feeding the exact lengths (0.002919 vs 0.002919 at n=64 with
the sliding window), against 0.2749 for c0.

The one structural obstacle is template selection, not the interface:
``IsUsingFAI()`` starts with ``inputLayoutStr == "TND"``, so a TND call is routed
to the split-fuse template whose ``CheckFAIMask`` demands sparseMode 3 or 4 and a
2048x2048 mask -- a per-request mask is impossible there. But every request in a
parallel-drafting draft build contributes exactly num_speculative_tokens+1 query
tokens, so TND [T,N,D] is the same buffer as BSND [B,S,N,D]; switching the label
leaves that template and the general one accepts [B, >=Q_S, >=KV_S] with
sparseMode 0.

Gated on NT_MASKFIX=1. NT_MASKFIX_VERIFY=N re-runs the first N draft builds with
the exact lengths and reports the difference, because a wrong mask here does not
crash -- it quietly changes the acceptance rate, which is the same shape of
failure this investigation already hit twice.
"""

import argparse
import hashlib
import os
import shutil
import sys

ROOT = os.environ.get("NT_ASCEND_ROOT", "/vllm-workspace/vllm-ascend/vllm_ascend")
ATTN = os.path.join(ROOT, "attention", "attention_v1.py")
BACKUP = ATTN + ".nt-maskfix-orig"
MARK = "_nt_maskfix_mask"

ANCHOR = """            **backend_metadata,
        )
        if self.pcp_enabled:
"""

INSERT = """            **backend_metadata,
        )
        # ---- NT MASKFIX (#16271) ----
        # Mark the build whose ``seq_lens_list`` is the optimistic upper bound
        # rather than the truth, and hand the impl what it needs to mask the
        # difference away on the device. ``seq_lens`` is already the exact
        # device tensor on this branch; nothing here reads it.
        attn_metadata.nt_maskfix = bool(
            _NT_MASKFIX
            and getattr(common_attn_metadata, "seq_lens_cpu_is_approximate", False)
            and not getattr(common_attn_metadata, "seq_lens_cpu_is_exact", False)
        )
        # Host-side query lengths: free, and BSND needs per-request lengths
        # rather than the cumulative form TND takes.
        attn_metadata.nt_maskfix_qlens = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).tolist()
        # One mask per step, shared by every layer in the group.
        attn_metadata.nt_maskfix_cache = {}
        # ---- end NT MASKFIX ----
        if self.pcp_enabled:
"""

HELPER = '''

# ---- NT MASKFIX (issue #16271): device-side tail mask ----
import os as _nt_os  # noqa: E402

_NT_MASKFIX = _nt_os.environ.get("NT_MASKFIX", "") == "1"
_NT_MASKFIX_EVERY = int(_nt_os.environ.get("NT_MASKFIX_EVERY", "500") or 500)
_NT_MASKFIX_VERIFY = int(_nt_os.environ.get("NT_MASKFIX_VERIFY", "0") or 0)
_nt_mf_stats = {
    "calls": 0, "hit": 0, "not_marked": 0, "nonuniform_q": 0,
    "shape": 0, "sinks": 0, "noncausal": 0, "capturing": 0,
    "verify_n": 0, "verify_max_rel": 0.0,
}


def _nt_maskfix_mask(attn_metadata, impl, s2, device):
    """[B, Q, S2] int8 mask, 1 = masked out, built only from device tensors.

    Query token j of request i may attend kv positions <= L_i - Q + j: that is
    the causal rule inside the draft block AND the cut that hides the rolled
    back tail [L_i, U_i), since the op is told U_i.

    The sliding window boundary is not a guess: band mode with pre_tokens=W was
    measured to keep W+1 positions, [hi-W, hi] closed at both ends. Writing the
    obvious `< hi - W + 1` instead costs 0.157 relative error -- 27x the bf16
    floor, and nowhere near large enough to crash.
    """
    q = int(attn_metadata.nt_maskfix_qlens[0])
    win = impl.sliding_window if impl.sliding_window else 0
    ckey = (s2, q, win)
    cached = attn_metadata.nt_maskfix_cache.get(ckey)
    if cached is not None:
        return cached
    lens = attn_metadata.seq_lens.to(torch.int32)            # device, exact
    ar = torch.arange(q, device=device, dtype=torch.int32)
    cols = torch.arange(s2, device=device, dtype=torch.int32).view(1, 1, -1)
    hi = ((lens - q).view(-1, 1) + ar.view(1, -1)).unsqueeze(-1)   # [B, Q, 1]
    m = cols > hi
    if win:
        m = m | (cols < (hi - win))
    m = m.to(torch.int8)
    attn_metadata.nt_maskfix_cache[ckey] = m
    return m


_nt_mf_orig_ffia = AscendAttentionBackendImpl.forward_fused_infer_attention


def _nt_mf_ffia(self, query, key, value, attn_metadata, output, kv_cache=None):
    """BSND + sparse_mode 0 + per-request device mask, or fall through.

    Every fall-through is counted rather than silently taken: an arm that never
    fires is indistinguishable from the do-nothing arm by its numbers alone.
    """
    s = _nt_mf_stats
    if not getattr(attn_metadata, "nt_maskfix", False):
        s["not_marked"] += 1
        return _nt_mf_orig_ffia(self, query, key, value, attn_metadata, output, kv_cache)
    s["calls"] += 1
    if _EXTRA_CTX.capturing:
        s["capturing"] += 1
        return _nt_mf_orig_ffia(self, query, key, value, attn_metadata, output, kv_cache)
    if self.sinks is not None:
        s["sinks"] += 1
        return _nt_mf_orig_ffia(self, query, key, value, attn_metadata, output, kv_cache)
    if not attn_metadata.causal:
        s["noncausal"] += 1
        return _nt_mf_orig_ffia(self, query, key, value, attn_metadata, output, kv_cache)
    qlens = attn_metadata.nt_maskfix_qlens
    b = len(qlens)
    q = int(qlens[0]) if b else 0
    if q < 2 or any(int(x) != q for x in qlens):
        # BSND needs a rectangular batch. A padded/ragged draft build is rare
        # but must not be silently reshaped.
        s["nonuniform_q"] += 1
        return _nt_mf_orig_ffia(self, query, key, value, attn_metadata, output, kv_cache)

    key_c, value_c, block_size, block_table, kvlens = self._get_fia_params(key, value, attn_metadata, kv_cache)
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    if block_table is None or num_tokens != b * q or attn_metadata.seq_lens.shape[0] < b:
        s["shape"] += 1
        return _nt_mf_orig_ffia(self, query, key, value, attn_metadata, output, kv_cache)

    s2 = int(block_table.shape[1]) * int(block_size)
    mask = _nt_maskfix_mask(attn_metadata, self, s2, query.device)
    qb = query[:num_tokens].view(b, q, self.num_heads, self.head_size)
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=qb,
        key=key_c,
        value=value_c,
        atten_mask=mask,
        block_table=block_table,
        input_layout="BSND",
        block_size=block_size,
        actual_seq_lengths=[q] * b,
        actual_seq_lengths_kv=kvlens,
        num_key_value_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=0,
    )
    attn_output = attn_output.reshape(num_tokens, self.num_heads, self.head_size)

    if _NT_MASKFIX_VERIFY and s["verify_n"] < _NT_MASKFIX_VERIFY:
        # Golden reference on the real model: the exact lengths, the way the
        # unpatched path would have used them. Costs a blocking D2H, so only for
        # the first few builds.
        s["verify_n"] += 1
        exact = attn_metadata.seq_lens[:b].tolist()
        ref, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:num_tokens],
            key=key_c,
            value=value_c,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=exact,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            pre_tokens=self.sliding_window if self.sliding_window else 2147483647,
            next_tokens=0,
            sparse_mode=4 if self.sliding_window else 3,
        )
        ref = ref.view(num_tokens, self.num_heads, self.head_size).float()
        denom = ref.abs().max()
        rel = ((attn_output.float() - ref).abs().max() / denom.clamp_min(1e-9)).item()
        s["verify_max_rel"] = max(s["verify_max_rel"], rel)
        print("[NT_MASKFIX] verify %d/%d b=%d q=%d s2=%d win=%s rel=%.5g (running max %.5g)"
              % (s["verify_n"], _NT_MASKFIX_VERIFY, b, q, s2, self.sliding_window, rel,
                 s["verify_max_rel"]), flush=True)

    output[:num_tokens] = attn_output
    s["hit"] += 1
    if _NT_MASKFIX_EVERY and s["hit"] % _NT_MASKFIX_EVERY == 0:
        print("[NT_MASKFIX] " + " ".join("%s=%s" % kv for kv in sorted(s.items())), flush=True)
    return output


AscendAttentionBackendImpl.forward_fused_infer_attention = _nt_mf_ffia
# ---- end NT MASKFIX ----
'''


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.status:
        src = open(ATTN, encoding="utf-8").read()
        print("%s md5=%s maskfix=%s" % (ATTN, _md5(ATTN), MARK in src))
        return 0

    if a.revert:
        # nt_patch.py --revert restores its own backup, which was taken before
        # --fix; reverting through it would also undo the fix. Keep an
        # independent one so the two can be unwound in either order.
        if os.path.exists(BACKUP):
            shutil.copy2(BACKUP, ATTN)
            print("MASKFIX REVERTED %s md5=%s" % (ATTN, _md5(ATTN)))
        else:
            print("MASKFIX NO BACKUP (nothing to revert)")
        return 0

    src = open(ATTN, encoding="utf-8").read()
    if MARK in src:
        print("SKIP already applied")
        return 0
    if "seq_lens_cpu_is_approximate" not in src:
        print("FAIL approx_patch.py must be applied first (no seq_lens_cpu_is_approximate)")
        return 1
    n = src.count(ANCHOR)
    if n != 1:
        print("FAIL builder anchor count=%d (expected 1)" % n)
        return 1
    if "class AscendAttentionBackendImpl" not in src:
        print("FAIL AscendAttentionBackendImpl not found")
        return 1
    if not os.path.exists(BACKUP):
        shutil.copy2(ATTN, BACKUP)
    out = src.replace(ANCHOR, INSERT, 1) + HELPER
    with open(ATTN, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("MASKFIX APPLIED %s md5=%s" % (ATTN, _md5(ATTN)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
