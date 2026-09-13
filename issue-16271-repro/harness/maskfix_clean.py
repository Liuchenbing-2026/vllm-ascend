#!/usr/bin/env python3
"""issue #16271, MRV2: exact draft attention without the blocking D2H.

This is the tidied form of scripts/maskfix_patch.py -- the same change, but as
real edits (dataclass fields, a module-level helper, a method on the impl and one
call site) instead of a monkeypatch, so the result can be diffed and reviewed.

Applies on top of:
  * fix/16271-dspark-attn-seq-lens-host-sync (adfa304e) -- exact host mirror for
    the target build;
  * feat/16271-approx-draft-kv-opt-in (367a4673) -- ``seq_lens_cpu_is_approximate``
    and the optimistic bound for the draft build.

Run ``nt_patch.py --fix`` then ``approx_patch.py`` first.
"""

import argparse
import hashlib
import os
import shutil
import sys

ROOT = os.environ.get("NT_ASCEND_ROOT", "/vllm-workspace/vllm-ascend/vllm_ascend")
ENVS = os.path.join(ROOT, "envs.py")
ATTN = os.path.join(ROOT, "attention", "attention_v1.py")
BACKUPS = {ENVS: ENVS + ".nt-clean-orig", ATTN: ATTN + ".nt-clean-orig"}
MARK = "build_draft_tail_mask"

# The branch declares env flags through ``_strict_binary_env``; the in-image
# tree is ~54 commits behind and predates that helper, so the same edit has two
# possible anchors. Whichever is present is the one to extend -- matching the
# surrounding style matters more than picking one form.
_ENVS_DOC = (
    "    # Restore exactness on top of the approximate bound above by masking the\n"
    "    # rolled-back KV tail on the device: the draft build is handed the\n"
    "    # optimistic bound as before, and ``atten_mask`` -- an ordinary device\n"
    "    # tensor, not a ValueDepend operator input -- hides the [L, U) tail, which\n"
    "    # is the draft KV the previous step rolled back. That tail is the whole of\n"
    "    # the approximation's cost, so masking it restores the exact result while\n"
    "    # the host still never learns the true lengths.\n"
    "    # Requires VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV=1; default 0.\n"
    "    # See https://github.com/vllm-project/vllm-ascend/issues/16271\n"
)
_ENVS_VARIANTS = [
    (
        '    "VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV": lambda: _strict_binary_env("VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV"),\n',
        _ENVS_DOC
        + '    "VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK": lambda: _strict_binary_env(\n'
        '        "VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK"\n'
        "    ),\n",
    ),
    (
        '    "VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV": lambda: os.getenv("VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV", "0") == "1",\n',
        _ENVS_DOC
        + '    "VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK": lambda: os.getenv(\n'
        '        "VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK", "0"\n'
        '    ) == "1",\n',
    ),
]


def _envs_edit():
    src = open(ENVS, encoding="utf-8").read()
    for old, added in _ENVS_VARIANTS:
        if src.count(old) == 1:
            return (ENVS, old, old + added)
    return (ENVS, "<<no known VLLM_ASCEND_DSPARK_APPROX_DRAFT_KV declaration>>", "")


EDITS = [
    _envs_edit(),
    # ------------------------------------------------------------ attn import
    (
        # attention_v1.py imports vLLM's envs but not vllm-ascend's. The runtime
        # A/B never caught this because it went through a monkeypatch that read
        # os.environ directly; only running the existing unit tests did.
        ATTN,
        "from vllm_ascend.ascend_forward_context import _EXTRA_CTX\n",
        "import vllm_ascend.envs as envs_ascend\nfrom vllm_ascend.ascend_forward_context import _EXTRA_CTX\n",
    ),
    # -------------------------------------------------- AscendMetadata fields
    (
        ATTN,
        "    seq_lens_list: list[int] = None  # type: ignore\n",
        "    seq_lens_list: list[int] = None  # type: ignore\n\n"
        "    # issue #16271. Set when ``seq_lens_list`` is the *optimistic upper\n"
        "    # bound* rather than the truth: a parallel-drafting draft build whose\n"
        "    # rejection count is resolved on the device. ``seq_lens`` still holds\n"
        "    # the exact lengths, on the device, and the attention impl masks the\n"
        "    # difference away instead of reading them back.\n"
        "    draft_kv_upper_bound: bool = False\n"
        "    # Per-request query lengths (BSND takes these; TND takes the\n"
        "    # cumulative form). Host-side and free: the scheduler set them.\n"
        "    draft_query_lens: list[int] = None  # type: ignore\n"
        "    # One mask per step, shared by every layer in the attention group.\n"
        "    draft_tail_mask_cache: dict = None  # type: ignore\n",
    ),
    # ------------------------------------------------------- mask constructor
    (
        ATTN,
        "class AscendAttentionBackendImpl(AttentionImpl):\n",
        'def build_draft_tail_mask(\n'
        "    seq_lens: torch.Tensor,\n"
        "    num_reqs: int,\n"
        "    query_len: int,\n"
        "    kv_span: int,\n"
        "    sliding_window: int | None,\n"
        ") -> torch.Tensor:\n"
        '    """Per-request attention mask for a draft build, built on the device.\n'
        "\n"
        "    ``seq_lens`` is the exact device-side KV length L per request; the op is\n"
        "    told the optimistic bound U >= L instead, which costs no host sync. Query\n"
        "    token j of request i may attend KV positions up to ``L_i - query_len + j``,\n"
        "    so masking beyond that reproduces exactly what passing L would have\n"
        "    computed -- and hides the [L_i, U_i) tail, which holds the draft KV this\n"
        "    step rolled back. Feeding U without this mask is what costs the approximate\n"
        "    path its acceptance.\n"
        "\n"
        "    ``atten_mask`` is an ordinary device tensor and is not declared\n"
        "    ``ValueDepend`` in the operator's IR, so unlike ``actual_seq_lengths_kv``\n"
        "    it may legally depend on device-resident values.\n"
        "\n"
        "    The sliding window is folded into the same mask. Band mode's\n"
        "    ``pre_tokens=W`` keeps W+1 positions -- ``[last - W, last]``, closed at\n"
        "    both ends -- so the comparison here is ``< last - W`` and not\n"
        "    ``<= last - W``. Getting that boundary wrong does not fail loudly; it\n"
        "    quietly changes the scores.\n"
        "\n"
        "    Returns an int8 [num_reqs, query_len, kv_span] tensor, 1 = masked out.\n"
        '    """\n'
        "    device = seq_lens.device\n"
        "    lens = seq_lens[:num_reqs].to(torch.int32)\n"
        "    offsets = torch.arange(query_len, device=device, dtype=torch.int32)\n"
        "    cols = torch.arange(kv_span, device=device, dtype=torch.int32).view(1, 1, -1)\n"
        "    # [num_reqs, query_len, 1]: the last KV position each query token may see.\n"
        "    last = ((lens - query_len).view(-1, 1) + offsets.view(1, -1)).unsqueeze(-1)\n"
        "    mask = cols > last\n"
        "    if sliding_window:\n"
        "        mask |= cols < (last - sliding_window)\n"
        "    return mask.to(torch.int8)\n"
        "\n"
        "\n"
        "class AscendAttentionBackendImpl(AttentionImpl):\n",
    ),
    # ------------------------------------------------------- builder: publish
    (
        ATTN,
        "        backend_metadata = self._build_backend_metadata(\n",
        "        # issue #16271: a draft build under the approximate bound keeps the\n"
        "        # exact lengths on the device; record enough for the impl to mask the\n"
        "        # difference rather than read them back.\n"
        "        draft_kv_upper_bound = bool(\n"
        "            envs_ascend.VLLM_ASCEND_DSPARK_DRAFT_KV_DEVICE_MASK\n"
        "            and common_attn_metadata.seq_lens_cpu_is_approximate\n"
        "            and not common_attn_metadata.seq_lens_cpu_is_exact\n"
        "        )\n"
        "        draft_query_lens = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).tolist()\n"
        "\n"
        "        backend_metadata = self._build_backend_metadata(\n",
    ),
    (
        ATTN,
        "            actual_seq_lengths_q=actual_seq_lengths_q,\n            slot_mapping=slot_mapping,\n",
        "            actual_seq_lengths_q=actual_seq_lengths_q,\n"
        "            draft_kv_upper_bound=draft_kv_upper_bound,\n"
        "            draft_query_lens=draft_query_lens,\n"
        "            draft_tail_mask_cache={},\n"
        "            slot_mapping=slot_mapping,\n",
    ),
    # ------------------------------------------------------------ impl: method
    (
        ATTN,
        "    def forward_fused_infer_attention(\n",
        "    def _forward_draft_tail_masked(\n"
        "        self,\n"
        "        query: torch.Tensor,\n"
        "        key: torch.Tensor,\n"
        "        value: torch.Tensor,\n"
        "        attn_metadata: AscendMetadata,\n"
        "        block_table: torch.Tensor,\n"
        "        block_size: int,\n"
        "        actual_seq_lengths_kv: list[int],\n"
        "        num_tokens: int,\n"
        "        output: torch.Tensor,\n"
        "    ) -> torch.Tensor | None:\n"
        '        """Exact draft attention from an optimistic KV bound. See #16271.\n'
        "\n"
        "        Returns ``None`` when this is not an eligible draft build, so the\n"
        "        caller falls through to the normal path.\n"
        "\n"
        "        The layout switch is not cosmetic. ``IsUsingFAI`` in the operator's\n"
        "        tiling begins with ``inputLayout == \"TND\"``, so a TND call is routed to\n"
        "        the split-fuse template, whose mask check accepts only sparse_mode 3/4\n"
        "        with a 2048x2048 mask -- a per-request mask is impossible there. Every\n"
        "        request in a parallel-drafting draft build contributes the same number\n"
        "        of query tokens, so TND [T, N, D] is the same buffer as\n"
        "        BSND [B, S, N, D]; relabelling it leaves that template for the general\n"
        "        one, which accepts [B, >=Q_S, >=KV_S] with sparse_mode 0.\n"
        '        """\n'
        "        if not attn_metadata.draft_kv_upper_bound:\n"
        "            return None\n"
        "        # A learnable sink or a non-causal build changes what the mask would\n"
        "        # have to express; neither occurs on this path today.\n"
        "        if self.sinks is not None or not attn_metadata.causal or block_table is None:\n"
        "            return None\n"
        "        query_lens = attn_metadata.draft_query_lens\n"
        "        num_reqs = len(query_lens) if query_lens else 0\n"
        "        query_len = int(query_lens[0]) if num_reqs else 0\n"
        "        # BSND needs a rectangular batch. Parallel drafting always produces\n"
        "        # one, but a padded or ragged build must not be silently reshaped.\n"
        "        if query_len < 2 or any(int(q) != query_len for q in query_lens):\n"
        "            return None\n"
        "        if num_tokens != num_reqs * query_len or attn_metadata.seq_lens.shape[0] < num_reqs:\n"
        "            return None\n"
        "\n"
        "        # Under paged attention the mask's last dimension must cover the whole\n"
        "        # addressable KV span, not just the longest actual sequence.\n"
        "        kv_span = int(block_table.shape[1]) * int(block_size)\n"
        "        cache_key = (kv_span, query_len, self.sliding_window)\n"
        "        mask = attn_metadata.draft_tail_mask_cache.get(cache_key)\n"
        "        if mask is None:\n"
        "            mask = build_draft_tail_mask(\n"
        "                attn_metadata.seq_lens, num_reqs, query_len, kv_span, self.sliding_window\n"
        "            )\n"
        "            attn_metadata.draft_tail_mask_cache[cache_key] = mask\n"
        "\n"
        "        attn_output, _ = torch_npu.npu_fused_infer_attention_score(\n"
        "            query=query.view(num_reqs, query_len, self.num_heads, self.head_size),\n"
        "            key=key,\n"
        "            value=value,\n"
        "            atten_mask=mask,\n"
        "            block_table=block_table,\n"
        '            input_layout="BSND",\n'
        "            block_size=block_size,\n"
        "            # BSND takes per-request query lengths, not the cumulative form.\n"
        "            actual_seq_lengths=[query_len] * num_reqs,\n"
        "            actual_seq_lengths_kv=actual_seq_lengths_kv,\n"
        "            num_key_value_heads=self.num_kv_heads,\n"
        "            num_heads=self.num_heads,\n"
        "            scale=self.scale,\n"
        "            sparse_mode=0,\n"
        "        )\n"
        "        output[:num_tokens] = attn_output.reshape(num_tokens, self.num_heads, self.head_size)\n"
        "        return output\n"
        "\n"
        "    def forward_fused_infer_attention(\n",
    ),
    # --------------------------------------------------------- impl: call site
    (
        ATTN,
        "        num_tokens = attn_metadata.actual_seq_lengths_q[-1]\n        query = query[:num_tokens]\n",
        "        num_tokens = attn_metadata.actual_seq_lengths_q[-1]\n"
        "        query = query[:num_tokens]\n"
        "        draft_output = self._forward_draft_tail_masked(\n"
        "            query, key, value, attn_metadata, block_table, block_size,\n"
        "            actual_seq_lengths_kv, num_tokens, output,\n"
        "        )\n"
        "        if draft_output is not None:\n"
        "            return draft_output\n",
    ),
]


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if a.revert:
        for path, backup in BACKUPS.items():
            if os.path.exists(backup):
                shutil.copy2(backup, path)
                print("CLEAN REVERTED %s md5=%s" % (os.path.basename(path), _md5(path)))
        return 0

    src_attn = open(ATTN, encoding="utf-8").read()
    if MARK in src_attn:
        print("SKIP already applied")
        return 0
    if "seq_lens_cpu_is_approximate" not in src_attn:
        print("FAIL approx_patch.py must be applied first")
        return 1

    # Check every anchor before writing anything: a half-applied file is worse
    # than a rejected patch.
    bad = 0
    for path, old, _new in EDITS:
        n = open(path, encoding="utf-8").read().count(old)
        if n != 1:
            print("FAIL anchor count=%d (expected 1) in %s: %r" % (n, os.path.basename(path), old[:70]))
            bad += 1
    if bad:
        return 1

    for path, backup in BACKUPS.items():
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
    for path, old, new in EDITS:
        src = open(path, encoding="utf-8").read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src.replace(old, new, 1))
    for path in BACKUPS:
        print("CLEAN APPLIED %s md5=%s" % (os.path.basename(path), _md5(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
