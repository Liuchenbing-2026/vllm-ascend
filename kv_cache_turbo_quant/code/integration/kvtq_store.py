"""Real-storage TurboQuant KV cache for vllm-ascend (prototype v1).

Unlike shadow mode (quantize + discard, cache stays BF16), store mode:
  - shrinks the paged KV cache allocation to the packed TurboQuant format
    (per token per head: mse_bits-packed indices + bf16 norm), giving ~3.9x
    more tokens for the same memory (bits=4: 66/130 B vs 256/512 B for
    head_dim 128/256);
  - writes real quantized data in reshape_and_cache via the custom
    aclnn KvCacheTurboQuant op;
  - reads by staged dequant: gather packed blocks -> unpack -> centroid
    lookup (rotated space) -> dense BF16 -> standard FIA TND attention.
    The rotation is absorbed query/output-side (q_rot = q @ R.T,
    out = out_rot @ R), so no per-token dequant matmul is needed.
  - QJL residual correction is NOT applied at read time in v1 (qjl/gamma
    bytes are not even stored). Accuracy therefore equals pure MSE
    quantization at mse_bits; bits=4 recommended.

Requirements: --enforce-eager (staged read has dynamic shapes),
--no-enable-prefix-caching, and max_num_batched_tokens >= max_model_len
(prefill must be single-chunk; a per-seq dequant+concat fallback exists but
is slow). Enable with VLLM_ASCEND_KVTQ_STORE=1.

INT8 mode (VLLM_ASCEND_KVTQ_STORE=int8): per-channel dynamic int8 KV cache
using the native FIA paged-attention antiquant path (NZ 5D view, mode 0).
Decode reads the compressed cache directly (no dense materialization), so
long-context decode gets a real bandwidth win (~2.5x attention speedup at
B>=8/128k in kernel microbench). Capacity gain is 2x. Per-channel scales
are calibrated online from the first prefill write and frozen (saturating
later outliers); prefill computes on in-flight dense K/V, only the cache
is quantized. Pure-GQA models only (hybrid page alignment not handled).
"""
import math
import os

import torch

_MODE = os.environ.get("VLLM_ASCEND_KVTQ_STORE", "0")
_ENABLED = _MODE in ("1", "int8")
_INT8 = _MODE == "int8"
_MSE_BITS = int(os.environ.get("VLLM_ASCEND_KVTQ_BITS", "4"))


def _idx_bytes(head_dim):
    return head_dim * _MSE_BITS // 8


def _row_bytes(head_dim, hybrid=False):
    raw = _idx_bytes(head_dim) + 2  # + bf16 norm; qjl/gamma not stored in v1
    if not hybrid:
        return raw
    # Hybrid (GDN/mamba) models on Ascend: vllm-ascend forces
    # attention page == mamba page via patch_mamba_config, requiring the packed
    # bf16 "head size" h to satisfy h * kv_heads * 2 | ssm_page (h | 3*2**10 for
    # the Qwen3.5/3.6/3.8 family). Pad the row to h=96 bf16 (192 B): keeps 2.67x
    # compression for bits<=4 at head_dim=256 instead of losing everything to
    # page padding.
    hs = (raw + 1) // 2
    if hs < 96:
        return 192
    return raw


def _is_hybrid_config(vllm_config):
    try:
        tc = vllm_config.model_config.hf_text_config
        lt = getattr(tc, "layer_types", None)
        if lt and any(t != "full_attention" for t in lt):
            return True
        return getattr(tc, "full_attention_interval", None) is not None
    except Exception:
        return False

CENTROIDS = {
    2: [-0.1335033178, -0.04002048075, 0.04002048075, 0.1335033178],
    3: [-0.19020693, -0.1187859178, -0.06682205945, -0.02166347019,
        0.02166347019, 0.06682205945, 0.1187859178, 0.19020693],
    4: [-0.2414890379, -0.1828317791, -0.1429702938, -0.1109927073,
        -0.08325428516, -0.05802082643, -0.03428063914, -0.01134236995,
        0.01134236995, 0.03428063914, 0.05802082643, 0.08325428516,
        0.1109927073, 0.1429702938, 0.1828317791, 0.2414890379],
}

_cache = {}
stats = {"quant_calls": 0, "quant_tokens": 0, "decode_calls": 0, "warned": set()}


def _get_consts(device, head_dim):
    key = (str(device), head_dim)
    if key not in _cache:
        gen = torch.Generator().manual_seed(20260929)
        q, _ = torch.linalg.qr(
            torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float64))
        rot = q.to(torch.float32).to(device).contiguous()
        cent = torch.tensor(CENTROIDS[_MSE_BITS], dtype=torch.bfloat16, device=device)
        _cache[key] = (rot, rot.T.contiguous(), cent)
    return _cache[key]


def _warn_once(tag, msg):
    if tag not in stats["warned"]:
        stats["warned"].add(tag)
        print(f"[KVTQ-STORE] WARNING {msg}", flush=True)


def _pack_rows(idx, norm, row_bytes=None):
    norm_b = norm.unsqueeze(-1).view(torch.uint8)  # [N,H,2]
    packed = torch.cat([idx, norm_b], dim=-1)  # [N,H,idx_bytes+2] uint8
    if row_bytes is not None and packed.shape[-1] < row_bytes:
        packed = torch.nn.functional.pad(packed, (0, row_bytes - packed.shape[-1]))
    return packed


def _get_byte_lut(device):
    """bits=4 fast path: byte -> (centroid[lo_nibble], centroid[hi_nibble]).

    Golden packing interleaves 4-bit lanes into bytes: byte j of a group holds
    lane 2j in its low nibble and lane 2j+1 in its high nibble, so a single
    256x2 lookup table replaces the whole bit-unpack + centroid gather.
    """
    key = ("lut", str(device))
    if key not in _cache:
        cent = torch.tensor(CENTROIDS[4], dtype=torch.bfloat16, device=device)
        b = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8)
        lut = torch.stack([cent[(b & 15).long()], cent[(b >> 4).long()]], dim=-1)
        _cache[key] = lut.contiguous()
    return _cache[key]


def _unpack_indices(packed, bit_width):
    groups = packed.shape[-1] // bit_width
    words = packed.reshape(*packed.shape[:-1], groups, bit_width).to(torch.int64)
    word = torch.zeros_like(words[..., 0])
    for byte in range(bit_width):
        word |= words[..., byte] << (8 * byte)
    lanes = [(word >> (lane * bit_width)) & ((1 << bit_width) - 1) for lane in range(8)]
    return torch.stack(lanes, dim=-1).reshape(*packed.shape[:-1], groups * 8)


def _bf16_from_bytes(lo, hi):
    bits = (lo.to(torch.int64) | (hi.to(torch.int64) << 8)) << 16
    return bits.to(torch.int32).view(torch.float32)


def _write_cache(self, tensor, cache_bf16, slots):
    x = tensor.contiguous()
    head_dim = x.shape[-1]
    rot32, qjl32 = _get_quant_matrices(x.device, head_dim)
    idx, _qjlb, norm, _gamma = torch.ops.turboquant.kv_cache_turbo_quant(
        x, rot32, qjl32, _MSE_BITS)
    row_bytes = cache_bf16.shape[-1] * 2  # self-describing layout (padded for hybrid)
    packed = _pack_rows(idx, norm, row_bytes)
    num_kv_heads = cache_bf16.shape[2]
    flat = cache_bf16.view(torch.uint8).view(-1, num_kv_heads, row_bytes)
    flat.index_copy_(0, slots, packed)
    stats["quant_calls"] += 1
    stats["quant_tokens"] += x.shape[0]
    if stats["quant_calls"] == 1 or stats["quant_calls"] % 1000 == 0:
        print(f"[KVTQ-STORE] pid={os.getpid()} quant_calls={stats['quant_calls']} "
              f"tokens={stats['quant_tokens']}", flush=True)


def _get_quant_matrices(device, head_dim):
    rot, _, _ = _get_consts(device, head_dim)
    key = ("quant", str(device), head_dim)
    if key not in _cache:
        gen = torch.Generator().manual_seed(20260929)
        _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float64))
        qjl = (torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float64)
               .to(torch.float32) / math.sqrt(head_dim)).to(device).contiguous()
        _cache[key] = qjl
    return rot, _cache[key]


def _tq_reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) > 1:
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output
        num = min(attn_metadata.num_actual_tokens, key.shape[0])
        if num > 0:
            slots = attn_metadata.slot_mapping[:num].long()
            _write_cache(self, key[:num], self.key_cache, slots)
            _write_cache(self, value[:num], self.value_cache, slots)
            from vllm_ascend.attention.attention_v1 import notify_kv_cache_written
            notify_kv_cache_written()
    return query, key, value, output


# ---------------- int8 mode (native FIA paged antiquant) ----------------

_I8_FUSED_ENV = os.environ.get("VLLM_ASCEND_KVTQ_INT8_FUSED", "1") == "1"
_i8_fused_state = {"checked": False, "ok": False}


def _i8_fused_ok():
    """Fused single-kernel int8 quantize+NZ-scatter (kvtq_int8_store). Falls
    back to the PyTorch op chain when the op or env opt-out is unavailable."""
    if not _i8_fused_state["checked"]:
        ok = _I8_FUSED_ENV
        if ok:
            try:
                _ = torch.ops.turboquant.kvtq_int8_store
            except (AttributeError, RuntimeError) as exc:
                print("[KVTQ-INT8] fused op unavailable (%s), torch fallback" % exc,
                      flush=True)
                ok = False
        _i8_fused_state["checked"] = True
        _i8_fused_state["ok"] = ok
    return _i8_fused_state["ok"]


def _int8_scale(self, x, which, num_kv_heads, head_dim):
    """Per-(kv_head, channel) affine int8 scale. Calibrated from the first
    write (a full prefill, representative) and frozen; later outliers clamp."""
    scale = getattr(self, "_i8_%s_scale" % which, None)
    if scale is None:
        scale = (x.float().abs().amax(dim=0) / 127.0).clamp_min(1e-8)  # (kvh, d)
        setattr(self, "_i8_%s_scale" % which, scale)
        setattr(self, "_i8_%s_aq" % which,
                scale.to(torch.bfloat16).view(num_kv_heads, 1, head_dim).contiguous())
        print("[KVTQ-INT8] pid=%d %s scale frozen, amax mean %.3f"
              % (os.getpid(), which, scale.mean().item() * 127), flush=True)
    return scale


def _int8_write(self, x, cache, slots, which):
    x = x.contiguous()
    nb, bs, kvh, d = cache.shape
    scale = _int8_scale(self, x, which, kvh, d)
    if _i8_fused_ok():
        s32 = slots if slots.dtype == torch.int32 else slots.to(torch.int32)
        torch.ops.turboquant.kvtq_int8_store(x, scale, s32, cache)
    else:
        q = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
        view5 = cache.view(nb, kvh, d // 32, bs, 32)  # NZ relabel, no copy
        sl = slots.long()
        view5[sl // bs, :, :, sl % bs, :] = q.view(-1, kvh, d // 32, 32)
    stats["quant_calls"] += 1
    stats["quant_tokens"] += x.shape[0]


def _int8_reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) > 1:
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output
        num = min(attn_metadata.num_actual_tokens, key.shape[0])
        if num > 0:
            slots = attn_metadata.slot_mapping[:num].long()
            _int8_write(self, key[:num], self.key_cache, slots, "k")
            _int8_write(self, value[:num], self.value_cache, slots, "v")
            from vllm_ascend.attention.attention_v1 import notify_kv_cache_written
            notify_kv_cache_written()
    return query, key, value, output


def _int8_decode(self, query, attn_metadata, output, num_decodes=None):
    """Decode reading the int8 cache directly via native FIA paged antiquant
    (NZ 5D view, per-channel mode 0) - no dense materialization."""
    if self.key_cache is None:
        _warn_once("nocache8", "decode called without kv cache, returning zeros")
        return output.fill_(0)
    import torch_npu
    batch = num_decodes or len(attn_metadata.seq_lens_list)
    nb, bs, kvh, d = self.key_cache.shape
    k5 = self.key_cache.view(nb, kvh, d // 32, bs, 32)
    v5 = self.value_cache.view(nb, kvh, d // 32, bs, 32)
    q = query[:batch]
    if q.dim() == 2:
        q = q.view(batch, self.num_heads, d)
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(2), k5, v5,
        key_antiquant_scale=self._i8_k_aq, value_antiquant_scale=self._i8_v_aq,
        block_table=attn_metadata.block_tables[:batch],
        actual_seq_lengths_kv=list(attn_metadata.seq_lens_list[:batch]),
        num_heads=self.num_heads, num_key_value_heads=self.num_kv_heads,
        input_layout="BNSD", scale=self.scale, block_size=bs,
        antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0,
        inner_precise=1, sparse_mode=0)
    output[:batch] = attn_out.squeeze(2)
    return output


def _int8_dequant_dense(self, cache, block_table, seq_lens_list, which):
    batch, max_blocks = block_table.shape
    nb, bs, kvh, d = cache.shape
    gathered = torch.index_select(cache, 0, block_table.reshape(-1).long())
    gathered = gathered.view(batch, max_blocks * bs, kvh, d)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=cache.device)
    positions = torch.arange(max_blocks * bs, dtype=torch.long, device=cache.device)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    rows = gathered[mask].float()
    return (rows * getattr(self, "_i8_%s_scale" % which)).to(torch.bfloat16).contiguous()


def _int8_prefill_from_cache(self, query, attn_metadata, output,
                             num_decode_tokens, num_tokens, num_decodes):
    import torch_npu
    prefill_q = query[num_decode_tokens:num_tokens]
    prefill_bt = attn_metadata.block_tables[num_decodes:]
    prefill_sl = list(attn_metadata.seq_lens_list[num_decodes:])
    k_dense = _int8_dequant_dense(self, self.key_cache, prefill_bt, prefill_sl, "k")
    v_dense = _int8_dequant_dense(self, self.value_cache, prefill_bt, prefill_sl, "v")
    kv_cum = torch.tensor(prefill_sl, dtype=torch.int32, device=query.device).cumsum(dim=0)
    q_cum = list(attn_metadata.actual_seq_lengths_q)
    prefill_q_cum = [q_cum[i] - num_decode_tokens for i in range(num_decodes, len(q_cum))]
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=prefill_q, key=k_dense, value=v_dense,
        atten_mask=attn_metadata.attn_mask, block_table=None,
        input_layout="TND", block_size=self.key_cache.shape[1],
        actual_seq_lengths=prefill_q_cum, actual_seq_lengths_kv=kv_cum,
        num_key_value_heads=self.num_kv_heads, num_heads=self.num_heads,
        scale=self.scale, sparse_mode=3)
    n_prefill = num_tokens - num_decode_tokens
    head_dim = self.key_cache.shape[-1]
    output[num_decode_tokens:num_tokens] = attn_out.view(n_prefill, self.num_heads, head_dim)
    return output


def _int8_forward_paged(self, query, attn_metadata, output=None):
    from vllm_ascend.attention import attention_v1 as _av1
    if getattr(_av1._EXTRA_CTX, "capturing", False):
        raise RuntimeError("[KVTQ-INT8] int8 mode requires --enforce-eager")
    return _int8_decode(self, query, attn_metadata, output)


def _int8_forward_fia(self, query, key, value, attn_metadata, output, kv_cache=None):
    from vllm_ascend.attention import attention_v1 as _av1
    if getattr(_av1._EXTRA_CTX, "capturing", False):
        raise RuntimeError("[KVTQ-INT8] int8 mode requires --enforce-eager")
    AscendAttentionState = _av1.AscendAttentionState
    state = attn_metadata.attn_state
    if state == AscendAttentionState.DecodeOnly:
        return _int8_decode(self, query, attn_metadata, output)
    if state == AscendAttentionState.PrefillCacheHit:
        raise RuntimeError("[KVTQ-INT8] prefix caching must be disabled "
                           "(--no-enable-prefix-caching)")
    num_tokens = int(attn_metadata.actual_seq_lengths_q[-1])
    query = query[:num_tokens]
    if state == AscendAttentionState.PrefillNoCache:
        return _prefill_fia(self, query, key[:num_tokens], value[:num_tokens],
                            attn_metadata.actual_seq_lengths_q,
                            attn_metadata.actual_seq_lengths_q,
                            attn_metadata, output, 0, num_tokens)
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_decodes = attn_metadata.num_decodes
    if num_decode_tokens > 0:
        _int8_decode(self, query[:num_decode_tokens], attn_metadata, output,
                     num_decodes=num_decodes)
    if attn_metadata.num_prefills > 0:
        seq_lens_list = list(attn_metadata.seq_lens_list)
        q_cum = list(attn_metadata.actual_seq_lengths_q)
        prefill_q_cum = [q_cum[i] - num_decode_tokens for i in range(num_decodes, len(q_cum))]
        prefill_kv_lens = seq_lens_list[num_decodes:]
        prefill_q_lens = [prefill_q_cum[0]] + [
            prefill_q_cum[i] - prefill_q_cum[i - 1] for i in range(1, len(prefill_q_cum))]
        if all(kv == ql for kv, ql in zip(prefill_kv_lens, prefill_q_lens)):
            k_float = key[num_decode_tokens:num_tokens]
            v_float = value[num_decode_tokens:num_tokens]
            _prefill_fia(self, query[num_decode_tokens:], k_float, v_float,
                         prefill_q_cum, prefill_q_cum, attn_metadata, output,
                         num_decode_tokens, num_tokens - num_decode_tokens)
        else:
            _warn_once("chunkhist8", "prefill chunk with history: "
                                     "int8 dequant-from-cache path engaged")
            _int8_prefill_from_cache(self, query, attn_metadata, output,
                                     num_decode_tokens, num_tokens, num_decodes)
    return output

def _dequant_dense(self, cache_bf16, block_table, seq_lens_list, head_dim):
    batch, max_blocks = block_table.shape
    block_size = cache_bf16.shape[1]
    num_kv_heads = cache_bf16.shape[2]
    idx_bytes = _idx_bytes(head_dim)
    row_bytes = cache_bf16.shape[-1] * 2  # self-describing (padded for hybrid)
    cache_u8 = cache_bf16.view(torch.uint8)  # [nb, blk, H, ROW]
    gathered = torch.index_select(
        cache_u8, 0, block_table.reshape(-1).long())  # [B*MB, blk, H, ROW]
    gathered = gathered.view(batch, max_blocks * block_size, num_kv_heads, row_bytes)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=cache_bf16.device)
    positions = torch.arange(max_blocks * block_size, dtype=torch.long, device=cache_bf16.device)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    rows = gathered[mask]  # [total, H, ROW]
    norm = _bf16_from_bytes(rows[..., idx_bytes], rows[..., idx_bytes + 1])
    if _MSE_BITS == 4:
        lut = _get_byte_lut(cache_bf16.device)  # [256, 2] bf16
        codes = rows[..., :idx_bytes].reshape(-1).to(torch.int32)
        dense = torch.index_select(lut, 0, codes).view(
            rows.shape[0], num_kv_heads, head_dim)
    else:
        _, _, centroids = _get_consts(cache_bf16.device, head_dim)
        idx = _unpack_indices(rows[..., :idx_bytes], _MSE_BITS)
        dense = centroids[idx].view(rows.shape[0], num_kv_heads, head_dim)
    dense = dense * norm.unsqueeze(-1).to(torch.bfloat16)
    return dense.contiguous()


def _tq_decode(self, query, attn_metadata, output, num_decodes=None):
    if self.key_cache is None:
        _warn_once("nocache", "decode called without kv cache (dummy run?), returning zeros")
        return output.fill_(0)
    import torch_npu
    batch = num_decodes or len(attn_metadata.seq_lens_list)
    seq_lens_list = list(attn_metadata.seq_lens_list[:batch])
    block_table = attn_metadata.block_tables[:batch]
    head_dim = query.shape[-1] if query.dim() == 3 else query.shape[-1] // self.num_heads  # real head dim (head_size may be packed)
    rot, rot_t, _ = _get_consts(query.device, head_dim)
    q_rot = torch.matmul(query[:batch].float(), rot_t).to(torch.bfloat16)
    k_dense = _dequant_dense(self, self.key_cache, block_table, seq_lens_list, head_dim)
    v_dense = _dequant_dense(self, self.value_cache, block_table, seq_lens_list, head_dim)
    kv_cum = torch.tensor(seq_lens_list, dtype=torch.int32, device=query.device).cumsum(dim=0)
    q_cum = torch.arange(1, batch + 1, dtype=torch.int32, device=query.device)
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=q_rot,
        key=k_dense,
        value=v_dense,
        block_table=None,
        input_layout="TND",
        block_size=self.key_cache.shape[1],
        actual_seq_lengths=q_cum,
        actual_seq_lengths_kv=kv_cum,
        num_key_value_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=0,
    )
    out = torch.matmul(attn_out.view(batch, self.num_heads, head_dim).float(), rot)
    output[:batch] = out.to(output.dtype)
    stats["decode_calls"] += 1
    return output


def _tq_forward_paged_attention(self, query, attn_metadata, output=None):
    from vllm_ascend.attention import attention_v1 as _av1
    if getattr(_av1._EXTRA_CTX, "capturing", False):
        raise RuntimeError("[KVTQ-STORE] store mode requires --enforce-eager (no graph capture)")
    return _tq_decode(self, query, attn_metadata, output)


def _prefill_fia(self, query, key, value, q_cumsum, kv_cumsum, attn_metadata, output, out_offset, num):
    import torch_npu
    head_dim = query.shape[-1] if query.dim() == 3 else query.shape[-1] // self.num_heads
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=attn_metadata.attn_mask,
        block_table=None,
        input_layout="TND",
        block_size=self.key_cache.shape[1] if self.key_cache is not None else 128,
        actual_seq_lengths=q_cumsum,
        actual_seq_lengths_kv=kv_cumsum,
        num_key_value_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=3,
    )
    output[out_offset:out_offset + num] = attn_out.view(num, self.num_heads, head_dim)
    return output


def _tq_forward_fia(self, query, key, value, attn_metadata, output, kv_cache=None):
    from vllm_ascend.attention import attention_v1 as _av1
    if getattr(_av1._EXTRA_CTX, "capturing", False):
        raise RuntimeError("[KVTQ-STORE] store mode requires --enforce-eager (no graph capture)")
    AscendAttentionState = _av1.AscendAttentionState
    state = attn_metadata.attn_state
    if state == AscendAttentionState.DecodeOnly:
        return _tq_decode(self, query, attn_metadata, output)
    if state == AscendAttentionState.PrefillCacheHit:
        raise RuntimeError("[KVTQ-STORE] prefix caching must be disabled "
                           "(--no-enable-prefix-caching)")
    num_tokens = int(attn_metadata.actual_seq_lengths_q[-1])
    query = query[:num_tokens]

    if state == AscendAttentionState.PrefillNoCache:
        return _prefill_fia(self, query, key[:num_tokens], value[:num_tokens],
                            attn_metadata.actual_seq_lengths_q,
                            attn_metadata.actual_seq_lengths_q,
                            attn_metadata, output, 0, num_tokens)

    # ChunkedPrefill: decodes occupy [0, num_decode_tokens), prefills the rest.
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_decodes = attn_metadata.num_decodes
    if num_decode_tokens > 0:
        _tq_decode(self, query[:num_decode_tokens], attn_metadata, output,
                   num_decodes=num_decodes)
    if attn_metadata.num_prefills > 0:
        seq_lens_list = list(attn_metadata.seq_lens_list)
        q_cum = list(attn_metadata.actual_seq_lengths_q)
        prefill_q_cum = [q_cum[i] - num_decode_tokens for i in range(num_decodes, len(q_cum))]
        prefill_kv_lens = seq_lens_list[num_decodes:]
        prefill_q_lens = [prefill_q_cum[0]] + [
            prefill_q_cum[i] - prefill_q_cum[i - 1] for i in range(1, len(prefill_q_cum))]
        if all(kv == ql for kv, ql in zip(prefill_kv_lens, prefill_q_lens)):
            k_float = key[num_decode_tokens:num_tokens]
            v_float = value[num_decode_tokens:num_tokens]
            _prefill_fia(self, query[num_decode_tokens:], k_float, v_float,
                         prefill_q_cum, prefill_q_cum, attn_metadata, output,
                         num_decode_tokens, num_tokens - num_decode_tokens)
        else:
            _warn_once("chunkhist", "prefill chunk with history: batched "
                                    "dequant-from-cache path engaged")
            _prefill_from_cache(self, query, attn_metadata, output,
                                num_decode_tokens, num_tokens, num_decodes)
    return output


def _prefill_from_cache(self, query, attn_metadata, output,
                        num_decode_tokens, num_tokens, num_decodes):
    """Batched prefill-with-history: dequant full seqs (incl. current chunk,
    already written to cache by reshape_and_cache) and run one FIA TND call,
    mirroring the production C8 chunked-prefill path."""
    import torch_npu
    head_dim = query.shape[-1] if query.dim() == 3 else query.shape[-1] // self.num_heads
    rot, rot_t, _ = _get_consts(query.device, head_dim)
    prefill_q = query[num_decode_tokens:num_tokens]
    q_rot = torch.matmul(prefill_q.float(), rot_t).to(torch.bfloat16)
    prefill_bt = attn_metadata.block_tables[num_decodes:]
    prefill_sl = list(attn_metadata.seq_lens_list[num_decodes:])
    k_dense = _dequant_dense(self, self.key_cache, prefill_bt, prefill_sl, head_dim)
    v_dense = _dequant_dense(self, self.value_cache, prefill_bt, prefill_sl, head_dim)
    kv_cum = torch.tensor(prefill_sl, dtype=torch.int32, device=query.device).cumsum(dim=0)
    q_cum = list(attn_metadata.actual_seq_lengths_q)
    prefill_q_cum = [q_cum[i] - num_decode_tokens for i in range(num_decodes, len(q_cum))]
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=q_rot, key=k_dense, value=v_dense,
        atten_mask=attn_metadata.attn_mask, block_table=None,
        input_layout="TND", block_size=self.key_cache.shape[1],
        actual_seq_lengths=prefill_q_cum, actual_seq_lengths_kv=kv_cum,
        num_key_value_heads=self.num_kv_heads, num_heads=self.num_heads,
        scale=self.scale, sparse_mode=3,
    )
    n_prefill = num_tokens - num_decode_tokens
    out = torch.matmul(attn_out.view(n_prefill, self.num_heads, head_dim).float(), rot)
    output[num_decode_tokens:num_tokens] = out.to(output.dtype)
    return output

_installed = False


def _patch_hybrid_page_alignment():
    """vllm-ascend patch_mamba_config computes attention/mamba page sizes from
    ModelConfig.get_head_size(), bypassing the Attention.get_kv_cache_spec hook.
    Wrap verify_and_update_config so the page math sees the packed row size;
    without this the whole capacity gain is eaten by hybrid page padding."""
    try:
        from vllm.config import ModelConfig
    except Exception:
        try:
            from vllm.config.model import ModelConfig
        except Exception as exc:
            print(f"[KVTQ-STORE] page-alignment patch skipped: {exc}",
                  flush=True)
            return
    try:
        # Importing the vllm-ascend patch module executes its bottom-level
        # assignment to HybridAttentionMambaModelConfig; wrap AFTER that so
        # our wrapper is not clobbered.
        import vllm_ascend.patch.platform.patch_mamba_config  # noqa: F401
        from vllm.model_executor.models.config import (
            HybridAttentionMambaModelConfig)
    except Exception as exc:
        print(f"[KVTQ-STORE] page-alignment patch skipped: {exc}", flush=True)
        return
    orig_verify = HybridAttentionMambaModelConfig.verify_and_update_config.__func__
    orig_ghs = ModelConfig.get_head_size

    def _verify(cls, vllm_config, *args, **kwargs):
        ModelConfig.get_head_size = lambda mc: _row_bytes(orig_ghs(mc), True) // 2
        try:
            return orig_verify(cls, vllm_config, *args, **kwargs)
        finally:
            ModelConfig.get_head_size = orig_ghs

    HybridAttentionMambaModelConfig.verify_and_update_config = classmethod(_verify)
    print("[KVTQ-STORE] hybrid page-alignment patch installed", flush=True)


def install():
    global _installed
    if _installed:
        return
    _installed = True
    if not _ENABLED:
        return
    import vllm_ascend.ops  # noqa: F401  (pre-load to avoid circular import)
    from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec
    from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                    AscendAttentionBackendImpl)

    # Shrink the KV cache at the spec level: report each BF16 vector
    # (2*head_dim B) as row_bytes packed bytes (head_dim*bits/8 + 2),
    # expressed as head_size=row_bytes//2 BF16 "elements" so that page
    # accounting, num_blocks derivation and tensor allocation stay consistent.
    import dataclasses

    from vllm.model_executor.layers.attention.attention import Attention

    orig_get_spec = Attention.get_kv_cache_spec

    def _get_spec(self_attn, vllm_config):
        spec = orig_get_spec(self_attn, vllm_config)
        if spec is not None and type(spec) is FullAttentionSpec:
            if _INT8:
                # int8 per-channel: same element count, half the bytes
                spec = dataclasses.replace(spec, dtype=torch.int8)
            else:
                row = _row_bytes(spec.head_size, _is_hybrid_config(vllm_config))
                spec = dataclasses.replace(spec, head_size=row // 2,
                                           head_size_v=row // 2)
        return spec

    Attention.get_kv_cache_spec = _get_spec

    if _INT8:
        # Hybrid page alignment for int8 mode is not handled (page math in
        # patch_mamba_config uses cache_config dtype, not the spec dtype).
        AscendAttentionBackendImpl.reshape_and_cache = _int8_reshape_and_cache
        AscendAttentionBackendImpl.forward_paged_attention = _int8_forward_paged
        AscendAttentionBackendImpl.forward_fused_infer_attention = _int8_forward_fia
        print("[KVTQ-INT8] installed (per-channel int8 KV, native FIA paged "
              "antiquant decode, ~2x capacity + decode bandwidth win)",
              flush=True)
        return

    _patch_hybrid_page_alignment()

    AscendAttentionBackendImpl.reshape_and_cache = _tq_reshape_and_cache
    AscendAttentionBackendImpl.forward_paged_attention = _tq_forward_paged_attention
    AscendAttentionBackendImpl.forward_fused_infer_attention = _tq_forward_fia

    print(f"[KVTQ-STORE] installed (mse_bits={_MSE_BITS}, "
          f"row_bytes=head_dim*{_MSE_BITS}/8+2, head_dim 128/256 -> "
          f"{_row_bytes(128)}/{_row_bytes(256)} B)", flush=True)
