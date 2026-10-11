"""Real-storage TurboQuant KV cache for vllm-ascend (prototype v1).

Unlike shadow mode (quantize + discard, cache stays BF16), store mode:
  - shrinks the paged KV cache allocation to the packed TurboQuant format
    (per token per head: mse_bits-packed indices + bf16 norm), giving ~3.9x
    more tokens for the same memory (bits=4, head_dim=128: 66 B vs 256 B);
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
"""
import math
import os
from itertools import accumulate

import torch

_ENABLED = os.environ.get("VLLM_ASCEND_KVTQ_STORE", "0") == "1"
_MSE_BITS = int(os.environ.get("VLLM_ASCEND_KVTQ_BITS", "4"))
_HEAD_DIM = 128
_IDX_BYTES = {2: 32, 3: 48, 4: 64}[_MSE_BITS]
_ROW_BYTES = _IDX_BYTES + 2  # + bf16 norm; qjl/gamma not stored in v1

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


def _get_consts(device):
    key = str(device)
    if key not in _cache:
        gen = torch.Generator().manual_seed(20260929)
        q, _ = torch.linalg.qr(
            torch.randn(_HEAD_DIM, _HEAD_DIM, generator=gen, dtype=torch.float64))
        rot = q.to(torch.float32).to(device).contiguous()
        cent = torch.tensor(CENTROIDS[_MSE_BITS], dtype=torch.bfloat16, device=device)
        _cache[key] = (rot, rot.T.contiguous(), cent)
    return _cache[key]


def _warn_once(tag, msg):
    if tag not in stats["warned"]:
        stats["warned"].add(tag)
        print(f"[KVTQ-STORE] WARNING {msg}", flush=True)


def _pack_rows(idx, norm):
    norm_b = norm.unsqueeze(-1).view(torch.uint8)  # [N,H,2]
    return torch.cat([idx, norm_b], dim=-1)  # [N,H,ROW_BYTES] uint8


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
    rot, _, _ = _get_consts(tensor.device)
    x = tensor.contiguous()
    rot32, qjl32 = _get_quant_matrices(x.device)
    idx, _qjlb, norm, _gamma = torch.ops.turboquant.kv_cache_turbo_quant(
        x, rot32, qjl32, _MSE_BITS)
    packed = _pack_rows(idx, norm)
    num_kv_heads = cache_bf16.shape[2]
    flat = cache_bf16.view(torch.uint8).view(-1, num_kv_heads, _ROW_BYTES)
    flat.index_copy_(0, slots, packed)
    stats["quant_calls"] += 1
    stats["quant_tokens"] += x.shape[0]
    if stats["quant_calls"] == 1 or stats["quant_calls"] % 1000 == 0:
        print(f"[KVTQ-STORE] pid={os.getpid()} quant_calls={stats['quant_calls']} "
              f"tokens={stats['quant_tokens']}", flush=True)


def _get_quant_matrices(device):
    rot, _, _ = _get_consts(device)
    key = ("quant", str(device))
    if key not in _cache:
        gen = torch.Generator().manual_seed(20260929)
        _ = torch.linalg.qr(torch.randn(_HEAD_DIM, _HEAD_DIM, generator=gen, dtype=torch.float64))
        qjl = (torch.randn(_HEAD_DIM, _HEAD_DIM, generator=gen, dtype=torch.float64)
               .to(torch.float32) / math.sqrt(_HEAD_DIM)).to(device).contiguous()
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


def _dequant_dense_reference(self, cache_bf16, block_table, seq_lens_list):
    batch, max_blocks = block_table.shape
    block_size = cache_bf16.shape[1]
    num_kv_heads = cache_bf16.shape[2]
    cache_u8 = cache_bf16.view(torch.uint8)  # [nb, blk, H, ROW]
    gathered = torch.index_select(
        cache_u8, 0, block_table.reshape(-1).long())  # [B*MB, blk, H, ROW]
    gathered = gathered.view(batch, max_blocks * block_size, num_kv_heads, _ROW_BYTES)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long, device=cache_bf16.device)
    positions = torch.arange(max_blocks * block_size, dtype=torch.long, device=cache_bf16.device)
    mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    rows = gathered[mask]  # [total, H, ROW]
    norm = _bf16_from_bytes(rows[..., _IDX_BYTES], rows[..., _IDX_BYTES + 1])
    if _MSE_BITS == 4:
        lut = _get_byte_lut(cache_bf16.device)  # [256, 2] bf16
        codes = rows[..., :_IDX_BYTES].reshape(-1).to(torch.int32)
        dense = torch.index_select(lut, 0, codes).view(
            rows.shape[0], num_kv_heads, _HEAD_DIM)
    else:
        _, _, centroids = _get_consts(cache_bf16.device)
        idx = _unpack_indices(rows[..., :_IDX_BYTES], _MSE_BITS)
        dense = centroids[idx].view(rows.shape[0], num_kv_heads, _HEAD_DIM)
    dense = dense * norm.unsqueeze(-1).to(torch.bfloat16)
    return dense.contiguous()


def _read_plan(attn_metadata, segment, device, seq_lens_list):
    """Share immutable sequence metadata across K/V and layers of one step.

    Only two small metadata entries are retained. Reused metadata objects are
    checked against the current lengths; block table values are always read
    fresh by the kernel, so physical-page reuse cannot return stale KV data.
    """
    fingerprint = (str(device), tuple(seq_lens_list))
    plans = getattr(attn_metadata, '_kvtq_read_plans', None) if attn_metadata is not None else None
    if plans is not None and segment in plans and plans[segment][0] == fingerprint:
        return plans[segment][1]
    cumulative = [0, *accumulate(seq_lens_list)]
    plan = (torch.tensor(cumulative, dtype=torch.int32, device=device),
            cumulative, sum(seq_lens_list), max(seq_lens_list, default=0))
    if attn_metadata is not None:
        if plans is None:
            plans = {}
            attn_metadata._kvtq_read_plans = plans
        plans[segment] = (fingerprint, plan)
    return plan


def _dequant_dense(self, cache_bf16, block_table, seq_lens_list,
                   attn_metadata=None, segment='decode'):
    if _MSE_BITS != 4:
        return _dequant_dense_reference(self, cache_bf16, block_table, seq_lens_list)
    key = ('paged_read4', str(cache_bf16.device))
    if key not in _cache:
        from kvtq_read import ModelNew
        _cache[key] = ModelNew()
    cumulative, _, total_tokens, max_seq_len = _read_plan(
        attn_metadata, segment, cache_bf16.device, seq_lens_list)
    _, _, centroids = _get_consts(cache_bf16.device)
    return _cache[key](cache_bf16.view(torch.uint8), block_table, cumulative,
                       centroids.view(torch.int16), total_tokens, max_seq_len,
                       seq_lens_list).view(torch.bfloat16)


def _tq_decode(self, query, attn_metadata, output, num_decodes=None):
    if self.key_cache is None:
        _warn_once("nocache", "decode called without kv cache (dummy run?), returning zeros")
        return output.fill_(0)
    import torch_npu
    batch = num_decodes or len(attn_metadata.seq_lens_list)
    seq_lens_list = list(attn_metadata.seq_lens_list[:batch])
    block_table = attn_metadata.block_tables[:batch]
    rot, rot_t, _ = _get_consts(query.device)
    q_rot = torch.matmul(query[:batch].float(), rot_t).to(torch.bfloat16)
    k_dense = _dequant_dense(self, self.key_cache, block_table, seq_lens_list, attn_metadata)
    v_dense = _dequant_dense(self, self.value_cache, block_table, seq_lens_list, attn_metadata)
    # FIA accepts host cumulative lengths. They are already known by the
    # scheduler, so tensor creation/cumsum and a later D2H read are unnecessary.
    kv_cum = list(accumulate(seq_lens_list))
    q_cum = list(range(1, batch + 1))
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
    out = torch.matmul(attn_out.view(batch, self.num_heads, _HEAD_DIM).float(), rot)
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
    output[out_offset:out_offset + num] = attn_out.view(num, self.num_heads, _HEAD_DIM)
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
    rot, rot_t, _ = _get_consts(query.device)
    prefill_q = query[num_decode_tokens:num_tokens]
    q_rot = torch.matmul(prefill_q.float(), rot_t).to(torch.bfloat16)
    prefill_bt = attn_metadata.block_tables[num_decodes:]
    prefill_sl = list(attn_metadata.seq_lens_list[num_decodes:])
    k_dense = _dequant_dense(self, self.key_cache, prefill_bt, prefill_sl,
                             attn_metadata, 'prefill')
    v_dense = _dequant_dense(self, self.value_cache, prefill_bt, prefill_sl,
                             attn_metadata, 'prefill')
    kv_cum = list(accumulate(prefill_sl))
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
    out = torch.matmul(attn_out.view(n_prefill, self.num_heads, _HEAD_DIM).float(), rot)
    output[num_decode_tokens:num_tokens] = out.to(output.dtype)
    return output

_installed = False


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

    # Shrink the KV cache at the spec level: report each 128-dim BF16 vector
    # (256 B) as _ROW_BYTES packed bytes (66 B at bits=4), expressed as
    # head_size=_ROW_BYTES//2 BF16 "elements" so that page accounting,
    # num_blocks derivation and tensor allocation all stay consistent.
    import dataclasses

    from vllm.model_executor.layers.attention.attention import Attention

    orig_get_spec = Attention.get_kv_cache_spec

    def _get_spec(self_attn, vllm_config):
        spec = orig_get_spec(self_attn, vllm_config)
        if spec is not None and type(spec) is FullAttentionSpec:
            spec = dataclasses.replace(spec, head_size=_ROW_BYTES // 2,
                                       head_size_v=_ROW_BYTES // 2)
        return spec

    Attention.get_kv_cache_spec = _get_spec

    AscendAttentionBackendImpl.reshape_and_cache = _tq_reshape_and_cache
    AscendAttentionBackendImpl.forward_paged_attention = _tq_forward_paged_attention
    AscendAttentionBackendImpl.forward_fused_infer_attention = _tq_forward_fia

    print(f"[KVTQ-STORE] installed (mse_bits={_MSE_BITS}, row_bytes={_ROW_BYTES}, "
          f"compression={2 * _HEAD_DIM * 2 / (2 * _ROW_BYTES):.2f}x)", flush=True)
