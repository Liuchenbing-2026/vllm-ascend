"""CPU mathematical prototype for decode directly from the existing store4 pages.

This is neither an NPU implementation nor an FIA precision/performance result.
The existing store codec remains the only dequantization reference. Attention
uses FP64 here to isolate the tiled recurrence from device reduction choices.
"""

import math

import torch
import torch.nn.functional as functional

import kvtq_store as store


def _validate(query, key, value, table, lengths, rotation, tile_tokens):
    if store._MSE_BITS != 4 or store._ROW_BYTES != 66:
        raise ValueError("this prototype requires the existing 4-bit/66-byte layout")
    tensors = (query, key, value, table, rotation)
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("CPU mathematical reference only")
    if (query.ndim != 3 or query.shape[-1] != store._HEAD_DIM
            or query.dtype != torch.bfloat16):
        raise ValueError("query must be BF16 [batch, query_heads, 128]")
    if (key.ndim != 4 or key.shape[-1] != store._ROW_BYTES
            or key.dtype != torch.uint8 or value.dtype != torch.uint8
            or value.shape != key.shape or not key.is_contiguous()
            or not value.is_contiguous()):
        raise ValueError("K/V must be contiguous uint8 [blocks, page, kv_heads, 66]")
    if key.shape[1] <= 0 or key.shape[2] <= 0 or query.shape[1] <= 0:
        raise ValueError("page size and head counts must be positive")
    if query.shape[1] % key.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if (table.ndim != 2 or table.shape[0] != query.shape[0]
            or table.dtype not in (torch.int32, torch.int64)):
        raise ValueError("block table must be integer [batch, max_pages]")
    if len(lengths) != query.shape[0]:
        raise ValueError("one KV length is required per query")
    if (rotation.shape != (store._HEAD_DIM, store._HEAD_DIM)
            or rotation.dtype != torch.float32):
        raise ValueError("rotation must be FP32 [128, 128]")
    if isinstance(tile_tokens, bool) or not isinstance(tile_tokens, int) or tile_tokens <= 0:
        raise ValueError("tile_tokens must be a positive integer")
    for batch, length in enumerate(lengths):
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("decode requires positive integer KV lengths")
        pages = (length + key.shape[1] - 1) // key.shape[1]
        if pages > table.shape[1]:
            raise ValueError("KV length exceeds block-table coverage")
        active = table[batch, :pages]
        if torch.any(active < 0) or torch.any(active >= key.shape[0]):
            raise ValueError("active physical page is outside the cache")


def _decode_tile(cache, table_row, start, stop):
    """Gather only this tile; delegate every codec rule to the existing reference."""
    positions = torch.arange(start, stop, dtype=torch.long)
    physical = table_row[positions // cache.shape[1]].long()
    rows = cache[physical, positions % cache.shape[1]].contiguous()
    # A temporary, single-page view lets the unchanged codec decode this tile.
    tile_cache = rows.unsqueeze(0).view(torch.bfloat16)
    tile_table = torch.zeros((1, 1), dtype=torch.int32)
    return store._dequant_dense_reference(None, tile_cache, tile_table, [stop - start])


def _finish(attention_output, rotation, output):
    # Preserve the BF16 attention-output boundary before the FP32 inverse rotation.
    result = (attention_output.to(torch.bfloat16).float() @ rotation).to(torch.bfloat16)
    if output is None:
        return result
    if output.device.type != "cpu" or output.dtype != result.dtype or output.shape != result.shape:
        raise ValueError("output must be CPU BF16 with the query shape")
    output.copy_(result)
    return output


def packed_decode_cpu(query, key, value, table, lengths, rotation, *,
                      tile_tokens=64, scale=None, output=None):
    """One-query GQA decode, reading packed K/V a tile at a time on CPU.

    Inactive table entries are ignored. Empty batch is supported; zero-length
    nonempty sequences are rejected, matching the existing decode contract.
    Only per-tile K/V and per-head online-softmax state are materialized here.
    """
    _validate(query, key, value, table, lengths, rotation, tile_tokens)
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    rotated_query = (query.float() @ rotation.T).to(torch.bfloat16).double()
    result = torch.empty(query.shape, dtype=torch.float64)
    groups = query.shape[1] // key.shape[2]
    for batch, length in enumerate(lengths):
        maximum = torch.full((query.shape[1],), -torch.inf, dtype=torch.float64)
        denominator = torch.zeros(query.shape[1], dtype=torch.float64)
        numerator = torch.zeros((query.shape[1], query.shape[2]), dtype=torch.float64)
        for start in range(0, length, tile_tokens):
            stop = min(length, start + tile_tokens)
            k = _decode_tile(key, table[batch], start, stop).double()
            v = _decode_tile(value, table[batch], start, stop).double()
            k = k.repeat_interleave(groups, dim=1).transpose(0, 1)
            v = v.repeat_interleave(groups, dim=1).transpose(0, 1)
            logits = torch.bmm(rotated_query[batch].unsqueeze(1), k.transpose(1, 2))[:, 0] * scale
            next_maximum = torch.maximum(maximum, logits.amax(dim=-1))
            carry = torch.exp(maximum - next_maximum)
            probabilities = torch.exp(logits - next_maximum.unsqueeze(-1))
            numerator = numerator * carry.unsqueeze(-1) + torch.bmm(
                probabilities.unsqueeze(1), v)[:, 0]
            denominator = denominator * carry + probabilities.sum(dim=-1)
            maximum = next_maximum
        result[batch] = numerator / denominator.unsqueeze(-1)
    return _finish(result, rotation, output)


def dense_decode_oracle_cpu(query, key, value, table, lengths, rotation, *, scale=None):
    """Independent dense SDPA math oracle using the unchanged production codec."""
    _validate(query, key, value, table, lengths, rotation, 1)
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if query.shape[0] == 0:
        return torch.empty_like(query)
    safe_table = table.clone()
    for batch, length in enumerate(lengths):
        safe_table[batch, (length + key.shape[1] - 1) // key.shape[1]:] = 0
    k = store._dequant_dense_reference(None, key.view(torch.bfloat16), safe_table, lengths)
    v = store._dequant_dense_reference(None, value.view(torch.bfloat16), safe_table, lengths)
    rotated_query = (query.float() @ rotation.T).to(torch.bfloat16)
    result = torch.empty(query.shape, dtype=torch.float64)
    offset = 0
    for batch, length in enumerate(lengths):
        groups = query.shape[1] // key.shape[2]
        dense_k = k[offset:offset + length].double().repeat_interleave(groups, dim=1).transpose(0, 1)
        dense_v = v[offset:offset + length].double().repeat_interleave(groups, dim=1).transpose(0, 1)
        result[batch] = functional.scaled_dot_product_attention(
            rotated_query[batch].double().unsqueeze(1), dense_k, dense_v,
            dropout_p=0.0, is_causal=False, scale=scale)[:, 0]
        offset += length
    return _finish(result, rotation, None)
