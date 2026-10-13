"""Pack indices and norms directly into the existing 66-byte store4 cache."""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

HEAD_DIM = 128
INDEX_BYTES = HEAD_DIM // 2
ROW_BYTES = INDEX_BYTES + 2
TILE_ROWS = 16


@triton.jit(do_not_specialize=["rows", "tiles"])
def _pack_scatter4(
    Indices,
    Norms,
    Slots,
    Cache,
    rows,
    tiles,
    heads: tl.constexpr,
    programs: tl.constexpr,
    tile_rows: tl.constexpr,
    index_bytes: tl.constexpr,
    row_bytes: tl.constexpr,
):
    lane = tl.arange(0, index_bytes)
    row_lane = tl.arange(0, tile_rows)
    for tile in range(tl.program_id(0), tiles, programs):
        row = tile * tile_rows + row_lane
        valid = row < rows
        slot = tl.load(Slots + row // heads, mask=valid, other=0).to(tl.int64)
        target = (slot * heads + row % heads) * row_bytes
        index = tl.load(
            Indices + row[:, None] * index_bytes + lane[None, :],
            mask=valid[:, None],
            other=0,
        )
        norm = tl.load(Norms + row, mask=valid, other=0).to(tl.uint16, bitcast=True)
        tl.store(Cache + target[:, None] + lane[None, :], index, mask=valid[:, None])
        tl.store(Cache + target + index_bytes, norm.to(tl.uint8), mask=valid)
        tl.store(Cache + target + index_bytes + 1, (norm >> 8).to(tl.uint8), mask=valid)


@triton.jit
def _scatter_packed_rows(
    Cache, target, index, norm, selected, index_bytes: tl.constexpr
):
    lane = tl.arange(0, index_bytes)
    tl.store(Cache + target[:, None] + lane[None, :], index, mask=selected[:, None])
    tl.store(Cache + target + index_bytes, norm.to(tl.uint8), mask=selected)
    tl.store(Cache + target + index_bytes + 1, (norm >> 8).to(tl.uint8), mask=selected)


@triton.jit(do_not_specialize=["rows", "tiles"])
def _pack_scatter_pair4(
    Indices,
    Norms,
    Slots,
    KeyCache,
    ValueCache,
    rows,
    tiles,
    heads: tl.constexpr,
    programs: tl.constexpr,
    tile_rows: tl.constexpr,
    index_bytes: tl.constexpr,
    row_bytes: tl.constexpr,
):
    lane = tl.arange(0, index_bytes)
    row_lane = tl.arange(0, tile_rows)
    for tile in range(tl.program_id(0), tiles, programs):
        row = tile * tile_rows + row_lane
        valid = row < rows
        slot = tl.load(Slots + row // heads, mask=valid, other=0).to(tl.int64)
        target = (slot * heads + row % heads) * row_bytes
        key_row = row // heads * (2 * heads) + row % heads
        index = tl.load(
            Indices + key_row[:, None] * index_bytes + lane[None, :],
            mask=valid[:, None],
            other=0,
        )
        value_index = tl.load(
            Indices + (key_row + heads)[:, None] * index_bytes + lane[None, :],
            mask=valid[:, None],
            other=0,
        )
        norm = tl.load(Norms + key_row, mask=valid, other=0).to(tl.uint16, bitcast=True)
        value_norm = tl.load(Norms + key_row + heads, mask=valid, other=0).to(
            tl.uint16, bitcast=True
        )
        _scatter_packed_rows(KeyCache, target, index, norm, valid, index_bytes)
        _scatter_packed_rows(
            ValueCache, target, value_index, value_norm, valid, index_bytes
        )


class PackedCacheWriter:
    def __init__(self):
        from triton.runtime import driver

        self.vector_cores = driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )["num_vectorcore"]

    def __call__(self, indices, norms, slots, cache):
        if not all(t.is_contiguous() for t in [indices, norms, slots, cache]):
            raise ValueError("packed cache writer requires contiguous tensors")
        if indices.dtype != torch.uint8 or norms.dtype != torch.bfloat16:
            raise ValueError(
                "packed cache writer requires uint8 indices and BF16 norms"
            )
        if indices.shape[-1] != INDEX_BYTES or cache.shape[-1] * 2 != ROW_BYTES:
            raise ValueError("packed cache writer requires the 66-byte store4 layout")
        heads = indices.shape[1]
        if norms.shape != indices.shape[:2] or slots.numel() != indices.shape[0]:
            raise ValueError("packed cache writer shape mismatch")
        rows = norms.numel()
        if rows == 0:
            return
        tiles = triton.cdiv(rows, TILE_ROWS)
        programs = min(tiles, self.vector_cores)
        _pack_scatter4[(programs,)](
            indices,
            norms,
            slots,
            cache.view(torch.uint8),
            rows,
            tiles,
            heads,
            programs,
            TILE_ROWS,
            INDEX_BYTES,
            ROW_BYTES,
        )

    def write_pair(self, indices, norms, slots, key_cache, value_cache):
        if not all(
            t.is_contiguous() for t in [indices, norms, slots, key_cache, value_cache]
        ):
            raise ValueError("paired cache writer requires contiguous tensors")
        if indices.dtype != torch.uint8 or norms.dtype != torch.bfloat16:
            raise ValueError(
                "paired cache writer requires uint8 indices and BF16 norms"
            )
        heads = key_cache.shape[-2]
        if value_cache.shape != key_cache.shape or key_cache.shape[-1] * 2 != ROW_BYTES:
            raise ValueError("paired cache writer requires matching 66-byte caches")
        if (
            indices.shape[1:] != (2 * heads, INDEX_BYTES)
            or norms.shape != indices.shape[:2]
        ):
            raise ValueError("paired cache writer shape mismatch")
        if slots.numel() != indices.shape[0]:
            raise ValueError("paired cache writer slot count mismatch")
        rows = norms.numel() // 2
        if rows == 0:
            return
        tiles = triton.cdiv(rows, TILE_ROWS)
        programs = min(tiles, self.vector_cores)
        _pack_scatter_pair4[(programs,)](
            indices,
            norms,
            slots,
            key_cache.view(torch.uint8),
            value_cache.view(torch.uint8),
            rows,
            tiles,
            heads,
            programs,
            TILE_ROWS,
            INDEX_BYTES,
            ROW_BYTES,
        )
