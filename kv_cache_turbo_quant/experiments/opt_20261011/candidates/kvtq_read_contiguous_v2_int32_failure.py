"""Fused paged read for the original 128-dimension store4 byte layout.

The kernel keeps the original BF16 centroid and norm rounding. It produces
only the current attention's dense workspace, never a persistent BF16 cache.
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['total_tokens', 'batch', 'num_tiles'])
def _paged_read4(Cache, Table, Cum, CentroidBits, Output,
                 table_stride0: tl.constexpr, table_stride1: tl.constexpr,
                 block_size: tl.constexpr, heads: tl.constexpr,
                 total_tokens, batch, num_tiles, search_steps: tl.constexpr,
                 tile_rows: tl.constexpr, programs: tl.constexpr, load_width: tl.constexpr):
    # Output tiles are disjoint, including a fully allocated padded last tile.
    # No masked store can read or preserve a neighbouring sequence's rows.
    lanes = tl.arange(0, 64)
    row_lane = tl.arange(0, tile_rows)
    cent = tl.load(CentroidBits + tl.arange(0, 16)).to(tl.uint16).to(
        tl.bfloat16, bitcast=True).to(tl.float32)
    for tile in range(tl.program_id(0), num_tiles, programs):
        row = tile * tile_rows + row_lane
        token = tl.minimum(row // heads, total_tokens - 1)
        head = row % heads
        lower = tl.full((tile_rows,), 0, tl.int32)
        upper = tl.full((tile_rows,), batch - 1, tl.int32)
        # Upper-bound search handles repeated cumulative values (empty seqs).
        for _ in range(search_steps):
            mid = (lower + upper + 1) // 2
            boundary = tl.load(Cum + mid)
            right = boundary <= token
            lower = tl.where(right, mid, lower)
            upper = tl.where(right, upper, mid - 1)
        seq = lower
        begin = tl.load(Cum + seq)
        position = token - begin
        physical = tl.load(Table + seq * table_stride0 +
                           (position // block_size) * table_stride1).to(tl.int64)
        source_row = ((physical * block_size + position % block_size) * heads + head)
        first_physical = tl.sum(tl.where(row_lane == 0, physical.to(tl.int32), 0), axis=0).to(tl.int64)
        first_position = tl.sum(tl.where(row_lane == 0, position, 0), axis=0)
        first_source = ((first_physical * block_size + first_position % block_size) * heads
                        + (tile * tile_rows) % heads)
        consecutive = tl.sum((source_row == first_source + row_lane).to(tl.int32), axis=0) == tile_rows
        if consecutive:
            byte_offsets = tl.arange(0, load_width)
            raw = tl.load(Cache + first_source * 66 + byte_offsets,
                          mask=byte_offsets < tile_rows * 66, other=0).to(tl.int32)
            packed_offsets = (row_lane[:, None] * 66 + lanes[None, :]).reshape((tile_rows * 64,))
            packed = tl.gather(raw, packed_offsets, axis=0).reshape((tile_rows, 64))
            norm_lo = tl.gather(raw, row_lane * 66 + 64, axis=0)
            norm_hi = tl.gather(raw, row_lane * 66 + 65, axis=0)
        else:
            packed = tl.load(Cache + source_row[:, None] * 66 + lanes[None, :]).to(tl.int32)
            norm_lo = tl.load(Cache + source_row * 66 + 64).to(tl.int32)
            norm_hi = tl.load(Cache + source_row * 66 + 65).to(tl.int32)
        lo = (packed & 15).reshape((tile_rows * 64,))
        hi = (packed >> 4).reshape((tile_rows * 64,))
        low_cent = tl.gather(cent, lo, axis=0).reshape((tile_rows, 64))
        high_cent = tl.gather(cent, hi, axis=0).reshape((tile_rows, 64))
        norm = ((norm_lo | (norm_hi << 8)) << 16).to(tl.float32, bitcast=True)
        # Original read casts the stored norm back to BF16 before multiplying
        # two BF16 values. FP32 multiplication followed by BF16 RNE is identical
        # for the finite BF16 operands accepted by this storage format.
        low = (low_cent * norm[:, None]).to(tl.bfloat16).to(tl.int16, bitcast=True)
        high = (high_cent * norm[:, None]).to(tl.bfloat16).to(tl.int16, bitcast=True)
        values = tl.join(low, high).reshape((tile_rows, 128))
        tl.store(Output + row.to(tl.int64)[:, None] * 128 + tl.arange(0, 128)[None, :], values)


class ModelNew(torch.nn.Module):
    """Integer-input/output wrapper enables strict bit equality in the verifier."""

    def __init__(self):
        super().__init__()
        from triton.runtime import driver
        self.vector_cores = driver.active.utils.get_device_properties(
            torch.npu.current_device())["num_vectorcore"]

    def forward(self, cache_u8, block_table, cumulative, centroid_bits,
                total_tokens, max_seq_len, seq_lens_list):
        heads = cache_u8.shape[2]
        if cache_u8.shape[-1] != 66 or not cache_u8.is_contiguous():
            raise ValueError('store4 read requires contiguous 66-byte packed rows')
        if total_tokens == 0:
            return torch.empty((0, heads, 128), dtype=torch.int16, device=cache_u8.device)
        tile_rows = 16
        num_tiles = triton.cdiv(total_tokens * heads, tile_rows)
        output = torch.empty((num_tiles * tile_rows, 128), dtype=torch.int16,
                             device=cache_u8.device)
        batch = block_table.shape[0]
        programs = min(num_tiles, self.vector_cores)
        _paged_read4[(programs,)](
            cache_u8, block_table, cumulative, centroid_bits, output,
            block_table.stride(0), block_table.stride(1), cache_u8.shape[1],
            heads, total_tokens, batch, num_tiles, max(1, (batch - 1).bit_length()),
            tile_rows, programs, triton.next_power_of_2(tile_rows * 66))
        return output[:total_tokens * heads].view(total_tokens, heads, 128)
