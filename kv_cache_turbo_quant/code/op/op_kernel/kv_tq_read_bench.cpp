// Microbenchmark: tq4 packed read + in-UB dequant vs raw bf16 read.
// Phase-1 discriminator for int4 fused-read feasibility
// (docs/int4_fused_read_design.md). Checksum accumulation prevents
// dead-code elimination; checksum values themselves are meaningless.
//
// Flat-word processing: every 32-bit word (code words AND pad/norm words)
// goes through the same nibble-extract + gather pipeline. Gather offsets are
// masked to [0,60] so garbage lanes stay inside the 64B LUT. Instruction mix
// per byte is identical to a per-row kernel with perfect lane utilization.
#include "kernel_operator.h"
#include "kv_tq_read_bench_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t ROWS_PER_TILE = 128;
constexpr uint32_t PACKED_ROW_BYTES = 96;   // 64B 4bit codes + 2B norm, padded to 32B*3
constexpr uint32_t DENSE_ROW_BYTES = 256;   // 128 x bf16
constexpr uint32_t PACKED_TILE_BYTES = ROWS_PER_TILE * PACKED_ROW_BYTES;  // 12KB
constexpr uint32_t DENSE_TILE_BYTES = ROWS_PER_TILE * DENSE_ROW_BYTES;    // 32KB
} // namespace

class KvTqReadBenchOp {
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR packed, GM_ADDR dense, GM_ADDR out,
        const KvTqReadBenchTilingData *td)
    {
        td_ = td;
        packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(packed),
            static_cast<uint64_t>(td->rows) * PACKED_ROW_BYTES + PACKED_TILE_BYTES);
        denseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dense),
            static_cast<uint64_t>(td->rows) * DENSE_ROW_BYTES + DENSE_TILE_BYTES);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), 1024);
        pipe->InitBuffer(inQueueTile_, 2, DENSE_TILE_BYTES);
        pipe->InitBuffer(bufW_, 64 * sizeof(int32_t));
        pipe->InitBuffer(bufW16_, 64 * sizeof(int32_t));
        pipe->InitBuffer(bufOff_, 64 * sizeof(uint32_t));
        pipe->InitBuffer(bufC_, 64 * sizeof(float));
        pipe->InitBuffer(bufLut_, 16 * sizeof(float));
        pipe->InitBuffer(bufMask16_, 128 * sizeof(uint16_t));
        pipe->InitBuffer(bufAccF_, 64 * sizeof(float));
        pipe->InitBuffer(bufAccI_, 64 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t nAiv = GetBlockNum();
        const uint32_t tiles = (td_->rows + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
        auto accF = bufAccF_.Get<float>();
        auto accI = bufAccI_.Get<int32_t>();
        auto mask16 = bufMask16_.Get<uint16_t>();
        Duplicate(accF, 0.0f, 64);
        Duplicate(accI, 0, 64);
        // alternating u16 mask [0x000F, 0]: keeps low nibble of the low u16
        // half and zeroes the high half, so the u32 view is exactly (w & 0xF).
        // (dav_c220 And supports u16 lanes only.)
        for (uint32_t i = 0; i < 128; ++i) {
            mask16.SetValue(i, (i % 2 == 0) ? 0xF : 0x0);
        }
        PipeBarrier<PIPE_V>();
        if (td_->mode == 2) {
            ProcessDense(tiles, aiv, nAiv);
        } else {
            ProcessPacked(tiles, aiv, nAiv);
        }
        if (td_->mode == 1) {
            DataCopy(outGm_[aiv * 16], accF, 16);
        } else {
            DataCopy(outGm_[aiv * 16], accI.ReinterpretCast<float>(), 16);
        }
    }

private:
    __aicore__ inline void ProcessPacked(uint32_t tiles, uint32_t aiv, uint32_t nAiv)
    {
        auto accF = bufAccF_.Get<float>();
        auto accI = bufAccI_.Get<int32_t>();
        auto mask16 = bufMask16_.Get<uint16_t>();
        auto lut = bufLut_.Get<float>();
        auto w = bufW_.Get<int32_t>();
        auto w16 = bufW16_.Get<uint16_t>();
        auto off = bufOff_.Get<uint32_t>();
        auto c = bufC_.Get<float>();
        Duplicate(lut, 0.5f, 16);  // LUT content irrelevant to timing
        PipeBarrier<PIPE_V>();
        for (uint32_t t = aiv; t < tiles; t += nAiv) {
            LocalTensor<uint8_t> tile = inQueueTile_.AllocTensor<uint8_t>();
            DataCopy(tile, packedGm_[static_cast<uint64_t>(t) * PACKED_TILE_BYTES],
                PACKED_TILE_BYTES);
            inQueueTile_.EnQue(tile);
            tile = inQueueTile_.DeQue<uint8_t>();
            LocalTensor<int32_t> words = tile.ReinterpretCast<int32_t>();
            if (td_->mode == 0) {
                // HBM floor: read + trivial integer accumulate
                for (uint32_t ch = 0; ch < PACKED_TILE_BYTES / 4 / 64; ++ch) {
                    Add(accI, accI, words[ch * 64], 64);
                }
            } else {
                // Full dequant: 8 nibble planes -> gather LUT -> norm mul -> acc
                for (uint32_t ch = 0; ch < PACKED_TILE_BYTES / 4 / 64; ++ch) {
                    LocalTensor<int32_t> wChunk = words[ch * 64];
                    for (uint32_t k = 0; k < 8; ++k) {
                        ShiftRight(w, wChunk, static_cast<int32_t>(4 * k), 64);
                        And(w16, w.ReinterpretCast<uint16_t>(), mask16,
                            static_cast<uint64_t>(128), 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
                        ShiftLeft(off, w16.ReinterpretCast<uint32_t>(),
                            static_cast<uint32_t>(2), 64);
                        Gather(c, lut, off, 0, static_cast<uint64_t>(64), 1, 8);
                        Muls(c, c, 1.0001f, 64);  // per-row norm mul (const proxy)
                        Add(accF, accF, c, 64);
                    }
                }
            }
            inQueueTile_.FreeTensor(tile);
        }
    }

    __aicore__ inline void ProcessDense(uint32_t tiles, uint32_t aiv, uint32_t nAiv)
    {
        auto accI = bufAccI_.Get<int32_t>();
        for (uint32_t t = aiv; t < tiles; t += nAiv) {
            LocalTensor<uint8_t> tile = inQueueTile_.AllocTensor<uint8_t>();
            DataCopy(tile, denseGm_[static_cast<uint64_t>(t) * DENSE_TILE_BYTES],
                DENSE_TILE_BYTES);
            inQueueTile_.EnQue(tile);
            tile = inQueueTile_.DeQue<uint8_t>();
            LocalTensor<int32_t> words = tile.ReinterpretCast<int32_t>();
            for (uint32_t ch = 0; ch < DENSE_TILE_BYTES / 4 / 64; ++ch) {
                Add(accI, accI, words[ch * 64], 64);
            }
            inQueueTile_.FreeTensor(tile);
        }
    }

    const KvTqReadBenchTilingData *td_;
    GlobalTensor<uint8_t> packedGm_;
    GlobalTensor<uint8_t> denseGm_;
    GlobalTensor<float> outGm_;
    TQue<QuePosition::VECIN, 2> inQueueTile_;
    TBuf<TPosition::VECCALC> bufW_;
    TBuf<TPosition::VECCALC> bufW16_;
    TBuf<TPosition::VECCALC> bufOff_;
    TBuf<TPosition::VECCALC> bufC_;
    TBuf<TPosition::VECCALC> bufLut_;
    TBuf<TPosition::VECCALC> bufMask16_;
    TBuf<TPosition::VECCALC> bufAccF_;
    TBuf<TPosition::VECCALC> bufAccI_;
};

extern "C" __global__ __aicore__ void kv_tq_read_bench(GM_ADDR packed, GM_ADDR dense,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KvTqReadBenchTilingData);
    // Same manual-copy workaround as KvCacheTurboQuant / KvCacheInt8Store.
    KvTqReadBenchTilingData tilingData;
    {
        const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&tilingData);
        for (uint32_t i = 0; i < sizeof(KvTqReadBenchTilingData) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
    }
    TPipe pipe;
    KvTqReadBenchOp op;
    op.Init(&pipe, packed, dense, out, &tilingData);
    op.Process();
}
