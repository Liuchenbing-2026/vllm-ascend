// Fused per-(kv_head, channel) affine int8 quantization + NZ-layout scatter
// into the paged KV cache. Canonical TQue style (EnQue/DeQue handles all
// pipe sync). Read path stays native FIA antiquant.
#include "kernel_operator.h"
#include "kv_cache_int8_store_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_D = 512;    // head_dim upper bound
constexpr uint32_t CHUNK = 32;     // NZ inner chunk (int8 32B)
} // namespace

class KvI8Store {
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR x, GM_ADDR scale, GM_ADDR slots, GM_ADDR cache,
        const KvCacheInt8StoreTilingData *td)
    {
        td_ = td;
        const uint64_t rows = static_cast<uint64_t>(td->numTokens) * td->numKvHeads;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x), rows * td->headDim);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale),
            static_cast<uint64_t>(td->numKvHeads) * td->headDim);
        slotPtr_ = reinterpret_cast<const __gm__ int32_t *>(slots);
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(cache),
            static_cast<uint64_t>(td->numBlocks) * td->blockSize * td->numKvHeads * td->headDim);
        pipe->InitBuffer(inQueueX_, 1, MAX_D * sizeof(bfloat16_t));
        pipe->InitBuffer(inQueueS_, 1, MAX_D * sizeof(float));
        pipe->InitBuffer(outQueueQ_, 1, MAX_D * sizeof(int8_t));
        pipe->InitBuffer(bufF_, MAX_D * sizeof(float));
        pipe->InitBuffer(bufH_, MAX_D * sizeof(half));
        pipe->InitBuffer(bufS32_, MAX_D * sizeof(int32_t));
            }

    __aicore__ inline void Process()
    {
        const uint32_t totalRows = td_->numTokens * td_->numKvHeads;
        const uint32_t aiv = GetBlockIdx();
        const uint32_t nAiv = GetBlockNum();
        if (nAiv == 0 || td_->numTokens == 0) {
            return;
        }

        const uint32_t d = td_->headDim;
        const uint32_t d32 = d / CHUNK;
        for (uint32_t r = aiv; r < totalRows; r += nAiv) {
            const uint32_t n = r / td_->numKvHeads;
            const uint32_t h = r - n * td_->numKvHeads;
            const uint32_t slot = static_cast<uint32_t>(slotPtr_[n]);
            const uint32_t blk = slot / td_->blockSize;
            const uint32_t off = slot - blk * td_->blockSize;

            LocalTensor<bfloat16_t> xUb = inQueueX_.AllocTensor<bfloat16_t>();
            LocalTensor<float> sUb = inQueueS_.AllocTensor<float>();
            DataCopy(xUb, xGm_[static_cast<uint64_t>(r) * d], d);
            DataCopy(sUb, scaleGm_[static_cast<uint64_t>(h) * d], d);
            inQueueX_.EnQue(xUb);
            inQueueS_.EnQue(sUb);

            xUb = inQueueX_.DeQue<bfloat16_t>();
            sUb = inQueueS_.DeQue<float>();
            LocalTensor<int8_t> qUb = outQueueQ_.AllocTensor<int8_t>();
            LocalTensor<float> fUb = bufF_.Get<float>();
            LocalTensor<half> hUb = bufH_.Get<half>();
            LocalTensor<int32_t> s32Ub = bufS32_.Get<int32_t>();
            Cast(fUb, xUb, RoundMode::CAST_NONE, d);            // bf16 -> f32
            PipeBarrier<PIPE_V>();
            Div(fUb, fUb, sUb, d);
            PipeBarrier<PIPE_V>();
            Maxs(fUb, fUb, -127.0f, d);
            PipeBarrier<PIPE_V>();
            Mins(fUb, fUb, 127.0f, d);
            PipeBarrier<PIPE_V>();
            // round-half-even via f32 -> s32 (CAST_RINT, proven in
            // KvCacheTurboQuant) -> f32 round-trip; values are clamped to
            // [-127,127] so both directions are exact. f16/s8 casts below are
            // then exact integer conversions (CAST_NONE truncates on dav_c220,
            // which is exact for integral inputs).
            Cast<int32_t, float>(s32Ub, fUb, RoundMode::CAST_RINT, d);
            PipeBarrier<PIPE_V>();
            Cast(fUb, s32Ub, RoundMode::CAST_NONE, d);
            PipeBarrier<PIPE_V>();
            Cast(hUb, fUb, RoundMode::CAST_NONE, d);
            PipeBarrier<PIPE_V>();
            Cast<int8_t, half>(qUb, hUb, RoundMode::CAST_NONE, d);
            inQueueX_.FreeTensor(xUb);
            inQueueS_.FreeTensor(sUb);
            outQueueQ_.EnQue(qUb);

            qUb = outQueueQ_.DeQue<int8_t>();
            // NZ target: cache viewed as [numBlocks, H, d/32, blockSize, 32]
            uint64_t base =
                ((static_cast<uint64_t>(blk) * td_->numKvHeads + h) * d32) * td_->blockSize + off;
            base *= CHUNK;
            const uint64_t strideD1 = static_cast<uint64_t>(td_->blockSize) * CHUNK;
            for (uint32_t d1 = 0; d1 < d32; ++d1) {
                DataCopy(cacheGm_[base + static_cast<uint64_t>(d1) * strideD1], qUb[d1 * CHUNK], CHUNK);
            }
            outQueueQ_.FreeTensor(qUb);
        }
    }

private:
    const KvCacheInt8StoreTilingData *td_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int8_t> cacheGm_;
    const __gm__ int32_t *slotPtr_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueS_;
    TQue<QuePosition::VECOUT, 1> outQueueQ_;
    TBuf<TPosition::VECCALC> bufF_;
    TBuf<TPosition::VECCALC> bufH_;
    TBuf<TPosition::VECCALC> bufS32_;
};

extern "C" __global__ __aicore__ void kv_cache_int8_store(GM_ADDR x, GM_ADDR scale, GM_ADDR slots,
    GM_ADDR cache, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KvCacheInt8StoreTilingData);
    // Same manual-copy workaround as KvCacheTurboQuant: do not trust the
    // compiler-generated GET_TILING_DATA_WITH_STRUCT.
    KvCacheInt8StoreTilingData tilingData;
    {
        const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&tilingData);
        for (uint32_t i = 0; i < sizeof(KvCacheInt8StoreTilingData) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
    }
    TPipe pipe;
    KvI8Store op;
    op.Init(&pipe, x, scale, slots, cache, &tilingData);
    op.Process();
}