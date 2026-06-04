// FusedKvNormRopeSwaCacheCustom v2.6 (cache-only): DSV4-DSA SWA-kv prep fusion.
//
// Optimizations over v1:
//   1. All-fp32 pipeline — NormKv result stays in fp32 xfBuf_; RopeTo works on fp32 directly;
//      only ONE Cast back to T at the very end (before HBM writes). Saves 3 Cast ops/token.
//   2. Double-buffered cos/sin — preloads token T+1 cos/sin into the alternate buffer while
//      token T's V-pipe compute runs; MTE loads overlap with Vector arithmetic.
//   3. Gamma pre-cast to fp32 — loaded once per core, no per-token Cast for the multiply.
//   4. Reduced PipeBarriers — PIPE_V within V-only sequences (norm/RoPE), PIPE_ALL only at
//      V→MTE transitions (after final Cast, before cache write).
//   5. BlockDim=40 always — uses all AIV cores regardless of nt (idle cores early-exit).
//      Tiling sets BlockDim=aivNum; kernel derives per-core range from GetBlockIdx/GetBlockNum.
//   6. Invalid-slot guard retained from v1 (skip cache write when block/offset < 0, graph-safe).
//   7. Invalid-slot early-skip: padding rows skip norm/RoPE/Cast/cache entirely.
//   8. Cache-only output: skip the unused kv_out GM write.
//
// Expected decode (nt=32) improvement: ~2× over v1 (Cast/buffer/barrier savings).
// Prefill (nt=1024) improvement: ~1.5-2× (41 cores all working, fewer casts/barriers).

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class KernelKvNormRopeSwa {
public:
    __aicore__ inline KernelKvNormRopeSwa() {}

    __aicore__ inline void Init(GM_ADDR kvIn, GM_ADDR gamma, GM_ADDR coss, GM_ADDR sinn,
                                GM_ADDR slotMapping, GM_ADDR kvCacheIn, GM_ADDR kvCacheOut,
                                uint32_t numTokens, uint32_t headDim, uint32_t nopeDim,
                                uint32_t ropeDim, uint32_t blockSize, uint32_t numBlocks,
                                float eps, float headDimF) {
        numTokens_ = numTokens; headDim_ = headDim; nopeDim_ = nopeDim; ropeDim_ = ropeDim;
        blockSize_ = blockSize; numBlocks_ = numBlocks; eps_ = eps;
        headDimF_ = headDimF;

        // Compute per-core token range from actual BlockDim (set by host tiling to aivNum=40).
        uint32_t coreIdx = GetBlockIdx();
        uint32_t totalCores = GetBlockNum();
        if (totalCores == 0) totalCores = 1;
        uint32_t tasksPerCore = (numTokens_ + totalCores - 1u) / totalCores;
        startTask_ = coreIdx * tasksPerCore;
        endTask_   = startTask_ + tasksPerCore;
        if (endTask_ > numTokens_) endTask_ = numTokens_;

        int64_t kvLen    = (int64_t)numTokens_ * headDim_;
        int64_t csLen    = (int64_t)numTokens_ * ropeDim_;
        int64_t cacheLen = (int64_t)numBlocks_ * blockSize_ * headDim_;

        kvInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(kvIn), kvLen);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), headDim_);
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coss), csLen);
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sinn), csLen);
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMapping), (int64_t)numTokens_ * 2);
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(kvCacheOut), cacheLen);
        (void)kvCacheIn;

        // Buffers:
        pipe_.InitBuffer(kvBuf_,          headDim_ * sizeof(T));         // input load, final output (T)
        pipe_.InitBuffer(gammaBuf_,       headDim_ * sizeof(T));         // gamma T (loaded once)
        pipe_.InitBuffer(gammaFp32Buf_,   headDim_ * sizeof(float));     // gamma fp32 (pre-cast once)
        pipe_.InitBuffer(xfBuf_,          headDim_ * sizeof(float));     // fp32 work: norm → rope → final
        pipe_.InitBuffer(tmpfBuf_,        headDim_ * sizeof(float));     // squares, scratch
        pipe_.InitBuffer(rdBuf_,          32 * sizeof(float));           // ReduceSum result
        pipe_.InitBuffer(rwBuf_,          headDim_ * sizeof(float));     // ReduceSum workspace
        // Interleave RoPE scratch:
        pipe_.InitBuffer(swapIdxBuf_,     ropeDim_ * sizeof(uint32_t));
        pipe_.InitBuffer(signvBuf_,       ropeDim_ * sizeof(float));
        pipe_.InitBuffer(swapBuf_,        ropeDim_ * sizeof(float));
        pipe_.InitBuffer(rotBuf_,         ropeDim_ * sizeof(float));
        // Double-buffered cos/sin (overlap MTE loads with V compute):
        pipe_.InitBuffer(cfBuf0_,         ropeDim_ * sizeof(float));
        pipe_.InitBuffer(sfBuf0_,         ropeDim_ * sizeof(float));
        pipe_.InitBuffer(cfBuf1_,         ropeDim_ * sizeof(float));
        pipe_.InitBuffer(sfBuf1_,         ropeDim_ * sizeof(float));
        // Slot mapping resident in UB (graph-safe, 32B-rounded):
        pipe_.InitBuffer(slotBuf_,        ((numTokens_ * 2u + 7u) / 8u * 8u) * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (startTask_ >= endTask_) return;

        // 1. One-time: load gamma, cast to fp32.
        LocalTensor<T>     gammaT  = gammaBuf_.Get<T>();
        LocalTensor<float> gammaFp = gammaFp32Buf_.Get<float>();
        DataCopy(gammaT, gammaGm_, headDim_);
        PipeBarrier<PIPE_ALL>();   // GM -> UB must complete before V reads gammaT.
        Cast(gammaFp, gammaT, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_ALL>();

        // 2. One-time: build interleave RoPE gather tables.
        BuildGatherTables();

        // 3. One-time: load slot_mapping into UB (graph-safe DataCopyPad).
        LocalTensor<int32_t> slotL = slotBuf_.Get<int32_t>();
        {
            DataCopyExtParams slotEp;
            slotEp.blockCount = 1;
            slotEp.blockLen   = numTokens_ * 2u * sizeof(int32_t);
            slotEp.srcStride  = 0; slotEp.dstStride = 0;
            DataCopyPadExtParams<int32_t> slotPad{false, 0, 0, 0};
            DataCopyPad(slotL, slotGm_, slotEp, slotPad);
        }
        PipeBarrier<PIPE_ALL>();

        // 4. Double-buffered token loop.
        uint32_t tok = startTask_;
        bool useBuf0 = true;

        // Preload cos/sin for the first token.
        {
            LocalTensor<float> c0 = cfBuf0_.Get<float>();
            LocalTensor<float> s0 = sfBuf0_.Get<float>();
            DataCopy(c0, cosGm_[(uint64_t)tok * ropeDim_], ropeDim_);
            DataCopy(s0, sinGm_[(uint64_t)tok * ropeDim_], ropeDim_);
        }
        PipeBarrier<PIPE_ALL>();   // first cos/sin ready

        for (; tok < endTask_; ++tok) {
            LocalTensor<float> cfCurr = useBuf0 ? cfBuf0_.Get<float>() : cfBuf1_.Get<float>();
            LocalTensor<float> sfCurr = useBuf0 ? sfBuf0_.Get<float>() : sfBuf1_.Get<float>();

            // Preload NEXT token's cos/sin into the OTHER buffer while current token computes.
            uint32_t next = tok + 1u;
            if (next < endTask_) {
                LocalTensor<float> cfNext = useBuf0 ? cfBuf1_.Get<float>() : cfBuf0_.Get<float>();
                LocalTensor<float> sfNext = useBuf0 ? sfBuf1_.Get<float>() : sfBuf0_.Get<float>();
                DataCopy(cfNext, cosGm_[(uint64_t)next * ropeDim_], ropeDim_);
                DataCopy(sfNext, sinGm_[(uint64_t)next * ropeDim_], ropeDim_);
                // No barrier — these MTE loads run in background. ProcessToken's final
                // PipeBarrier<PIPE_ALL> will sync them before the next iteration's use.
            }
            useBuf0 = !useBuf0;

            // ---- process token `tok` ----
            // slot
            int32_t block  = slotL.GetValue(tok * 2u);
            int32_t offset = slotL.GetValue(tok * 2u + 1u);
            bool validSlot = (block >= 0 && offset >= 0 &&
                              (uint32_t)block < numBlocks_ && (uint32_t)offset < blockSize_);
            if (!validSlot) {
                PipeBarrier<PIPE_ALL>();  // drain next cos/sin preload before the next iteration.
                continue;
            }

            // Norm + RoPE (all fp32, then one Cast to T)
            NormKvFp32(tok, gammaFp);
            RopeToFp32(cfCurr, sfCurr);
            // Final Cast: fp32 xfBuf_ → T kvBuf_
            {
                LocalTensor<T> kvOut = kvBuf_.Get<T>();
                LocalTensor<float> xf = xfBuf_.Get<float>();
                Cast(kvOut, xf, RoundMode::CAST_RINT, headDim_);
            }
            PipeBarrier<PIPE_ALL>();  // V Cast -> MTE3 writes

            // Write cache (MTE3)
            {
                LocalTensor<T> kvOut = kvBuf_.Get<T>();
                uint64_t dst = ((uint64_t)(uint32_t)block * blockSize_ + (uint32_t)offset) * headDim_;
                DataCopy(cacheGm_[dst], kvOut, headDim_);
            }
            PipeBarrier<PIPE_ALL>();  // MTE writes done + preloaded cos/sin ready for next iteration
        }
    }

private:
    // Build swap permutation + sign vector (same as v1).
    __aicore__ inline void BuildGatherTables() {
        LocalTensor<uint32_t> idx = swapIdxBuf_.Get<uint32_t>();
        LocalTensor<float>    sg  = signvBuf_.Get<float>();
        for (uint32_t i = 0; i < ropeDim_; i += 2u) {
            idx.SetValue(i,      (i + 1u) * 4u);
            idx.SetValue(i + 1u, i * 4u);
            sg.SetValue(i,      -1.0f);
            sg.SetValue(i + 1u,  1.0f);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // RMSNorm over full headDim, *gamma, result in xfBuf_ (fp32). Loads kvIn[token] → kvBuf_ (T).
    __aicore__ inline void NormKvFp32(uint32_t token, LocalTensor<float>& gammaFp) {
        LocalTensor<T>     kvT  = kvBuf_.Get<T>();
        LocalTensor<float> xf   = xfBuf_.Get<float>();
        LocalTensor<float> tmpf = tmpfBuf_.Get<float>();
        LocalTensor<float> rd   = rdBuf_.Get<float>();
        LocalTensor<float> rw   = rwBuf_.Get<float>();

        DataCopy(kvT, kvInGm_[(uint64_t)token * headDim_], headDim_);
        PipeBarrier<PIPE_ALL>();   // MTE2 read -> V Cast
        Cast(xf, kvT, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        Mul(tmpf, xf, xf, headDim_);     // squares
        PipeBarrier<PIPE_V>();
        ReduceSum(rd, tmpf, rw, headDim_);
        PipeBarrier<PIPE_ALL>();         // drain before scalar read
        float sumSq  = rd.GetValue(0);
        float invRms = 1.0f / sqrt(sumSq / headDimF_ + eps_);
        Muls(xf, xf, invRms, headDim_);
        PipeBarrier<PIPE_V>();
        Mul(xf, xf, gammaFp, headDim_);  // gamma already fp32 (pre-cast once)
        PipeBarrier<PIPE_V>();
        // Result: xfBuf_[0:headDim_] holds normed*gamma fp32. RopeTo works on [nopeDim_:] next.
    }

    // Interleave RoPE on xfBuf_[nopeDim_:nopeDim_+ropeDim_] in place (fp32).
    __aicore__ inline void RopeToFp32(LocalTensor<float>& cf, LocalTensor<float>& sf) {
        LocalTensor<float>    xf  = xfBuf_.Get<float>();
        LocalTensor<float>    sw  = swapBuf_.Get<float>();
        LocalTensor<float>    rot = rotBuf_.Get<float>();
        LocalTensor<uint32_t> idx = swapIdxBuf_.Get<uint32_t>();
        LocalTensor<float>    sg  = signvBuf_.Get<float>();

        Gather(sw, xf[nopeDim_], idx, (uint32_t)0, ropeDim_);
        PipeBarrier<PIPE_V>();
        Mul(rot, sw, sg, ropeDim_);
        PipeBarrier<PIPE_V>();
        Mul(xf[nopeDim_], xf[nopeDim_], cf, ropeDim_);
        PipeBarrier<PIPE_V>();
        Mul(rot, rot, sf, ropeDim_);
        PipeBarrier<PIPE_V>();
        Add(xf[nopeDim_], xf[nopeDim_], rot, ropeDim_);
        PipeBarrier<PIPE_V>();
        // xfBuf_[nopeDim_:] now holds the roped tail. Leading [0:nopeDim_] is normed passthrough.
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> kvBuf_, gammaBuf_, gammaFp32Buf_, xfBuf_, tmpfBuf_, rdBuf_, rwBuf_;
    TBuf<TPosition::VECCALC> swapIdxBuf_, signvBuf_, swapBuf_, rotBuf_;
    TBuf<TPosition::VECCALC> cfBuf0_, sfBuf0_, cfBuf1_, sfBuf1_;
    TBuf<TPosition::VECCALC> slotBuf_;

    GlobalTensor<T> kvInGm_, gammaGm_, cacheGm_;
    GlobalTensor<float> cosGm_, sinGm_;
    GlobalTensor<int32_t> slotGm_;

    uint32_t numTokens_, headDim_, nopeDim_, ropeDim_, blockSize_, numBlocks_;
    float eps_, headDimF_;
    uint32_t startTask_, endTask_;
};

extern "C" __global__ __aicore__ void fused_kv_norm_rope_swa_cache_custom(
    GM_ADDR kv_in, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR slot_mapping,
    GM_ADDR kv_cache_in, GM_ADDR kv_cache_out,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    GET_TILING_DATA(td, tilingGm);
    if (TILING_KEY_IS(1)) {
        if (td.isBf16 == 1u) {
            KernelKvNormRopeSwa<bfloat16_t> op;
            op.Init(kv_in, gamma, cos, sin, slot_mapping, kv_cache_in, kv_cache_out,
                    td.numTokens, td.headDim, td.nopeDim, td.ropeDim,
                    td.blockSize, td.numBlocks, td.eps, td.headDimF);
            op.Process();
        } else {
            KernelKvNormRopeSwa<half> op;
            op.Init(kv_in, gamma, cos, sin, slot_mapping, kv_cache_in, kv_cache_out,
                    td.numTokens, td.headDim, td.nopeDim, td.ropeDim,
                    td.blockSize, td.numBlocks, td.eps, td.headDimF);
            op.Process();
        }
    }
}
