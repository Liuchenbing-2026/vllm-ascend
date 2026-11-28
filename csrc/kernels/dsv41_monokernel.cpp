/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "types.h"

#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)

namespace {

constexpr uint32_t kMonoTileK = 1024;
constexpr uint32_t kMonoTileM = 8;
constexpr uint32_t kRowVec = 8;
constexpr uint32_t kRowBlk = 16;
constexpr uint32_t kMonoSyncInts = 1024;
constexpr uint32_t kHcMult = 4;
constexpr uint32_t kHcMixDim = 24;
constexpr uint32_t kHcCombDim = 16;
constexpr uint32_t kMonoAlign = 64;
constexpr uint32_t kAccSlotFloats = kMonoTileM * kRowBlk;
constexpr uint32_t kUbSkewWf = 16;
constexpr uint32_t kUbSkewProd = 32;
constexpr uint32_t kExpGrpStride = 18;

__aicore__ inline uint64_t MonoAlign64(uint64_t val)
{
    return ((val + kMonoAlign - 1) / kMonoAlign) * kMonoAlign;
}

__aicore__ inline uint32_t MonoVecAlign8(uint32_t val)
{
    return ((val + 7U) / 8U) * 8U;
}

__aicore__ inline uint32_t MonoPadReduceLen(uint32_t len)
{
    return (len <= 512U) ? 512U : kMonoTileK;
}

struct MonoMeta {
    uint32_t mTokens;
    uint32_t hidden;
    uint32_t qLora;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t nopeDim;
    uint32_t ropeDim;
    uint32_t oGroups;
    uint32_t oLora;
    uint32_t windowSize;
    uint32_t blockSize;
    uint32_t maxBlocks;
    uint32_t nExperts;
    uint32_t topK;
    uint32_t sharedInter;
    uint32_t routedInter;
    uint32_t sinkhornIters;
    uint32_t hasEBias;
    uint32_t useSigmoid;
    uint32_t renorm;
    uint32_t hasSink;
    float normEps;
    float hcEps;
    float softmaxScale;
    float routedScale;
    float swigluLimit;
};

class DSV41MonoLayerKernel {
public:
    __aicore__ inline DSV41MonoLayerKernel() {}

    __aicore__ inline void Init(
        __gm__ void *resIn, __gm__ void *preMixIn, __gm__ void *positions,
        __gm__ void *cosIn, __gm__ void *sinIn, __gm__ void *slotMapping,
        __gm__ void *blockTable, __gm__ void *seqLens,
        __gm__ void *resOut, __gm__ void *preMixOut,
        __gm__ void *inputLnW, __gm__ void *hcAttnFn, __gm__ void *hcAttnScale, __gm__ void *hcAttnBase,
        __gm__ void *wqA, __gm__ void *qNormW, __gm__ void *wqB, __gm__ void *wkv, __gm__ void *kvNormW,
        __gm__ void *attnSink, __gm__ void *woA, __gm__ void *woB,
        __gm__ void *postAttnLnW, __gm__ void *hcFfnFn, __gm__ void *hcFfnScale, __gm__ void *hcFfnBase,
        __gm__ void *gateW, __gm__ void *eBias, __gm__ void *sharedW13, __gm__ void *sharedW2,
        __gm__ void *routedW13, __gm__ void *routedW2,
        __gm__ void *swaKvCache, __gm__ void *workspace,
        __gm__ void *metaI64, __gm__ void *metaF32,
        AscendC::TPipe *pipePtr)
    {
        pipe_ = pipePtr;
        blkIdx_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        blkNum_ = static_cast<uint32_t>(AscendC::GetBlockNum());

        AscendC::GlobalTensor<int64_t> miGm;
        AscendC::GlobalTensor<float> mfGm;
        miGm.SetGlobalBuffer((__gm__ int64_t *)metaI64);
        mfGm.SetGlobalBuffer((__gm__ float *)metaF32);

        meta_.mTokens = static_cast<uint32_t>(miGm.GetValue(0));
        meta_.hidden = static_cast<uint32_t>(miGm.GetValue(1));
        meta_.qLora = static_cast<uint32_t>(miGm.GetValue(2));
        meta_.nHeads = static_cast<uint32_t>(miGm.GetValue(3));
        meta_.headDim = static_cast<uint32_t>(miGm.GetValue(4));
        meta_.nopeDim = static_cast<uint32_t>(miGm.GetValue(5));
        meta_.ropeDim = static_cast<uint32_t>(miGm.GetValue(6));
        meta_.oGroups = static_cast<uint32_t>(miGm.GetValue(7));
        meta_.oLora = static_cast<uint32_t>(miGm.GetValue(8));
        meta_.windowSize = static_cast<uint32_t>(miGm.GetValue(9));
        meta_.blockSize = static_cast<uint32_t>(miGm.GetValue(10));
        meta_.maxBlocks = static_cast<uint32_t>(miGm.GetValue(11));
        meta_.nExperts = static_cast<uint32_t>(miGm.GetValue(12));
        meta_.topK = static_cast<uint32_t>(miGm.GetValue(13));
        meta_.sharedInter = static_cast<uint32_t>(miGm.GetValue(14));
        meta_.routedInter = static_cast<uint32_t>(miGm.GetValue(15));
        meta_.sinkhornIters = static_cast<uint32_t>(miGm.GetValue(16));
        meta_.hasEBias = static_cast<uint32_t>(miGm.GetValue(17));
        meta_.useSigmoid = static_cast<uint32_t>(miGm.GetValue(18));
        meta_.renorm = static_cast<uint32_t>(miGm.GetValue(19));
        meta_.hasSink = static_cast<uint32_t>(miGm.GetValue(20));

        meta_.normEps = mfGm.GetValue(0);
        meta_.hcEps = mfGm.GetValue(1);
        meta_.softmaxScale = mfGm.GetValue(2);
        meta_.routedScale = mfGm.GetValue(3);
        meta_.swigluLimit = mfGm.GetValue(4);

        resInGm_.SetGlobalBuffer((__gm__ bfloat16_t *)resIn);
        preMixInGm_.SetGlobalBuffer((__gm__ float *)preMixIn);
        posGm_.SetGlobalBuffer((__gm__ int64_t *)positions);
        cosGm_.SetGlobalBuffer((__gm__ float *)cosIn);
        sinGm_.SetGlobalBuffer((__gm__ float *)sinIn);
        slotGm_.SetGlobalBuffer((__gm__ int64_t *)slotMapping);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        seqLensGm_.SetGlobalBuffer((__gm__ int32_t *)seqLens);
        resOutGm_.SetGlobalBuffer((__gm__ bfloat16_t *)resOut);
        preMixOutGm_.SetGlobalBuffer((__gm__ float *)preMixOut);

        inputLnWGm_.SetGlobalBuffer((__gm__ bfloat16_t *)inputLnW);
        hcAttnFnGm_.SetGlobalBuffer((__gm__ float *)hcAttnFn);
        hcAttnScaleGm_.SetGlobalBuffer((__gm__ float *)hcAttnScale);
        hcAttnBaseGm_.SetGlobalBuffer((__gm__ float *)hcAttnBase);
        wqAGm_.SetGlobalBuffer((__gm__ bfloat16_t *)wqA);
        qNormWGm_.SetGlobalBuffer((__gm__ bfloat16_t *)qNormW);
        wqBGm_.SetGlobalBuffer((__gm__ bfloat16_t *)wqB);
        wkvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)wkv);
        kvNormWGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kvNormW);
        attnSinkGm_.SetGlobalBuffer((__gm__ float *)attnSink);
        woAGm_.SetGlobalBuffer((__gm__ bfloat16_t *)woA);
        woBGm_.SetGlobalBuffer((__gm__ bfloat16_t *)woB);
        postAttnLnWGm_.SetGlobalBuffer((__gm__ bfloat16_t *)postAttnLnW);
        hcFfnFnGm_.SetGlobalBuffer((__gm__ float *)hcFfnFn);
        hcFfnScaleGm_.SetGlobalBuffer((__gm__ float *)hcFfnScale);
        hcFfnBaseGm_.SetGlobalBuffer((__gm__ float *)hcFfnBase);
        gateWGm_.SetGlobalBuffer((__gm__ bfloat16_t *)gateW);
        eBiasGm_.SetGlobalBuffer((__gm__ float *)eBias);
        sharedW13Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)sharedW13);
        sharedW2Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)sharedW2);
        routedW13Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)routedW13);
        routedW2Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)routedW2);
        swaKvCacheGm_.SetGlobalBuffer((__gm__ bfloat16_t *)swaKvCache);

        SetupWorkspace(workspace);
        SetupPipeBuffers();
    }

    __aicore__ inline void Process()
    {
        if (meta_.mTokens == 0) {
            return;
        }
        Stage1And2QkvRopeCache();
        BarrierAllCores();
        Stage3FlashSwaAndOProjPost();
        BarrierAllCores();
        Stage4And5MoEFinalPost();
    }

private:
    __aicore__ inline void BarrierAllCores()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (blkNum_ > 1) {
            AscendC::LocalTensor<int32_t> syncUb = bufSync_.Get<int32_t>();
            AscendC::SyncAll<true>(wsSync_, syncUb, static_cast<int32_t>(blkNum_));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SyncVecToScalar()
    {
        event_t evVS = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evVS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVS);
    }

    __aicore__ inline __gm__ float *AllocWsF(__gm__ float *base, uint64_t &off, uint64_t nF)
    {
        __gm__ float *p = base + off;
        off += MonoAlign64(nF);
        return p;
    }

    __aicore__ inline __gm__ bfloat16_t *AllocWsB(__gm__ float *base, uint64_t &off, uint64_t nB)
    {
        __gm__ bfloat16_t *p = (__gm__ bfloat16_t *)(base + off);
        off += MonoAlign64((nB + 1) / 2);
        return p;
    }

    __aicore__ inline __gm__ int32_t *AllocWsI(__gm__ float *base, uint64_t &off, uint64_t nI)
    {
        __gm__ int32_t *p = (__gm__ int32_t *)(base + off);
        off += MonoAlign64(nI);
        return p;
    }

    __aicore__ inline void SetupWorkspace(__gm__ void *workspace)
    {
        __gm__ float *base = (__gm__ float *)workspace;
        uint64_t M = meta_.mTokens;
        uint64_t D = meta_.hidden;
        uint64_t off = 0;

        wsSync_.SetGlobalBuffer(AllocWsI(base, off, kMonoSyncInts));
        wsXNorm_.SetGlobalBuffer(AllocWsB(base, off, M * D));
        wsResMid_.SetGlobalBuffer(AllocWsB(base, off, M * kHcMult * D));
        wsInvRms_.SetGlobalBuffer(AllocWsF(base, off, M));
        wsMixes_.SetGlobalBuffer(AllocWsF(base, off, M * 32));
        wsAttnPre_.SetGlobalBuffer(AllocWsF(base, off, M * 8));
        wsAttnPost_.SetGlobalBuffer(AllocWsF(base, off, M * 8));
        wsAttnComb_.SetGlobalBuffer(AllocWsF(base, off, M * kHcCombDim));
        wsQA_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.qLora));
        wsQR_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.qLora));
        wsKV_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.headDim));
        wsQ_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.nHeads * meta_.headDim));
        wsAttnOut_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.nHeads * meta_.headDim));
        wsOA_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.oGroups * meta_.oLora));
        wsFfnPre_.SetGlobalBuffer(AllocWsF(base, off, M * 8));
        wsFfnPost_.SetGlobalBuffer(AllocWsF(base, off, M * 8));
        wsFfnComb_.SetGlobalBuffer(AllocWsF(base, off, M * kHcCombDim));
        wsRouterLogits_.SetGlobalBuffer(AllocWsF(base, off, M * MonoAlign64(meta_.nExperts)));
        wsTopkIds_.SetGlobalBuffer(AllocWsI(base, off, M * MonoAlign64(meta_.topK)));
        wsTopkW_.SetGlobalBuffer(AllocWsF(base, off, M * MonoAlign64(meta_.topK)));
        wsSharedAct_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.sharedInter));
        wsSharedOut_.SetGlobalBuffer(AllocWsB(base, off, M * D));
        wsRoutedAct_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.topK * meta_.routedInter));
        wsRoutedOut_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.topK * D));
        wsExpMeta_.SetGlobalBuffer(AllocWsI(base, off, 64 + M * meta_.topK * kExpGrpStride));
    }

    __aicore__ inline void SetupPipeBuffers()
    {
        uint32_t tileBytes = kMonoTileK * sizeof(float);
        pipe_->InitBuffer(inQueA_, 2, tileBytes);
        pipe_->InitBuffer(inQueB_, 2, kRowVec * kMonoTileK * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQue_, 1, tileBytes);

        pipe_->InitBuffer(bufXBatch_, kMonoTileM * tileBytes);
        pipe_->InitBuffer(bufWF32_, (kRowVec * kMonoTileK + kUbSkewWf) * sizeof(float));
        pipe_->InitBuffer(bufProd_, (kRowVec * kMonoTileK + kUbSkewProd) * sizeof(float));
        pipe_->InitBuffer(bufAux0_, tileBytes);
        pipe_->InitBuffer(bufAux1_, tileBytes);
        pipe_->InitBuffer(bufWork_, kRowVec * 64 * sizeof(float));
        pipe_->InitBuffer(bufSmall_, 256 * sizeof(float));
        pipe_->InitBuffer(bufSync_, kMonoSyncInts * sizeof(int32_t));
        pipe_->InitBuffer(bufAcc_, 3 * kAccSlotFloats * sizeof(float));
    }

    __aicore__ inline float ScalarExp(float x)
    {
        AscendC::LocalTensor<float> sm = bufSmall_.Get<float>();
        sm.SetValue(0, x);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(sm, sm, 8);
        SyncVecToScalar();
        return sm.GetValue(0);
    }

    __aicore__ inline float ScalarRsqrt(float x)
    {
        AscendC::LocalTensor<float> sm = bufSmall_.Get<float>();
        sm.SetValue(0, x);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sqrt(sm, sm, 8);
        SyncVecToScalar();
        return 1.0f / sm.GetValue(0);
    }

    template <typename T, typename TQue>
    __aicore__ inline AscendC::LocalTensor<T> CopyInTile(
        TQue &que,
        const AscendC::GlobalTensor<T> &gm,
        uint64_t offset,
        uint32_t count)
    {
        AscendC::LocalTensor<T> ub = que.template AllocTensor<T>();
        if ((offset * sizeof(T)) % 32 == 0 && (count * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(ub, gm[offset], count);
        } else {
            uint16_t bytes = static_cast<uint16_t>(count * sizeof(T));
            AscendC::DataCopyPad(ub, gm[offset], {1, bytes, 0, 0}, {false, 0, 0, 0});
        }
        que.EnQue(ub);
        return que.template DeQue<T>();
    }

    template <typename T>
    __aicore__ inline void CopyOutTile(
        const AscendC::GlobalTensor<T> &gm,
        uint64_t offset,
        AscendC::LocalTensor<T> &ub,
        uint32_t count)
    {
        if ((offset * sizeof(T)) % 32 == 0 && (count * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(gm[offset], ub, count);
        } else {
            uint16_t bytes = static_cast<uint16_t>(count * sizeof(T));
            AscendC::DataCopyPad(gm[offset], ub, {1, bytes, 0, 0});
        }
        outQue_.FreeTensor(ub);
    }

    __aicore__ inline void EnqueWeightGroup(
        const AscendC::GlobalTensor<bfloat16_t> &wGm,
        uint64_t wBase,
        uint32_t K,
        uint32_t k0,
        uint32_t rowStart,
        uint32_t grpRows,
        uint32_t len,
        uint32_t padLen)
    {
        AscendC::LocalTensor<bfloat16_t> wb = inQueB_.AllocTensor<bfloat16_t>();
        uint64_t startOff = wBase + static_cast<uint64_t>(rowStart) * K + k0;
        uint32_t rowGap = K - len;
        if ((startOff % 16U == 0) && (len % 16U == 0) && (rowGap % 16U == 0) &&
            (padLen == len) && ((rowGap / 16U) <= 65535U)) {
            AscendC::DataCopyParams params;
            params.blockCount = static_cast<uint16_t>(grpRows);
            params.blockLen = static_cast<uint16_t>(len / 16U);
            params.srcStride = static_cast<uint16_t>(rowGap / 16U);
            params.dstStride = 0;
            AscendC::DataCopy(wb, wGm[startOff], params);
        } else {
            uint16_t bytes = static_cast<uint16_t>(len * sizeof(bfloat16_t));
            for (uint32_t r = 0; r < grpRows; ++r) {
                AscendC::DataCopyPad(wb[r * padLen], wGm[startOff + static_cast<uint64_t>(r) * K],
                                     {1, bytes, 0, 0}, {false, 0, 0, 0});
            }
        }
        inQueB_.EnQue(wb);
    }

    // Pure vector GEMV accumulation into UB accumulator accUb[m * kRowBlk + r], zero SyncVecToScalar().
    // Uses 8-row (up to 16 KB) double-buffered burst DataCopy, bank-skewed UB buffers, and M<=8 weight reuse.
    __aicore__ inline void GemmAccumBf16Ub(
        const AscendC::GlobalTensor<bfloat16_t> &xGm,
        uint64_t xBase,
        uint32_t xStride,
        uint32_t mCount,
        const AscendC::GlobalTensor<bfloat16_t> &wGm,
        uint64_t wBase,
        uint32_t K,
        uint32_t rows,
        AscendC::LocalTensor<float> &accUb,
        const uint32_t *tokIdx = nullptr)
    {
        AscendC::LocalTensor<float> xf = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> wf = bufWF32_.Get<float>()[kUbSkewWf];
        AscendC::LocalTensor<float> prod = bufProd_.Get<float>()[kUbSkewProd];
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        AscendC::LocalTensor<float> outM = bufAux1_.Get<float>();

        AscendC::Duplicate(accUb, 0.0f, mCount * kRowBlk);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t numGrps = (rows + kRowVec - 1) / kRowVec;

        for (uint32_t k0 = 0; k0 < K; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= K) ? kMonoTileK : (K - k0);
            uint32_t padLen = MonoPadReduceLen(len);
            uint32_t nChunks = padLen / 64U;
            uint8_t redRepStride = static_cast<uint8_t>(nChunks / 8U);

            for (uint32_t m = 0; m < mCount; ++m) {
                uint64_t rowBase = tokIdx ? (static_cast<uint64_t>(tokIdx[m]) * xStride)
                                          : (xBase + static_cast<uint64_t>(m) * xStride);
                AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, xGm, rowBase + k0, len);
                AscendC::Cast(xf[m * padLen], xb, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(xb);
                if (len < padLen) {
                    for (uint32_t i = len; i < padLen; ++i) {
                        xf.SetValue(m * padLen + i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }

            uint32_t firstRows = (kRowVec <= rows) ? kRowVec : rows;
            EnqueWeightGroup(wGm, wBase, K, k0, 0, firstRows, len, padLen);
            for (uint32_t g = 0; g < numGrps; ++g) {
                uint32_t gStart = g * kRowVec;
                uint32_t grpRows = (gStart + kRowVec <= rows) ? kRowVec : (rows - gStart);
                if (g + 1 < numGrps) {
                    uint32_t nextStart = (g + 1) * kRowVec;
                    uint32_t nextRows = (nextStart + kRowVec <= rows) ? kRowVec : (rows - nextStart);
                    EnqueWeightGroup(wGm, wBase, K, k0, nextStart, nextRows, len, padLen);
                }

                AscendC::LocalTensor<bfloat16_t> wb = inQueB_.DeQue<bfloat16_t>();
                AscendC::Cast(wf, wb, AscendC::RoundMode::CAST_NONE, grpRows * padLen);
                AscendC::PipeBarrier<PIPE_V>();
                inQueB_.FreeTensor(wb);
                if (len < padLen) {
                    for (uint32_t r = 0; r < grpRows; ++r) {
                        for (uint32_t i = len; i < padLen; ++i) {
                            wf.SetValue(r * padLen + i, 0.0f);
                        }
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                for (uint32_t m = 0; m < mCount; ++m) {
                    for (uint32_t r = 0; r < grpRows; ++r) {
                        AscendC::Mul(prod[r * padLen], xf[m * padLen], wf[r * padLen], padLen);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceSum(work, prod, 64, grpRows * nChunks, 1, 1, 8);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceSum(outM[m * 8], work, static_cast<int32_t>(nChunks), grpRows, 1, 1,
                                            redRepStride);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(accUb[m * kRowBlk + gStart], accUb[m * kRowBlk + gStart], outM[m * 8], 8);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    // Paired Gate+Up GEMV: loads & casts input xGm ONCE per k0 tile for both gate and up weights!
    __aicore__ inline void GemmAccumBf16PairedUb(
        const AscendC::GlobalTensor<bfloat16_t> &xGm,
        uint64_t xBase,
        uint32_t xStride,
        uint32_t mCount,
        const AscendC::GlobalTensor<bfloat16_t> &wGm,
        uint64_t wGateBase,
        uint64_t wUpBase,
        uint32_t K,
        uint32_t rows,
        AscendC::LocalTensor<float> &accGUb,
        AscendC::LocalTensor<float> &accUUb,
        const uint32_t *tokIdx = nullptr)
    {
        AscendC::LocalTensor<float> xf = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> wf = bufWF32_.Get<float>()[kUbSkewWf];
        AscendC::LocalTensor<float> prod = bufProd_.Get<float>()[kUbSkewProd];
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        AscendC::LocalTensor<float> outM = bufAux1_.Get<float>();

        AscendC::Duplicate(accGUb, 0.0f, mCount * kRowBlk);
        AscendC::Duplicate(accUUb, 0.0f, mCount * kRowBlk);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t numGrps = (rows + kRowVec - 1) / kRowVec;

        for (uint32_t k0 = 0; k0 < K; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= K) ? kMonoTileK : (K - k0);
            uint32_t padLen = MonoPadReduceLen(len);
            uint32_t nChunks = padLen / 64U;
            uint8_t redRepStride = static_cast<uint8_t>(nChunks / 8U);

            for (uint32_t m = 0; m < mCount; ++m) {
                uint64_t rowBase = tokIdx ? (static_cast<uint64_t>(tokIdx[m]) * xStride)
                                          : (xBase + static_cast<uint64_t>(m) * xStride);
                AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, xGm, rowBase + k0, len);
                AscendC::Cast(xf[m * padLen], xb, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(xb);
                if (len < padLen) {
                    for (uint32_t i = len; i < padLen; ++i) {
                        xf.SetValue(m * padLen + i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }

            uint32_t firstRows = (kRowVec <= rows) ? kRowVec : rows;
            EnqueWeightGroup(wGm, wGateBase, K, k0, 0, firstRows, len, padLen);
            for (uint32_t g = 0; g < numGrps; ++g) {
                uint32_t gStart = g * kRowVec;
                uint32_t grpRows = (gStart + kRowVec <= rows) ? kRowVec : (rows - gStart);
                // Prefetch Up group for same g while computing Gate group
                EnqueWeightGroup(wGm, wUpBase, K, k0, gStart, grpRows, len, padLen);

                {
                    AscendC::LocalTensor<bfloat16_t> wbG = inQueB_.DeQue<bfloat16_t>();
                    AscendC::Cast(wf, wbG, AscendC::RoundMode::CAST_NONE, grpRows * padLen);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueB_.FreeTensor(wbG);
                    if (len < padLen) {
                        for (uint32_t r = 0; r < grpRows; ++r) {
                            for (uint32_t i = len; i < padLen; ++i) {
                                wf.SetValue(r * padLen + i, 0.0f);
                            }
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    for (uint32_t m = 0; m < mCount; ++m) {
                        for (uint32_t r = 0; r < grpRows; ++r) {
                            AscendC::Mul(prod[r * padLen], xf[m * padLen], wf[r * padLen], padLen);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WholeReduceSum(work, prod, 64, grpRows * nChunks, 1, 1, 8);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WholeReduceSum(outM[m * 8], work, static_cast<int32_t>(nChunks), grpRows, 1, 1,
                                                redRepStride);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(accGUb[m * kRowBlk + gStart], accGUb[m * kRowBlk + gStart], outM[m * 8], 8);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }

                // Prefetch next Gate group (if any) while computing Up group
                if (g + 1 < numGrps) {
                    uint32_t nextStart = (g + 1) * kRowVec;
                    uint32_t nextRows = (nextStart + kRowVec <= rows) ? kRowVec : (rows - nextStart);
                    EnqueWeightGroup(wGm, wGateBase, K, k0, nextStart, nextRows, len, padLen);
                }

                {
                    AscendC::LocalTensor<bfloat16_t> wbU = inQueB_.DeQue<bfloat16_t>();
                    AscendC::Cast(wf, wbU, AscendC::RoundMode::CAST_NONE, grpRows * padLen);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueB_.FreeTensor(wbU);
                    if (len < padLen) {
                        for (uint32_t r = 0; r < grpRows; ++r) {
                            for (uint32_t i = len; i < padLen; ++i) {
                                wf.SetValue(r * padLen + i, 0.0f);
                            }
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    for (uint32_t m = 0; m < mCount; ++m) {
                        for (uint32_t r = 0; r < grpRows; ++r) {
                            AscendC::Mul(prod[r * padLen], xf[m * padLen], wf[r * padLen], padLen);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WholeReduceSum(work, prod, 64, grpRows * nChunks, 1, 1, 8);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WholeReduceSum(outM[m * 8], work, static_cast<int32_t>(nChunks), grpRows, 1, 1,
                                                redRepStride);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(accUUb[m * kRowBlk + gStart], accUUb[m * kRowBlk + gStart], outM[m * 8], 8);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }
        }
    }

    __aicore__ inline void StoreAccBatchBf16Ub(
        const AscendC::GlobalTensor<bfloat16_t> &outGm,
        uint32_t mBase,
        uint32_t mCount,
        uint32_t N,
        uint32_t r0,
        uint32_t rows,
        const AscendC::LocalTensor<float> &accUb,
        const uint32_t *tokIdx = nullptr)
    {
        for (uint32_t m = 0; m < mCount; ++m) {
            uint32_t tok = tokIdx ? tokIdx[m] : (mBase + m);
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, accUb[m * kRowBlk], AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(outGm, static_cast<uint64_t>(tok) * N + r0, ob, rows);
        }
    }

    __aicore__ inline void ComputeRmsNormRow(
        const AscendC::GlobalTensor<bfloat16_t> &inGm,
        uint64_t inBase,
        const AscendC::GlobalTensor<bfloat16_t> &wGm,
        const AscendC::GlobalTensor<bfloat16_t> &outGm,
        uint64_t outBase,
        uint32_t K,
        float eps)
    {
        AscendC::LocalTensor<float> xf = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> wf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        AscendC::LocalTensor<float> red = bufSmall_.Get<float>();
        float sumSq = 0.0f;

        for (uint32_t k0 = 0; k0 < K; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= K) ? kMonoTileK : (K - k0);
            uint32_t padLen = MonoPadReduceLen(len);
            uint32_t nChunks = padLen / 64U;

            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, inGm, inBase + k0, len);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);
            AscendC::Mul(xf, xf, xf, len);
            AscendC::PipeBarrier<PIPE_V>();
            if (len < padLen) {
                for (uint32_t i = len; i < padLen; ++i) {
                    xf.SetValue(i, 0.0f);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::WholeReduceSum(work, xf, 64, nChunks, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(red, work, static_cast<int32_t>(nChunks), 1, 1, 1, static_cast<uint8_t>(nChunks / 8U));
            SyncVecToScalar();
            sumSq += red.GetValue(0);
        }

        float invRms = ScalarRsqrt(sumSq / static_cast<float>(K) + eps);
        for (uint32_t k0 = 0; k0 < K; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= K) ? kMonoTileK : (K - k0);
            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, inGm, inBase + k0, len);
            AscendC::LocalTensor<bfloat16_t> wb = CopyInTile(inQueB_, wGm, k0, len);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(wf, wb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);
            inQueB_.FreeTensor(wb);
            AscendC::Muls(xf, xf, invRms, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(xf, xf, wf, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, xf, AscendC::RoundMode::CAST_RINT, len);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(outGm, outBase + k0, ob, len);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ApplyInterleavedRopeInPlace(
        const AscendC::GlobalTensor<bfloat16_t> &gm,
        uint64_t ropeBase,
        uint64_t cosSinBase,
        uint32_t ropeDim)
    {
        AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, gm, ropeBase, ropeDim);
        AscendC::LocalTensor<float> cs = CopyInTile(inQueB_, cosGm_, cosSinBase, ropeDim);
        AscendC::LocalTensor<float> sn = CopyInTile(inQueB_, sinGm_, cosSinBase, ropeDim);
        AscendC::LocalTensor<float> xf = bufXBatch_.Get<float>();
        AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, ropeDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        inQueA_.FreeTensor(xb);

        AscendC::LocalTensor<float> acc = bufAux0_.Get<float>();
        for (uint32_t i = 0; i < ropeDim; i += 2) {
            float x0 = xf.GetValue(i);
            float x1 = xf.GetValue(i + 1);
            float c0 = cs.GetValue(i);
            float c1 = cs.GetValue(i + 1);
            float s0 = sn.GetValue(i);
            float s1 = sn.GetValue(i + 1);
            acc.SetValue(i, x0 * c0 - x1 * s0);
            acc.SetValue(i + 1, x1 * c1 + x0 * s1);
        }
        inQueB_.FreeTensor(cs);
        inQueB_.FreeTensor(sn);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
        AscendC::Cast(ob, acc, AscendC::RoundMode::CAST_RINT, ropeDim);
        AscendC::PipeBarrier<PIPE_V>();
        outQue_.EnQue(ob);
        ob = outQue_.DeQue<bfloat16_t>();
        CopyOutTile(gm, ropeBase, ob, ropeDim);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeHcMixChunk(
        const AscendC::GlobalTensor<bfloat16_t> &resGm,
        const AscendC::GlobalTensor<float> &hcFnGm,
        uint32_t m,
        uint32_t cBase)
    {
        uint32_t hcD = kHcMult * meta_.hidden;
        AscendC::LocalTensor<float> xf = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> acc = bufProd_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        AscendC::LocalTensor<float> red = bufAux1_.Get<float>();

        AscendC::LocalTensor<float> invUb = CopyInTile(inQueB_, wsInvRms_, m, 1);
        float invRms = invUb.GetValue(0);
        inQueB_.FreeTensor(invUb);

        float out8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t k0 = 0; k0 < hcD; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= hcD) ? kMonoTileK : (hcD - k0);
            uint32_t padLen = MonoPadReduceLen(len);
            uint32_t nChunks = padLen / 64U;

            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, resGm, static_cast<uint64_t>(m) * hcD + k0, len);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);
            if (len < padLen) {
                for (uint32_t i = len; i < padLen; ++i) {
                    xf.SetValue(i, 0.0f);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            for (uint32_t c = 0; c < 8; ++c) {
                AscendC::LocalTensor<float> wf =
                    CopyInTile(inQueB_, hcFnGm, static_cast<uint64_t>(cBase + c) * hcD + k0, len);
                AscendC::Mul(acc, xf, wf, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueB_.FreeTensor(wf);
                if (len < padLen) {
                    for (uint32_t i = len; i < padLen; ++i) {
                        acc.SetValue(i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::WholeReduceSum(work, acc, 64, nChunks, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(red[c * 8], work, static_cast<int32_t>(nChunks), 1, 1, 1,
                                        static_cast<uint8_t>(nChunks / 8U));
                AscendC::PipeBarrier<PIPE_V>();
            }
            SyncVecToScalar();
            for (uint32_t c = 0; c < 8; ++c) {
                out8[c] += red.GetValue(c * 8);
            }
        }
        AscendC::LocalTensor<float> ob = outQue_.AllocTensor<float>();
        for (uint32_t c = 0; c < 8; ++c) {
            ob.SetValue(c, out8[c] * invRms);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(ob);
        ob = outQue_.DeQue<float>();
        CopyOutTile(wsMixes_, static_cast<uint64_t>(m) * 32 + cBase, ob, 8);
    }

    __aicore__ inline void ComputeSinkhornOneRow(
        uint32_t m,
        const AscendC::GlobalTensor<float> &scaleGm,
        const AscendC::GlobalTensor<float> &baseGm,
        const AscendC::GlobalTensor<float> &preOutGm,
        const AscendC::GlobalTensor<float> &postOutGm,
        const AscendC::GlobalTensor<float> &combOutGm)
    {
        float s0 = scaleGm.GetValue(0);
        float s1 = scaleGm.GetValue(1);
        float s2 = scaleGm.GetValue(2);
        AscendC::LocalTensor<float> baseUb = CopyInTile(inQueB_, baseGm, 0, kHcMixDim);
        float bVals[kHcMixDim];
        for (uint32_t i = 0; i < kHcMixDim; ++i) {
            bVals[i] = baseUb.GetValue(i);
        }
        inQueB_.FreeTensor(baseUb);

        AscendC::LocalTensor<float> mx = CopyInTile(inQueA_, wsMixes_, static_cast<uint64_t>(m) * 32, kHcMixDim);
        float pre4[4];
        float post4[4];
        AscendC::LocalTensor<float> sm = bufSmall_.Get<float>();
        for (uint32_t i = 0; i < 4; ++i) {
            sm.SetValue(i, -(mx.GetValue(i) * s0 + bVals[i]));
            sm.SetValue(4 + i, -(mx.GetValue(4 + i) * s1 + bVals[4 + i]));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(sm, sm, 8);
        SyncVecToScalar();
        for (uint32_t i = 0; i < 4; ++i) {
            pre4[i] = 1.0f / (1.0f + sm.GetValue(i)) + meta_.hcEps;
            post4[i] = 2.0f / (1.0f + sm.GetValue(4 + i));
        }

        float cm[4][4];
        for (uint32_t r = 0; r < 4; ++r) {
            float rowMax = -1e30f;
            for (uint32_t c = 0; c < 4; ++c) {
                float v = mx.GetValue(8 + r * 4 + c) * s2 + bVals[8 + r * 4 + c];
                cm[r][c] = v;
                if (v > rowMax) {
                    rowMax = v;
                }
            }
            for (uint32_t c = 0; c < 4; ++c) {
                sm.SetValue(r * 4 + c, cm[r][c] - rowMax);
            }
        }
        inQueA_.FreeTensor(mx);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(sm, sm, 16);
        SyncVecToScalar();

        for (uint32_t r = 0; r < 4; ++r) {
            float rowSum = 0.0f;
            for (uint32_t c = 0; c < 4; ++c) {
                cm[r][c] = sm.GetValue(r * 4 + c);
                rowSum += cm[r][c];
            }
            float invS = 1.0f / rowSum;
            for (uint32_t c = 0; c < 4; ++c) {
                cm[r][c] = cm[r][c] * invS + meta_.hcEps;
            }
        }

        for (uint32_t c = 0; c < 4; ++c) {
            float colSum = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + meta_.hcEps;
            float invC = 1.0f / colSum;
            for (uint32_t r = 0; r < 4; ++r) {
                cm[r][c] *= invC;
            }
        }
        for (uint32_t it = 1; it < meta_.sinkhornIters; ++it) {
            for (uint32_t r = 0; r < 4; ++r) {
                float rSum = cm[r][0] + cm[r][1] + cm[r][2] + cm[r][3] + meta_.hcEps;
                float invR = 1.0f / rSum;
                for (uint32_t c = 0; c < 4; ++c) {
                    cm[r][c] *= invR;
                }
            }
            for (uint32_t c = 0; c < 4; ++c) {
                float cSum = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + meta_.hcEps;
                float invC = 1.0f / cSum;
                for (uint32_t r = 0; r < 4; ++r) {
                    cm[r][c] *= invC;
                }
            }
        }

        AscendC::LocalTensor<float> obPre = outQue_.AllocTensor<float>();
        for (uint32_t i = 0; i < 4; ++i) {
            obPre.SetValue(i, pre4[i]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(obPre);
        obPre = outQue_.DeQue<float>();
        CopyOutTile(preOutGm, static_cast<uint64_t>(m) * 8, obPre, 4);

        AscendC::LocalTensor<float> obPost = outQue_.AllocTensor<float>();
        for (uint32_t i = 0; i < 4; ++i) {
            obPost.SetValue(i, post4[i]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(obPost);
        obPost = outQue_.DeQue<float>();
        CopyOutTile(postOutGm, static_cast<uint64_t>(m) * 8, obPost, 4);

        AscendC::LocalTensor<float> obComb = outQue_.AllocTensor<float>();
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                obComb.SetValue(r * 4 + c, cm[r][c]);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(obComb);
        obComb = outQue_.DeQue<float>();
        CopyOutTile(combOutGm, static_cast<uint64_t>(m) * kHcCombDim, obComb, kHcCombDim);
    }

    // Collapse resGm[m, 4, D] with preGm[m, 4], compute RMSNorm + 4D inv_rms.
    __aicore__ inline void CollapseNormAndInvRms(
        const AscendC::GlobalTensor<bfloat16_t> &resGm,
        const AscendC::GlobalTensor<float> &preGm,
        uint32_t preStride,
        const AscendC::GlobalTensor<bfloat16_t> &lnWGm)
    {
        uint32_t D = meta_.hidden;
        uint32_t hcD = kHcMult * D;
        AscendC::LocalTensor<float> xf = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> uAcc = bufSmall_.Get<float>();
        AscendC::LocalTensor<float> wf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        AscendC::LocalTensor<float> uTile = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> sq4d = bufAux1_.Get<float>();

        for (uint32_t m = blkIdx_; m < meta_.mTokens; m += blkNum_) {
            AscendC::LocalTensor<float> preUb = CopyInTile(inQueB_, preGm, static_cast<uint64_t>(m) * preStride, kHcMult);
            float k0w = preUb.GetValue(0);
            float k1w = preUb.GetValue(1);
            float k2w = preUb.GetValue(2);
            float k3w = preUb.GetValue(3);
            inQueB_.FreeTensor(preUb);

            float sumSqU = 0.0f;
            float sumSq4D = 0.0f;
            uint64_t rBase = static_cast<uint64_t>(m) * hcD;

            for (uint32_t d0 = 0; d0 < D; d0 += kMonoTileK) {
                uint32_t len = (d0 + kMonoTileK <= D) ? kMonoTileK : (D - d0);
                uint32_t padLen = MonoPadReduceLen(len);
                uint32_t nChunks = padLen / 64U;
                uint8_t redRepStride = static_cast<uint8_t>(nChunks / 8U);

                AscendC::LocalTensor<bfloat16_t> r0 = CopyInTile(inQueA_, resGm, rBase + d0, len);
                AscendC::Cast(xf, r0, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(r0);
                AscendC::Muls(uTile, xf, k0w, len);
                AscendC::Mul(sq4d, xf, xf, len);
                AscendC::PipeBarrier<PIPE_V>();

                float kw[3] = {k1w, k2w, k3w};
                for (uint32_t h = 1; h < 4; ++h) {
                    AscendC::LocalTensor<bfloat16_t> rh =
                        CopyInTile(inQueA_, resGm, rBase + static_cast<uint64_t>(h) * D + d0, len);
                    AscendC::Cast(xf, rh, AscendC::RoundMode::CAST_NONE, len);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueA_.FreeTensor(rh);
                    AscendC::Muls(wf, xf, kw[h - 1], len);
                    AscendC::Mul(xf, xf, xf, len);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(uTile, uTile, wf, len);
                    AscendC::Add(sq4d, sq4d, xf, len);
                    AscendC::PipeBarrier<PIPE_V>();
                }

                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, uTile, AscendC::RoundMode::CAST_RINT, len);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                AscendC::Cast(uTile, ob, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                CopyOutTile(wsXNorm_, static_cast<uint64_t>(m) * D + d0, ob, len);

                AscendC::Mul(uTile, uTile, uTile, len);
                AscendC::PipeBarrier<PIPE_V>();
                if (len < padLen) {
                    for (uint32_t i = len; i < padLen; ++i) {
                        uTile.SetValue(i, 0.0f);
                        sq4d.SetValue(i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::WholeReduceSum(work, uTile, 64, nChunks, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(uAcc[0], work, static_cast<int32_t>(nChunks), 1, 1, 1, redRepStride);
                AscendC::WholeReduceSum(work, sq4d, 64, nChunks, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(uAcc[8], work, static_cast<int32_t>(nChunks), 1, 1, 1, redRepStride);
                SyncVecToScalar();
                sumSqU += uAcc.GetValue(0);
                sumSq4D += uAcc.GetValue(8);
            }

            float invU = ScalarRsqrt(sumSqU / static_cast<float>(D) + meta_.normEps);
            float inv4D = ScalarRsqrt(sumSq4D / static_cast<float>(hcD) + meta_.normEps);

            AscendC::LocalTensor<float> obInv = outQue_.AllocTensor<float>();
            obInv.SetValue(0, inv4D);
            AscendC::PipeBarrier<PIPE_ALL>();
            outQue_.EnQue(obInv);
            obInv = outQue_.DeQue<float>();
            CopyOutTile(wsInvRms_, m, obInv, 1);

            for (uint32_t d0 = 0; d0 < D; d0 += kMonoTileK) {
                uint32_t len = (d0 + kMonoTileK <= D) ? kMonoTileK : (D - d0);
                AscendC::LocalTensor<bfloat16_t> ub = CopyInTile(inQueA_, wsXNorm_, static_cast<uint64_t>(m) * D + d0, len);
                AscendC::LocalTensor<bfloat16_t> wb = CopyInTile(inQueB_, lnWGm, d0, len);
                AscendC::Cast(xf, ub, AscendC::RoundMode::CAST_NONE, len);
                AscendC::Cast(wf, wb, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(ub);
                inQueB_.FreeTensor(wb);
                AscendC::Muls(xf, xf, invU, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(xf, xf, wf, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, xf, AscendC::RoundMode::CAST_RINT, len);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(wsXNorm_, static_cast<uint64_t>(m) * D + d0, ob, len);
            }
        }
    }

    __aicore__ inline void Stage1And2QkvRopeCache()
    {
        CollapseNormAndInvRms(resInGm_, preMixInGm_, kHcMult, inputLnWGm_);
        BarrierAllCores();

        uint32_t M = meta_.mTokens;
        uint32_t D = meta_.hidden;
        uint32_t mTiles = (M + kMonoTileM - 1) / kMonoTileM;
        uint32_t qBlks = (meta_.qLora + kRowBlk - 1) / kRowBlk;
        uint32_t kvBlks = (meta_.headDim + kRowBlk - 1) / kRowBlk;
        uint32_t qkvTasks = mTiles * (qBlks + kvBlks);
        uint32_t mixTasks = M * 3U;
        uint32_t totalPoolA = qkvTasks + mixTasks;

        for (uint32_t tid = blkIdx_; tid < totalPoolA; tid += blkNum_) {
            if (tid < qkvTasks) {
                uint32_t mTile = tid / (qBlks + kvBlks);
                uint32_t b = tid % (qBlks + kvBlks);
                uint32_t mBase = mTile * kMonoTileM;
                uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
                AscendC::LocalTensor<float> accUb = bufAcc_.Get<float>();
                if (b < qBlks) {
                    uint32_t r0 = b * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= meta_.qLora) ? kRowBlk : (meta_.qLora - r0);
                    GemmAccumBf16Ub(wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mCount,
                                    wqAGm_, static_cast<uint64_t>(r0) * D, D, rows, accUb);
                    StoreAccBatchBf16Ub(wsQA_, mBase, mCount, meta_.qLora, r0, rows, accUb);
                } else {
                    uint32_t r0 = (b - qBlks) * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= meta_.headDim) ? kRowBlk : (meta_.headDim - r0);
                    GemmAccumBf16Ub(wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mCount,
                                    wkvGm_, static_cast<uint64_t>(r0) * D, D, rows, accUb);
                    StoreAccBatchBf16Ub(wsKV_, mBase, mCount, meta_.headDim, r0, rows, accUb);
                }
            } else {
                uint32_t mt = tid - qkvTasks;
                uint32_t m = mt / 3U;
                uint32_t cBase = (mt % 3U) * 8U;
                ComputeHcMixChunk(resInGm_, hcAttnFnGm_, m, cBase);
            }
        }
        BarrierAllCores();

        // Pool B: q_norm + kv_norm + kv RoPE + SWA cache scatter + attn Sinkhorn (no dependency conflict!)
        for (uint32_t m = blkIdx_; m < M; m += blkNum_) {
            ComputeRmsNormRow(wsQA_, static_cast<uint64_t>(m) * meta_.qLora, qNormWGm_, wsQR_,
                              static_cast<uint64_t>(m) * meta_.qLora, meta_.qLora, meta_.normEps);
            ComputeRmsNormRow(wsKV_, static_cast<uint64_t>(m) * meta_.headDim, kvNormWGm_, wsKV_,
                              static_cast<uint64_t>(m) * meta_.headDim, meta_.headDim, meta_.normEps);
            ApplyInterleavedRopeInPlace(wsKV_, static_cast<uint64_t>(m) * meta_.headDim + meta_.nopeDim,
                                        static_cast<uint64_t>(m) * meta_.ropeDim, meta_.ropeDim);
            int64_t slot = slotGm_.GetValue(m);
            if (slot >= 0) {
                AscendC::LocalTensor<bfloat16_t> kvUb =
                    CopyInTile(inQueA_, wsKV_, static_cast<uint64_t>(m) * meta_.headDim, meta_.headDim);
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::DataCopy(ob, kvUb, meta_.headDim);
                AscendC::PipeBarrier<PIPE_ALL>();
                inQueA_.FreeTensor(kvUb);
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(swaKvCacheGm_, static_cast<uint64_t>(slot) * meta_.headDim, ob, meta_.headDim);
            }
            ComputeSinkhornOneRow(m, hcAttnScaleGm_, hcAttnBaseGm_, wsAttnPre_, wsAttnPost_, wsAttnComb_);
        }
        BarrierAllCores();

        // q_b GEMV with fused interleaved partial Q RoPE epilogue when aligned
        uint32_t qN = meta_.nHeads * meta_.headDim;
        uint32_t qbBlks = (qN + kRowBlk - 1) / kRowBlk;
        uint32_t totalQB = mTiles * qbBlks;
        bool fuseQRope = (meta_.headDim % kRowBlk == 0) && (meta_.nopeDim % kRowBlk == 0);

        for (uint32_t tid = blkIdx_; tid < totalQB; tid += blkNum_) {
            uint32_t mTile = tid / qbBlks;
            uint32_t mBase = mTile * kMonoTileM;
            uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
            uint32_t r0 = (tid % qbBlks) * kRowBlk;
            uint32_t rows = (r0 + kRowBlk <= qN) ? kRowBlk : (qN - r0);

            AscendC::LocalTensor<float> accUb = bufAcc_.Get<float>();
            GemmAccumBf16Ub(wsQR_, static_cast<uint64_t>(mBase) * meta_.qLora, meta_.qLora, mCount,
                            wqBGm_, static_cast<uint64_t>(r0) * meta_.qLora, meta_.qLora, rows, accUb);

            if (fuseQRope) {
                uint32_t hdOff = r0 % meta_.headDim;
                if (hdOff >= meta_.nopeDim) {
                    SyncVecToScalar();
                    uint32_t rOff = hdOff - meta_.nopeDim;
                    for (uint32_t m = 0; m < mCount; ++m) {
                        uint64_t csBase = static_cast<uint64_t>(mBase + m) * meta_.ropeDim + rOff;
                        AscendC::LocalTensor<float> cs = CopyInTile(inQueA_, cosGm_, csBase, rows);
                        AscendC::LocalTensor<float> sn = CopyInTile(inQueB_, sinGm_, csBase, rows);
                        for (uint32_t i = 0; i < rows; i += 2) {
                            float x0 = accUb.GetValue(m * kRowBlk + i);
                            float x1 = accUb.GetValue(m * kRowBlk + i + 1);
                            float c0 = cs.GetValue(i);
                            float c1 = cs.GetValue(i + 1);
                            float s0 = sn.GetValue(i);
                            float s1 = sn.GetValue(i + 1);
                            accUb.SetValue(m * kRowBlk + i, x0 * c0 - x1 * s0);
                            accUb.SetValue(m * kRowBlk + i + 1, x1 * c1 + x0 * s1);
                        }
                        inQueA_.FreeTensor(cs);
                        inQueB_.FreeTensor(sn);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            StoreAccBatchBf16Ub(wsQ_, mBase, mCount, qN, r0, rows, accUb);
        }

        if (!fuseQRope) {
            BarrierAllCores();
            uint32_t totalHeads = M * meta_.nHeads;
            for (uint32_t tid = blkIdx_; tid < totalHeads; tid += blkNum_) {
                uint32_t m = tid / meta_.nHeads;
                uint32_t h = tid % meta_.nHeads;
                uint64_t ropeOff = (static_cast<uint64_t>(m) * meta_.nHeads + h) * meta_.headDim + meta_.nopeDim;
                ApplyInterleavedRopeInPlace(wsQ_, ropeOff, static_cast<uint64_t>(m) * meta_.ropeDim, meta_.ropeDim);
            }
        }
    }

    __aicore__ inline bool TryEnqueKvRow(uint32_t m, int32_t t, uint32_t hd)
    {
        uint32_t blkIdxLogical = static_cast<uint32_t>(t) / meta_.blockSize;
        uint32_t blkOff = static_cast<uint32_t>(t) % meta_.blockSize;
        int32_t physBlk = blockTableGm_.GetValue(static_cast<uint64_t>(m) * meta_.maxBlocks + blkIdxLogical);
        if (physBlk < 0) {
            return false;
        }
        uint64_t kvOff = (static_cast<uint64_t>(physBlk) * meta_.blockSize + blkOff) * hd;
        AscendC::LocalTensor<bfloat16_t> kb = inQueA_.AllocTensor<bfloat16_t>();
        if ((kvOff * sizeof(bfloat16_t)) % 32 == 0 && (hd * sizeof(bfloat16_t)) % 32 == 0) {
            AscendC::DataCopy(kb, swaKvCacheGm_[kvOff], hd);
        } else {
            AscendC::DataCopyPad(kb, swaKvCacheGm_[kvOff], {1, static_cast<uint16_t>(hd * sizeof(bfloat16_t)), 0, 0},
                                 {false, 0, 0, 0});
        }
        inQueA_.EnQue(kb);
        return true;
    }

    __aicore__ inline void Stage3FlashSwaAndOProjPost()
    {
        uint32_t M = meta_.mTokens;
        uint32_t H = meta_.nHeads;
        uint32_t hd = meta_.headDim;
        uint32_t padLen = MonoPadReduceLen(hd);
        uint32_t nChunks = padLen / 64U;
        uint8_t redRepStride = static_cast<uint8_t>(nChunks / 8U);
        uint32_t totalHeads = M * H;

        AscendC::LocalTensor<float> qf = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> kf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> acc = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> dotRes = bufProd_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        AscendC::LocalTensor<float> dotOut = bufAux1_.Get<float>();
        AscendC::LocalTensor<float> sm = bufSmall_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalHeads; tid += blkNum_) {
            uint32_t m = tid / H;
            uint32_t h = tid % H;
            uint64_t qOff = static_cast<uint64_t>(tid) * hd;

            AscendC::LocalTensor<bfloat16_t> qb = CopyInTile(inQueA_, wsQ_, qOff, hd);
            AscendC::Cast(qf, qb, AscendC::RoundMode::CAST_NONE, hd);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(qb);
            if (hd < padLen) {
                for (uint32_t i = hd; i < padLen; ++i) {
                    qf.SetValue(i, 0.0f);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            AscendC::Duplicate(acc, 0.0f, hd);
            AscendC::PipeBarrier<PIPE_V>();

            float mMax = meta_.hasSink ? attnSinkGm_.GetValue(h) : -1e30f;
            float lSum = meta_.hasSink ? 1.0f : 0.0f;

            int32_t seqLen = seqLensGm_.GetValue(m);
            int32_t startT = seqLen - static_cast<int32_t>(meta_.windowSize);
            if (startT < 0) {
                startT = 0;
            }

            int32_t curT = startT;
            while (curT < seqLen) {
                if (!TryEnqueKvRow(m, curT, hd)) {
                    ++curT;
                    continue;
                }
                while (curT < seqLen) {
                    int32_t nextT = curT + 1;
                    bool hasNext = false;
                    while (nextT < seqLen) {
                        if (TryEnqueKvRow(m, nextT, hd)) {
                            hasNext = true;
                            break;
                        }
                        ++nextT;
                    }

                    AscendC::LocalTensor<bfloat16_t> kb = inQueA_.DeQue<bfloat16_t>();
                    AscendC::Cast(kf, kb, AscendC::RoundMode::CAST_NONE, hd);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueA_.FreeTensor(kb);
                    if (hd < padLen) {
                        for (uint32_t i = hd; i < padLen; ++i) {
                            kf.SetValue(i, 0.0f);
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }

                    AscendC::Mul(dotRes, qf, kf, padLen);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceSum(work, dotRes, 64, nChunks, 1, 1, 8);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceSum(dotOut, work, static_cast<int32_t>(nChunks), 1, 1, 1, redRepStride);
                    SyncVecToScalar();
                    float score = dotOut.GetValue(0) * meta_.softmaxScale;

                    float newMax = (score > mMax) ? score : mMax;
                    bool hadSum = (lSum > 0.0f);
                    sm.SetValue(0, hadSum ? (mMax - newMax) : -80.0f);
                    sm.SetValue(1, score - newMax);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Exp(sm, sm, 8);
                    SyncVecToScalar();
                    float alpha = hadSum ? sm.GetValue(0) : 0.0f;
                    float pVal = sm.GetValue(1);
                    lSum = lSum * alpha + pVal;
                    mMax = newMax;

                    AscendC::Muls(acc, acc, alpha, hd);
                    AscendC::Muls(kf, kf, pVal, hd);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(acc, acc, kf, hd);
                    AscendC::PipeBarrier<PIPE_V>();

                    if (!hasNext) {
                        curT = seqLen;
                        break;
                    }
                    curT = nextT;
                }
            }

            float invL = (lSum > 0.0f) ? (1.0f / lSum) : 0.0f;
            AscendC::Muls(acc, acc, invL, hd);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, acc, AscendC::RoundMode::CAST_RINT, hd);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(wsAttnOut_, qOff, ob, hd);
        }
        BarrierAllCores();

        // Grouped wo_a with mTile <= 8 weight reuse per group
        uint32_t groupHidden = (meta_.nHeads * meta_.headDim) / meta_.oGroups;
        uint32_t gRowBlks = (meta_.oLora + kRowBlk - 1) / kRowBlk;
        uint32_t mTiles = (M + kMonoTileM - 1) / kMonoTileM;
        uint32_t totalOA = mTiles * meta_.oGroups * gRowBlks;
        uint32_t xOAStride = meta_.nHeads * meta_.headDim;
        uint32_t outOAStride = meta_.oGroups * meta_.oLora;

        for (uint32_t tid = blkIdx_; tid < totalOA; tid += blkNum_) {
            uint32_t mtg = tid / gRowBlks;
            uint32_t mTile = mtg / meta_.oGroups;
            uint32_t g = mtg % meta_.oGroups;
            uint32_t mBase = mTile * kMonoTileM;
            uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
            uint32_t r0 = (tid % gRowBlks) * kRowBlk;
            uint32_t rows = (r0 + kRowBlk <= meta_.oLora) ? kRowBlk : (meta_.oLora - r0);

            AscendC::LocalTensor<float> oAccUb = bufAcc_.Get<float>();
            uint64_t xOff = static_cast<uint64_t>(mBase) * xOAStride + static_cast<uint64_t>(g) * groupHidden;
            uint64_t wOff = (static_cast<uint64_t>(g) * meta_.oLora + r0) * groupHidden;
            GemmAccumBf16Ub(wsAttnOut_, xOff, xOAStride, mCount, woAGm_, wOff, groupHidden, rows, oAccUb);

            for (uint32_t m = 0; m < mCount; ++m) {
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, oAccUb[m * kRowBlk], AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(wsOA_, static_cast<uint64_t>(mBase + m) * outOAStride + static_cast<uint64_t>(g) * meta_.oLora + r0,
                            ob, rows);
            }
        }
        BarrierAllCores();

        // wo_b + Vectorized Fused Mid hc_post directly into wsResMid_ (0 SyncVecToScalar!)
        uint32_t D = meta_.hidden;
        uint32_t oaK = meta_.oGroups * meta_.oLora;
        uint32_t dBlks = (D + kRowBlk - 1) / kRowBlk;
        uint32_t totalOB = mTiles * dBlks;

        for (uint32_t tid = blkIdx_; tid < totalOB; tid += blkNum_) {
            uint32_t mTile = tid / dBlks;
            uint32_t mBase = mTile * kMonoTileM;
            uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
            uint32_t d0 = (tid % dBlks) * kRowBlk;
            uint32_t rows = (d0 + kRowBlk <= D) ? kRowBlk : (D - d0);
            uint32_t vRows = MonoVecAlign8(rows);

            AscendC::LocalTensor<float> xAccUb = bufAcc_.Get<float>();
            GemmAccumBf16Ub(wsOA_, static_cast<uint64_t>(mBase) * oaK, oaK, mCount,
                            woBGm_, static_cast<uint64_t>(d0) * oaK, oaK, rows, xAccUb);

            AscendC::LocalTensor<float> rAll = bufXBatch_.Get<float>();
            AscendC::LocalTensor<float> rf = bufAux1_.Get<float>();
            AscendC::LocalTensor<float> tmp = bufWork_.Get<float>();
            for (uint32_t m = 0; m < mCount; ++m) {
                uint32_t tok = mBase + m;
                AscendC::LocalTensor<float> poUb = CopyInTile(inQueB_, wsAttnPost_, static_cast<uint64_t>(tok) * 8, kHcMult);
                float post4[4];
                for (uint32_t i = 0; i < 4; ++i) {
                    post4[i] = poUb.GetValue(i);
                }
                inQueB_.FreeTensor(poUb);

                AscendC::LocalTensor<float> cmUb =
                    CopyInTile(inQueB_, wsAttnComb_, static_cast<uint64_t>(tok) * kHcCombDim, kHcCombDim);
                float cm44[4][4];
                for (uint32_t r = 0; r < 4; ++r) {
                    for (uint32_t c = 0; c < 4; ++c) {
                        cm44[r][c] = cmUb.GetValue(r * 4 + c);
                    }
                }
                inQueB_.FreeTensor(cmUb);

                uint64_t rBase = static_cast<uint64_t>(tok) * kHcMult * D + d0;
                for (uint32_t i = 0; i < 4; ++i) {
                    AscendC::LocalTensor<bfloat16_t> rb =
                        CopyInTile(inQueA_, resInGm_, rBase + static_cast<uint64_t>(i) * D, rows);
                    AscendC::Cast(rAll[i * 16], rb, AscendC::RoundMode::CAST_NONE, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueA_.FreeTensor(rb);
                }

                for (uint32_t j = 0; j < 4; ++j) {
                    AscendC::Muls(rf, xAccUb[m * kRowBlk], post4[j], vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t i = 0; i < 4; ++i) {
                        AscendC::Muls(tmp, rAll[i * 16], cm44[i][j], vRows);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(rf, rf, tmp, vRows);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                    AscendC::Cast(ob, rf, AscendC::RoundMode::CAST_RINT, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    outQue_.EnQue(ob);
                    ob = outQue_.DeQue<bfloat16_t>();
                    CopyOutTile(wsResMid_, rBase + static_cast<uint64_t>(j) * D, ob, rows);
                }
            }
        }
    }

    __aicore__ inline void ComputeClippedSwigluBatchUb(
        const AscendC::GlobalTensor<bfloat16_t> &outGm,
        uint64_t outBase,
        uint32_t outStride,
        uint32_t mCount,
        uint32_t r0,
        uint32_t rows,
        const AscendC::LocalTensor<float> &accGUb,
        const AscendC::LocalTensor<float> &accUUb,
        const uint32_t *tokOutIdx = nullptr)
    {
        float lim = meta_.swigluLimit;
        AscendC::LocalTensor<float> gf = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> sig = bufAux1_.Get<float>();
        AscendC::LocalTensor<float> uf = bufProd_.Get<float>()[kUbSkewProd];
        uint32_t totalE = mCount * kRowBlk;

        if (lim > 0.0f) {
            AscendC::Mins(gf, accGUb, lim, totalE);
            AscendC::Mins(uf, accUUb, lim, totalE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(uf, uf, -lim, totalE);
        } else {
            AscendC::Muls(gf, accGUb, 1.0f, totalE);
            AscendC::Muls(uf, accUUb, 1.0f, totalE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(sig, gf, -1.0f, totalE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(sig, sig, totalE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sig, sig, 1.0f, totalE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(gf, gf, sig, totalE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gf, gf, uf, totalE);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t m = 0; m < mCount; ++m) {
            uint64_t dstBase = tokOutIdx ? (static_cast<uint64_t>(tokOutIdx[m]) * outStride)
                                         : (outBase + static_cast<uint64_t>(m) * outStride);
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, gf[m * kRowBlk], AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(outGm, dstBase + r0, ob, rows);
        }
    }

    __aicore__ inline void SelectTopKPerToken(uint32_t m)
    {
        uint32_t E = meta_.nExperts;
        uint32_t K = meta_.topK;
        uint64_t lStride = MonoAlign64(E);
        AscendC::LocalTensor<float> lg = CopyInTile(inQueA_, wsRouterLogits_, static_cast<uint64_t>(m) * lStride, E);
        AscendC::LocalTensor<float> scores = bufXBatch_.Get<float>();

        if (meta_.useSigmoid) {
            for (uint32_t e = 0; e < E; ++e) {
                scores.SetValue(e, -lg.GetValue(e));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t eAl = MonoVecAlign8(E);
            AscendC::Exp(scores, scores, eAl);
            SyncVecToScalar();
            for (uint32_t e = 0; e < E; ++e) {
                scores.SetValue(e, 1.0f / (1.0f + scores.GetValue(e)));
            }
        } else {
            float lMax = -1e30f;
            for (uint32_t e = 0; e < E; ++e) {
                float v = lg.GetValue(e);
                if (v > lMax) {
                    lMax = v;
                }
            }
            for (uint32_t e = 0; e < E; ++e) {
                scores.SetValue(e, lg.GetValue(e) - lMax);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t eAl = MonoVecAlign8(E);
            AscendC::Exp(scores, scores, eAl);
            SyncVecToScalar();
            float lSum = 0.0f;
            for (uint32_t e = 0; e < E; ++e) {
                lSum += scores.GetValue(e);
            }
            float invS = 1.0f / lSum;
            for (uint32_t e = 0; e < E; ++e) {
                scores.SetValue(e, scores.GetValue(e) * invS);
            }
        }
        inQueA_.FreeTensor(lg);

        AscendC::LocalTensor<float> sel = bufWF32_.Get<float>();
        if (meta_.hasEBias) {
            AscendC::LocalTensor<float> eb = CopyInTile(inQueB_, eBiasGm_, 0, E);
            for (uint32_t e = 0; e < E; ++e) {
                sel.SetValue(e, scores.GetValue(e) + eb.GetValue(e));
            }
            inQueB_.FreeTensor(eb);
        } else {
            for (uint32_t e = 0; e < E; ++e) {
                sel.SetValue(e, scores.GetValue(e));
            }
        }

        AscendC::LocalTensor<int32_t> obIds = outQue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<float> obW = bufAux0_.Get<float>();
        float wSum = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            int32_t bestIdx = 0;
            float bestVal = -1e30f;
            for (uint32_t e = 0; e < E; ++e) {
                float sv = sel.GetValue(e);
                if (sv > bestVal) {
                    bestVal = sv;
                    bestIdx = static_cast<int32_t>(e);
                }
            }
            sel.SetValue(bestIdx, -1e30f);
            obIds.SetValue(k, bestIdx);
            float wVal = scores.GetValue(bestIdx);
            obW.SetValue(k, wVal);
            wSum += wVal;
        }
        float scale = meta_.renorm ? (meta_.routedScale / (wSum + 1e-20f)) : meta_.routedScale;
        for (uint32_t k = 0; k < K; ++k) {
            obW.SetValue(k, obW.GetValue(k) * scale);
        }

        uint64_t tkStride = MonoAlign64(K);
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(obIds);
        obIds = outQue_.DeQue<int32_t>();
        CopyOutTile(wsTopkIds_, static_cast<uint64_t>(m) * tkStride, obIds, K);

        AscendC::LocalTensor<float> obWOut = outQue_.AllocTensor<float>();
        for (uint32_t k = 0; k < K; ++k) {
            obWOut.SetValue(k, obW.GetValue(k));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(obWOut);
        obWOut = outQue_.DeQue<float>();
        CopyOutTile(wsTopkW_, static_cast<uint64_t>(m) * tkStride, obWOut, K);
    }

    __aicore__ inline void BuildExpertGroups(uint32_t M, uint32_t K)
    {
        uint32_t MK = M * K;
        uint64_t tkStride = MonoAlign64(K);
        // Use bufXBatch_ (32 KB = 8192 int32s) to stage ids, mIdx, kIdx, used flags in UB
        AscendC::LocalTensor<int32_t> tmpUb = bufXBatch_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> slotIds = tmpUb[0];
        AscendC::LocalTensor<int32_t> slotM = tmpUb[1024];
        AscendC::LocalTensor<int32_t> slotK = tmpUb[2048];
        AscendC::LocalTensor<int32_t> used = tmpUb[3072];

        for (uint32_t m = 0; m < M; ++m) {
            AscendC::LocalTensor<int32_t> idUb =
                CopyInTile(inQueA_, wsTopkIds_, static_cast<uint64_t>(m) * tkStride, K);
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t idx = m * K + k;
                slotIds.SetValue(idx, idUb.GetValue(k));
                slotM.SetValue(idx, static_cast<int32_t>(m));
                slotK.SetValue(idx, static_cast<int32_t>(k));
                used.SetValue(idx, 0);
            }
            inQueA_.FreeTensor(idUb);
        }

        uint32_t numGroups = 0;
        for (uint32_t i = 0; i < MK; ++i) {
            if (used.GetValue(i) != 0) {
                continue;
            }
            int32_t eid = slotIds.GetValue(i);
            uint32_t cnt = 0;
            AscendC::LocalTensor<int32_t> mob = outQue_.AllocTensor<int32_t>();
            mob.SetValue(0, eid);
            for (uint32_t j = i; j < MK && cnt < kMonoTileM; ++j) {
                if (used.GetValue(j) == 0 && slotIds.GetValue(j) == eid) {
                    used.SetValue(j, 1);
                    mob.SetValue(2 + cnt, slotM.GetValue(j));
                    mob.SetValue(2 + kMonoTileM + cnt, slotK.GetValue(j));
                    ++cnt;
                }
            }
            mob.SetValue(1, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();
            outQue_.EnQue(mob);
            mob = outQue_.DeQue<int32_t>();
            CopyOutTile(wsExpMeta_, 64ULL + static_cast<uint64_t>(numGroups) * kExpGrpStride, mob, kExpGrpStride);
            ++numGroups;
        }
        AscendC::LocalTensor<int32_t> cntOb = outQue_.AllocTensor<int32_t>();
        cntOb.SetValue(0, static_cast<int32_t>(numGroups));
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(cntOb);
        cntOb = outQue_.DeQue<int32_t>();
        CopyOutTile(wsExpMeta_, 0, cntOb, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void VectorizedFinalHcPost(
        uint32_t M,
        uint32_t D,
        uint32_t K,
        uint32_t hasShared,
        uint32_t dRowBlks,
        bool groupedRouted)
    {
        uint32_t totalFinal = M * dRowBlks;
        uint64_t tkStride = MonoAlign64(K);
        AscendC::LocalTensor<float> rAll = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> combUb = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> rf = bufAux1_.Get<float>();
        AscendC::LocalTensor<float> tmp = bufWork_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalFinal; tid += blkNum_) {
            uint32_t m = tid / dRowBlks;
            uint32_t d0 = (tid % dRowBlks) * kRowBlk;
            uint32_t rows = (d0 + kRowBlk <= D) ? kRowBlk : (D - d0);
            uint32_t vRows = MonoVecAlign8(rows);

            if (hasShared) {
                AscendC::LocalTensor<bfloat16_t> sb =
                    CopyInTile(inQueA_, wsSharedOut_, static_cast<uint64_t>(m) * D + d0, rows);
                AscendC::Cast(combUb, sb, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(sb);
            } else {
                AscendC::Duplicate(combUb, 0.0f, vRows);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::LocalTensor<int32_t> idUb =
                CopyInTile(inQueA_, wsTopkIds_, static_cast<uint64_t>(m) * tkStride, K);
            AscendC::LocalTensor<float> wUb = CopyInTile(inQueB_, wsTopkW_, static_cast<uint64_t>(m) * tkStride, K);
            uint32_t ids[16];
            float ws[16];
            for (uint32_t k = 0; k < K; ++k) {
                ids[k] = static_cast<uint32_t>(idUb.GetValue(k));
                ws[k] = wUb.GetValue(k);
            }
            inQueA_.FreeTensor(idUb);
            inQueB_.FreeTensor(wUb);

            if (groupedRouted) {
                for (uint32_t k = 0; k < K; ++k) {
                    uint64_t roOff = (static_cast<uint64_t>(m) * K + k) * D + d0;
                    AscendC::LocalTensor<bfloat16_t> rob = CopyInTile(inQueA_, wsRoutedOut_, roOff, rows);
                    AscendC::Cast(tmp, rob, AscendC::RoundMode::CAST_NONE, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueA_.FreeTensor(rob);
                    AscendC::Muls(tmp, tmp, ws[k], vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(combUb, combUb, tmp, vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                AscendC::LocalTensor<float> rAccUb = bufAcc_.Get<float>()[kAccSlotFloats];
                for (uint32_t k = 0; k < K; ++k) {
                    uint64_t wOff = (static_cast<uint64_t>(ids[k]) * D + d0) * meta_.routedInter;
                    uint64_t actOff = (static_cast<uint64_t>(m) * K + k) * meta_.routedInter;
                    GemmAccumBf16Ub(wsRoutedAct_, actOff, meta_.routedInter, 1, routedW2Gm_, wOff, meta_.routedInter,
                                    rows, rAccUb);
                    AscendC::Muls(tmp, rAccUb, ws[k], vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(combUb, combUb, tmp, vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            AscendC::LocalTensor<float> poUb = CopyInTile(inQueB_, wsFfnPost_, static_cast<uint64_t>(m) * 8, kHcMult);
            float post4[4];
            for (uint32_t i = 0; i < 4; ++i) {
                post4[i] = poUb.GetValue(i);
            }
            inQueB_.FreeTensor(poUb);

            AscendC::LocalTensor<float> cmUb =
                CopyInTile(inQueB_, wsFfnComb_, static_cast<uint64_t>(m) * kHcCombDim, kHcCombDim);
            float cm44[4][4];
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t c = 0; c < 4; ++c) {
                    cm44[r][c] = cmUb.GetValue(r * 4 + c);
                }
            }
            inQueB_.FreeTensor(cmUb);

            uint64_t rBase = static_cast<uint64_t>(m) * kHcMult * D + d0;
            for (uint32_t i = 0; i < 4; ++i) {
                AscendC::LocalTensor<bfloat16_t> rb =
                    CopyInTile(inQueA_, wsResMid_, rBase + static_cast<uint64_t>(i) * D, rows);
                AscendC::Cast(rAll[i * 16], rb, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(rb);
            }

            for (uint32_t j = 0; j < 4; ++j) {
                AscendC::Muls(rf, combUb, post4[j], vRows);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < 4; ++i) {
                    AscendC::Muls(tmp, rAll[i * 16], cm44[i][j], vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(rf, rf, tmp, vRows);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, rf, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(resOutGm_, rBase + static_cast<uint64_t>(j) * D, ob, rows);
            }
        }
    }

    __aicore__ inline void Stage4And5MoEFinalPost()
    {
        // Stage 4a: CollapseNorm + 4D inv_rms on wsResMid_
        CollapseNormAndInvRms(wsResMid_, wsAttnPre_, 8, postAttnLnWGm_);
        BarrierAllCores();

        uint32_t M = meta_.mTokens;
        uint32_t D = meta_.hidden;
        uint32_t E = meta_.nExperts;
        uint32_t K = meta_.topK;
        uint32_t mTiles = (M + kMonoTileM - 1) / kMonoTileM;
        uint32_t hasShared = (meta_.sharedInter > 0) ? 1U : 0U;

        // Horizontal Pool: (hc_ffn_mix || gate || shared_w13_paired_swiglu)
        uint32_t gateBlks = (E + kRowBlk - 1) / kRowBlk;
        uint32_t sharedPairBlks = hasShared ? ((meta_.sharedInter + kRowBlk - 1) / kRowBlk) : 0U;
        uint32_t gemmTasksA = mTiles * (gateBlks + sharedPairBlks);
        uint32_t mixTasks = M * 3U;
        uint32_t totalPool5a = gemmTasksA + mixTasks;
        uint64_t lStride = MonoAlign64(E);

        for (uint32_t tid = blkIdx_; tid < totalPool5a; tid += blkNum_) {
            if (tid < gemmTasksA) {
                uint32_t mTile = tid / (gateBlks + sharedPairBlks);
                uint32_t b = tid % (gateBlks + sharedPairBlks);
                uint32_t mBase = mTile * kMonoTileM;
                uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
                AscendC::LocalTensor<float> acc0 = bufAcc_.Get<float>()[0];
                AscendC::LocalTensor<float> acc1 = bufAcc_.Get<float>()[kAccSlotFloats];
                if (b < gateBlks) {
                    uint32_t r0 = b * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= E) ? kRowBlk : (E - r0);
                    GemmAccumBf16Ub(wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mCount,
                                    gateWGm_, static_cast<uint64_t>(r0) * D, D, rows, acc0);
                    for (uint32_t m = 0; m < mCount; ++m) {
                        AscendC::LocalTensor<float> ob = outQue_.AllocTensor<float>();
                        AscendC::Muls(ob, acc0[m * kRowBlk], 1.0f, MonoVecAlign8(rows));
                        AscendC::PipeBarrier<PIPE_V>();
                        outQue_.EnQue(ob);
                        ob = outQue_.DeQue<float>();
                        CopyOutTile(wsRouterLogits_, static_cast<uint64_t>(mBase + m) * lStride + r0, ob, rows);
                    }
                } else {
                    uint32_t r0 = (b - gateBlks) * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= meta_.sharedInter) ? kRowBlk : (meta_.sharedInter - r0);
                    GemmAccumBf16PairedUb(wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mCount,
                                          sharedW13Gm_, static_cast<uint64_t>(r0) * D,
                                          static_cast<uint64_t>(meta_.sharedInter + r0) * D,
                                          D, rows, acc0, acc1);
                    ComputeClippedSwigluBatchUb(wsSharedAct_, static_cast<uint64_t>(mBase) * meta_.sharedInter,
                                                meta_.sharedInter, mCount, r0, rows, acc0, acc1);
                }
            } else {
                uint32_t mt = tid - gemmTasksA;
                ComputeHcMixChunk(wsResMid_, hcFfnFnGm_, mt / 3U, (mt % 3U) * 8U);
            }
        }
        BarrierAllCores();

        // FFN Sinkhorn + TopK selection per token
        for (uint32_t m = blkIdx_; m < M; m += blkNum_) {
            ComputeSinkhornOneRow(m, hcFfnScaleGm_, hcFfnBaseGm_, wsFfnPre_, wsFfnPost_, wsFfnComb_);
            SelectTopKPerToken(m);
        }
        BarrierAllCores();

        uint32_t shared2Blks = hasShared ? ((D + kRowBlk - 1) / kRowBlk) : 0U;
        uint32_t shared2Tasks = mTiles * shared2Blks;
        uint32_t routedPairBlksPerExp = (meta_.routedInter + kRowBlk - 1) / kRowBlk;
        uint32_t dRowBlks = (D + kRowBlk - 1) / kRowBlk;
        bool useGroupedRouted = (M > 1) && (K > 0) && (M * K <= 512U);

        if (useGroupedRouted) {
            if (blkIdx_ == 0) {
                BuildExpertGroups(M, K);
            }
            BarrierAllCores();

            uint32_t numGroups = static_cast<uint32_t>(wsExpMeta_.GetValue(0));
            uint32_t routedTasks = numGroups * routedPairBlksPerExp;
            uint32_t totalPool5b = shared2Tasks + routedTasks;

            for (uint32_t tid = blkIdx_; tid < totalPool5b; tid += blkNum_) {
                AscendC::LocalTensor<float> acc0 = bufAcc_.Get<float>()[0];
                AscendC::LocalTensor<float> acc1 = bufAcc_.Get<float>()[kAccSlotFloats];
                if (tid < shared2Tasks) {
                    uint32_t mTile = tid / shared2Blks;
                    uint32_t mBase = mTile * kMonoTileM;
                    uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
                    uint32_t r0 = (tid % shared2Blks) * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= D) ? kRowBlk : (D - r0);
                    GemmAccumBf16Ub(wsSharedAct_, static_cast<uint64_t>(mBase) * meta_.sharedInter, meta_.sharedInter,
                                    mCount, sharedW2Gm_, static_cast<uint64_t>(r0) * meta_.sharedInter,
                                    meta_.sharedInter, rows, acc0);
                    StoreAccBatchBf16Ub(wsSharedOut_, mBase, mCount, D, r0, rows, acc0);
                } else {
                    uint32_t rr = tid - shared2Tasks;
                    uint32_t gIdx = rr / routedPairBlksPerExp;
                    uint32_t r0 = (rr % routedPairBlksPerExp) * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= meta_.routedInter) ? kRowBlk : (meta_.routedInter - r0);

                    AscendC::LocalTensor<int32_t> mUb =
                        CopyInTile(inQueB_, wsExpMeta_, 64ULL + static_cast<uint64_t>(gIdx) * kExpGrpStride, kExpGrpStride);
                    uint32_t eid = static_cast<uint32_t>(mUb.GetValue(0));
                    uint32_t cnt = static_cast<uint32_t>(mUb.GetValue(1));
                    uint32_t tokM[kMonoTileM];
                    uint32_t tokAct[kMonoTileM];
                    for (uint32_t j = 0; j < cnt; ++j) {
                        uint32_t tm = static_cast<uint32_t>(mUb.GetValue(2 + j));
                        uint32_t tk = static_cast<uint32_t>(mUb.GetValue(2 + kMonoTileM + j));
                        tokM[j] = tm;
                        tokAct[j] = tm * K + tk;
                    }
                    inQueB_.FreeTensor(mUb);

                    uint64_t wGateOff = (static_cast<uint64_t>(eid) * (2 * meta_.routedInter) + r0) * D;
                    uint64_t wUpOff = (static_cast<uint64_t>(eid) * (2 * meta_.routedInter) + meta_.routedInter + r0) * D;
                    GemmAccumBf16PairedUb(wsXNorm_, 0, D, cnt, routedW13Gm_, wGateOff, wUpOff, D, rows, acc0, acc1, tokM);
                    ComputeClippedSwigluBatchUb(wsRoutedAct_, 0, meta_.routedInter, cnt, r0, rows, acc0, acc1, tokAct);
                }
            }
            BarrierAllCores();

            // Pool 5c: Expert-grouped routed_w2 across all cores
            uint32_t totalRouted2 = numGroups * dRowBlks;
            for (uint32_t tid = blkIdx_; tid < totalRouted2; tid += blkNum_) {
                uint32_t gIdx = tid / dRowBlks;
                uint32_t d0 = (tid % dRowBlks) * kRowBlk;
                uint32_t rows = (d0 + kRowBlk <= D) ? kRowBlk : (D - d0);

                AscendC::LocalTensor<int32_t> mUb =
                    CopyInTile(inQueB_, wsExpMeta_, 64ULL + static_cast<uint64_t>(gIdx) * kExpGrpStride, kExpGrpStride);
                uint32_t eid = static_cast<uint32_t>(mUb.GetValue(0));
                uint32_t cnt = static_cast<uint32_t>(mUb.GetValue(1));
                uint32_t tokAct[kMonoTileM];
                for (uint32_t j = 0; j < cnt; ++j) {
                    uint32_t tm = static_cast<uint32_t>(mUb.GetValue(2 + j));
                    uint32_t tk = static_cast<uint32_t>(mUb.GetValue(2 + kMonoTileM + j));
                    tokAct[j] = tm * K + tk;
                }
                inQueB_.FreeTensor(mUb);

                AscendC::LocalTensor<float> rAccUb = bufAcc_.Get<float>();
                uint64_t wOff = (static_cast<uint64_t>(eid) * D + d0) * meta_.routedInter;
                GemmAccumBf16Ub(wsRoutedAct_, 0, meta_.routedInter, cnt, routedW2Gm_, wOff, meta_.routedInter,
                                rows, rAccUb, tokAct);
                StoreAccBatchBf16Ub(wsRoutedOut_, 0, cnt, D, d0, rows, rAccUb, tokAct);
            }
            BarrierAllCores();

            VectorizedFinalHcPost(M, D, K, hasShared, dRowBlks, true);
        } else {
            // M == 1 (zero extra grouping barriers!)
            uint32_t routedTasks = M * K * routedPairBlksPerExp;
            uint32_t totalPool5b = shared2Tasks + routedTasks;
            uint64_t tkStride = MonoAlign64(K);

            for (uint32_t tid = blkIdx_; tid < totalPool5b; tid += blkNum_) {
                AscendC::LocalTensor<float> acc0 = bufAcc_.Get<float>()[0];
                AscendC::LocalTensor<float> acc1 = bufAcc_.Get<float>()[kAccSlotFloats];
                if (tid < shared2Tasks) {
                    uint32_t mTile = tid / shared2Blks;
                    uint32_t mBase = mTile * kMonoTileM;
                    uint32_t mCount = (mBase + kMonoTileM <= M) ? kMonoTileM : (M - mBase);
                    uint32_t r0 = (tid % shared2Blks) * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= D) ? kRowBlk : (D - r0);
                    GemmAccumBf16Ub(wsSharedAct_, static_cast<uint64_t>(mBase) * meta_.sharedInter, meta_.sharedInter,
                                    mCount, sharedW2Gm_, static_cast<uint64_t>(r0) * meta_.sharedInter,
                                    meta_.sharedInter, rows, acc0);
                    StoreAccBatchBf16Ub(wsSharedOut_, mBase, mCount, D, r0, rows, acc0);
                } else {
                    uint32_t rr = tid - shared2Tasks;
                    uint32_t mk = rr / routedPairBlksPerExp;
                    uint32_t m = mk / K;
                    uint32_t kSlot = mk % K;
                    uint32_t r0 = (rr % routedPairBlksPerExp) * kRowBlk;
                    uint32_t rows = (r0 + kRowBlk <= meta_.routedInter) ? kRowBlk : (meta_.routedInter - r0);

                    AscendC::LocalTensor<int32_t> idUb =
                        CopyInTile(inQueB_, wsTopkIds_, static_cast<uint64_t>(m) * tkStride, K);
                    uint32_t eid = static_cast<uint32_t>(idUb.GetValue(kSlot));
                    inQueB_.FreeTensor(idUb);

                    uint64_t wGateOff = (static_cast<uint64_t>(eid) * (2 * meta_.routedInter) + r0) * D;
                    uint64_t wUpOff = (static_cast<uint64_t>(eid) * (2 * meta_.routedInter) + meta_.routedInter + r0) * D;
                    GemmAccumBf16PairedUb(wsXNorm_, static_cast<uint64_t>(m) * D, D, 1,
                                          routedW13Gm_, wGateOff, wUpOff, D, rows, acc0, acc1);
                    uint64_t actBase = (static_cast<uint64_t>(m) * K + kSlot) * meta_.routedInter;
                    ComputeClippedSwigluBatchUb(wsRoutedAct_, actBase, meta_.routedInter, 1, r0, rows, acc0, acc1);
                }
            }
            BarrierAllCores();

            VectorizedFinalHcPost(M, D, K, hasShared, dRowBlks, false);
        }

        for (uint32_t m = blkIdx_; m < M; m += blkNum_) {
            AscendC::LocalTensor<float> prUb = CopyInTile(inQueB_, wsFfnPre_, static_cast<uint64_t>(m) * 8, kHcMult);
            AscendC::LocalTensor<float> ob = outQue_.AllocTensor<float>();
            for (uint32_t i = 0; i < 4; ++i) {
                ob.SetValue(i, prUb.GetValue(i));
            }
            inQueB_.FreeTensor(prUb);
            AscendC::PipeBarrier<PIPE_ALL>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<float>();
            CopyOutTile(preMixOutGm_, static_cast<uint64_t>(m) * kHcMult, ob, kHcMult);
        }
    }

private:
    AscendC::TPipe *pipe_;
    uint32_t blkIdx_{0};
    uint32_t blkNum_{1};
    MonoMeta meta_;

    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueA_, inQueB_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufXBatch_, bufWF32_, bufProd_, bufAux0_, bufAux1_, bufWork_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufSmall_, bufSync_, bufAcc_;

    AscendC::GlobalTensor<bfloat16_t> resInGm_, resOutGm_, inputLnWGm_, wqAGm_, qNormWGm_, wqBGm_;
    AscendC::GlobalTensor<bfloat16_t> wkvGm_, kvNormWGm_, woAGm_, woBGm_, postAttnLnWGm_;
    AscendC::GlobalTensor<bfloat16_t> gateWGm_, sharedW13Gm_, sharedW2Gm_, routedW13Gm_, routedW2Gm_, swaKvCacheGm_;
    AscendC::GlobalTensor<float> preMixInGm_, preMixOutGm_, cosGm_, sinGm_, hcAttnFnGm_, hcAttnScaleGm_, hcAttnBaseGm_;
    AscendC::GlobalTensor<float> attnSinkGm_, hcFfnFnGm_, hcFfnScaleGm_, hcFfnBaseGm_, eBiasGm_;
    AscendC::GlobalTensor<int64_t> posGm_, slotGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_, seqLensGm_;

    AscendC::GlobalTensor<bfloat16_t> wsXNorm_, wsResMid_, wsQA_, wsQR_, wsKV_, wsQ_, wsAttnOut_, wsOA_;
    AscendC::GlobalTensor<bfloat16_t> wsSharedAct_, wsSharedOut_, wsRoutedAct_, wsRoutedOut_;
    AscendC::GlobalTensor<float> wsInvRms_, wsMixes_, wsAttnPre_, wsAttnPost_, wsAttnComb_;
    AscendC::GlobalTensor<float> wsFfnPre_, wsFfnPost_, wsFfnComb_, wsRouterLogits_, wsTopkW_;
    AscendC::GlobalTensor<int32_t> wsSync_, wsTopkIds_, wsExpMeta_;
};

} // namespace

extern "C" __global__ __aicore__ void dsv41_mono_layer_bf16(
    __gm__ void* resIn, __gm__ void* preMixIn, __gm__ void* positions,
    __gm__ void* cosIn, __gm__ void* sinIn, __gm__ void* slotMapping,
    __gm__ void* blockTable, __gm__ void* seqLens,
    __gm__ void* resOut, __gm__ void* preMixOut,
    __gm__ void* inputLnW, __gm__ void* hcAttnFn, __gm__ void* hcAttnScale, __gm__ void* hcAttnBase,
    __gm__ void* wqA, __gm__ void* qNormW, __gm__ void* wqB, __gm__ void* wkv, __gm__ void* kvNormW,
    __gm__ void* attnSink, __gm__ void* woA, __gm__ void* woB,
    __gm__ void* postAttnLnW, __gm__ void* hcFfnFn, __gm__ void* hcFfnScale, __gm__ void* hcFfnBase,
    __gm__ void* gateW, __gm__ void* eBias, __gm__ void* sharedW13, __gm__ void* sharedW2,
    __gm__ void* routedW13, __gm__ void* routedW2,
    __gm__ void* swaKvCache, __gm__ void* workspace,
    __gm__ void* metaI64, __gm__ void* metaF32)
{
    AscendC::TPipe pipe;
    DSV41MonoLayerKernel op;
    op.Init(resIn, preMixIn, positions, cosIn, sinIn, slotMapping, blockTable, seqLens, resOut, preMixOut,
            inputLnW, hcAttnFn, hcAttnScale, hcAttnBase, wqA, qNormW, wqB, wkv, kvNormW, attnSink, woA, woB,
            postAttnLnW, hcFfnFn, hcFfnScale, hcFfnBase, gateW, eBias, sharedW13, sharedW2, routedW13, routedW2,
            swaKvCache, workspace, metaI64, metaF32, &pipe);
    op.Process();
}

namespace vllm_ascend {
extern void dsv41_mono_layer_impl(
    void *stream, uint32_t blockDim,
    void *resIn, void *preMixIn, void *positions, void *cosIn, void *sinIn,
    void *slotMapping, void *blockTable, void *seqLens, void *resOut, void *preMixOut,
    void *inputLnW, void *hcAttnFn, void *hcAttnScale, void *hcAttnBase,
    void *wqA, void *qNormW, void *wqB, void *wkv, void *kvNormW,
    void *attnSink, void *woA, void *woB,
    void *postAttnLnW, void *hcFfnFn, void *hcFfnScale, void *hcFfnBase,
    void *gateW, void *eBias, void *sharedW13, void *sharedW2,
    void *routedW13, void *routedW2,
    void *swaKvCache, void *workspace, void *metaI64, void *metaF32)
{
    dsv41_mono_layer_bf16<<<blockDim, nullptr, stream>>>(
        resIn, preMixIn, positions, cosIn, sinIn, slotMapping, blockTable, seqLens, resOut, preMixOut,
        inputLnW, hcAttnFn, hcAttnScale, hcAttnBase, wqA, qNormW, wqB, wkv, kvNormW, attnSink, woA, woB,
        postAttnLnW, hcFfnFn, hcFfnScale, hcFfnBase, gateW, eBias, sharedW13, sharedW2, routedW13, routedW2,
        swaKvCache, workspace, metaI64, metaF32);
}
} // namespace vllm_ascend

#endif
