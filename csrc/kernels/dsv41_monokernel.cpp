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

constexpr uint32_t kMonoTileK = 2048;
constexpr uint32_t kMonoSyncInts = 1024;
constexpr uint32_t kRowBlk = 16;
constexpr uint32_t kHcMult = 4;
constexpr uint32_t kHcMixDim = 24;
constexpr uint32_t kHcCombDim = 16;
constexpr uint32_t kMonoAlign = 64;

__aicore__ inline uint64_t MonoAlign64(uint64_t val)
{
    return ((val + kMonoAlign - 1) / kMonoAlign) * kMonoAlign;
}

__aicore__ inline uint32_t MonoVecAlign8(uint32_t val)
{
    return ((val + 7U) / 8U) * 8U;
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
        Stage1PreAndAttnMix();
        BarrierAllCores();
        Stage2QkvProjNormRopeCache();
        BarrierAllCores();
        Stage3FlashSwaAndOProj();
        BarrierAllCores();
        Stage4MidSeamAndFfnMix();
        BarrierAllCores();
        Stage5MoEAndFinalPost();
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
        wsXAttn_.SetGlobalBuffer(AllocWsB(base, off, M * D));
        wsFfnPre_.SetGlobalBuffer(AllocWsF(base, off, M * 8));
        wsFfnPost_.SetGlobalBuffer(AllocWsF(base, off, M * 8));
        wsFfnComb_.SetGlobalBuffer(AllocWsF(base, off, M * kHcCombDim));
        wsRouterLogits_.SetGlobalBuffer(AllocWsF(base, off, M * MonoAlign64(meta_.nExperts)));
        wsTopkIds_.SetGlobalBuffer(AllocWsI(base, off, M * MonoAlign64(meta_.topK)));
        wsTopkW_.SetGlobalBuffer(AllocWsF(base, off, M * MonoAlign64(meta_.topK)));
        wsSharedGU_.SetGlobalBuffer(AllocWsB(base, off, M * 2 * meta_.sharedInter));
        wsSharedAct_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.sharedInter));
        wsSharedOut_.SetGlobalBuffer(AllocWsB(base, off, M * D));
        wsRoutedGU_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.topK * 2 * meta_.routedInter));
        wsRoutedAct_.SetGlobalBuffer(AllocWsB(base, off, M * meta_.topK * meta_.routedInter));
    }

    __aicore__ inline void SetupPipeBuffers()
    {
        uint32_t tileBytes = kMonoTileK * sizeof(float);
        pipe_->InitBuffer(inQueA_, 2, tileBytes);
        pipe_->InitBuffer(inQueB_, 2, tileBytes);
        pipe_->InitBuffer(outQue_, 1, tileBytes);

        pipe_->InitBuffer(bufXF32_, tileBytes);
        pipe_->InitBuffer(bufWF32_, tileBytes);
        pipe_->InitBuffer(bufAcc_, tileBytes);
        pipe_->InitBuffer(bufAux0_, tileBytes);
        pipe_->InitBuffer(bufAux1_, tileBytes);
        pipe_->InitBuffer(bufWork_, tileBytes);
        pipe_->InitBuffer(bufSmall_, 256 * sizeof(float));
        pipe_->InitBuffer(bufOutRow_, kRowBlk * sizeof(float));
        pipe_->InitBuffer(bufSync_, kMonoSyncInts * sizeof(int32_t));
    }

    __aicore__ inline float ScalarExp(float x)
    {
        AscendC::LocalTensor<float> sm = bufSmall_.Get<float>();
        sm.SetValue(0, x);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(sm, sm, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        return sm.GetValue(0);
    }

    __aicore__ inline float ScalarRsqrt(float x)
    {
        AscendC::LocalTensor<float> sm = bufSmall_.Get<float>();
        sm.SetValue(0, x);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sqrt(sm, sm, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        return 1.0f / sm.GetValue(0);
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> CopyInTile(
        AscendC::TQue<AscendC::QuePosition::VECIN, 2> &que,
        const AscendC::GlobalTensor<T> &gm,
        uint64_t offset,
        uint32_t count)
    {
        AscendC::LocalTensor<T> ub = que.AllocTensor<T>();
        uint16_t bytes = static_cast<uint16_t>(count * sizeof(T));
        AscendC::DataCopyPad(ub, gm[offset], {1, bytes, 0, 0}, {false, 0, 0, 0});
        que.EnQue(ub);
        return que.DeQue<T>();
    }

    template <typename T>
    __aicore__ inline void CopyOutTile(
        const AscendC::GlobalTensor<T> &gm,
        uint64_t offset,
        AscendC::LocalTensor<T> &ub,
        uint32_t count)
    {
        uint16_t bytes = static_cast<uint16_t>(count * sizeof(T));
        AscendC::DataCopyPad(gm[offset], ub, {1, bytes, 0, 0});
        outQue_.FreeTensor(ub);
    }

    // Tile-batched BF16 GEMV: computes up to `rows` (<= kRowBlk) rows of W @ x^T.
    __aicore__ inline void GemvRowBlockBf16(
        const AscendC::GlobalTensor<bfloat16_t> &xGm,
        uint64_t xBase,
        const AscendC::GlobalTensor<bfloat16_t> &wGm,
        uint64_t wBase,
        uint32_t K,
        uint32_t rows,
        AscendC::LocalTensor<float> &outRowF32)
    {
        AscendC::LocalTensor<float> xf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> wf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();

        uint32_t reduceLen = (K >= kMonoTileK) ? kMonoTileK : MonoVecAlign8(K);
        if (K <= kMonoTileK) {
            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, xGm, xBase, K);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, K);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);
            if (K < reduceLen) {
                for (uint32_t i = K; i < reduceLen; ++i) {
                    xf.SetValue(i, 0.0f);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::LocalTensor<bfloat16_t> wb = CopyInTile(inQueB_, wGm, wBase + static_cast<uint64_t>(r) * K, K);
                AscendC::Cast(wf, wb, AscendC::RoundMode::CAST_NONE, K);
                AscendC::PipeBarrier<PIPE_V>();
                inQueB_.FreeTensor(wb);
                AscendC::Mul(wf, xf, wf, K);
                AscendC::PipeBarrier<PIPE_V>();
                if (K < reduceLen) {
                    for (uint32_t i = K; i < reduceLen; ++i) {
                        wf.SetValue(i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::ReduceSum<float>(wf, wf, work, reduceLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                outRowF32.SetValue(r, wf.GetValue(0));
            }
            return;
        }

        for (uint32_t r = 0; r < rows; ++r) {
            outRowF32.SetValue(r, 0.0f);
        }
        for (uint32_t k0 = 0; k0 < K; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= K) ? kMonoTileK : (K - k0);
            uint32_t rLen = (len == kMonoTileK) ? kMonoTileK : MonoVecAlign8(len);
            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, xGm, xBase + k0, len);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::LocalTensor<bfloat16_t> wb =
                    CopyInTile(inQueB_, wGm, wBase + static_cast<uint64_t>(r) * K + k0, len);
                AscendC::Cast(wf, wb, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueB_.FreeTensor(wb);
                AscendC::Mul(wf, xf, wf, len);
                AscendC::PipeBarrier<PIPE_V>();
                if (len < rLen) {
                    for (uint32_t i = len; i < rLen; ++i) {
                        wf.SetValue(i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::ReduceSum<float>(acc, wf, work, rLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                outRowF32.SetValue(r, outRowF32.GetValue(r) + acc.GetValue(0));
            }
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
        AscendC::LocalTensor<float> xf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> wf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> red = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();
        float sumSq = 0.0f;

        for (uint32_t k0 = 0; k0 < K; k0 += kMonoTileK) {
            uint32_t len = (k0 + kMonoTileK <= K) ? kMonoTileK : (K - k0);
            uint32_t rLen = (len == kMonoTileK) ? kMonoTileK : MonoVecAlign8(len);
            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, inGm, inBase + k0, len);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);
            AscendC::Mul(xf, xf, xf, len);
            AscendC::PipeBarrier<PIPE_V>();
            if (len < rLen) {
                for (uint32_t i = len; i < rLen; ++i) {
                    xf.SetValue(i, 0.0f);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::ReduceSum<float>(red, xf, work, rLen);
            AscendC::PipeBarrier<PIPE_ALL>();
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
        AscendC::LocalTensor<float> xf = bufXF32_.Get<float>();
        AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, ropeDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        inQueA_.FreeTensor(xb);

        AscendC::LocalTensor<float> sn = CopyInTile(inQueB_, sinGm_, cosSinBase, ropeDim);
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();
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

    __aicore__ inline void ComputeHcMixProject(
        const AscendC::GlobalTensor<bfloat16_t> &resGm,
        const AscendC::GlobalTensor<float> &hcFnGm)
    {
        uint32_t M = meta_.mTokens;
        uint32_t hcD = kHcMult * meta_.hidden;
        uint32_t chunksPerToken = kHcMixDim / 8;
        uint32_t totalTasks = M * chunksPerToken;

        AscendC::LocalTensor<float> xf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalTasks; tid += blkNum_) {
            uint32_t m = tid / chunksPerToken;
            uint32_t cBase = (tid % chunksPerToken) * 8;
            AscendC::LocalTensor<float> invUb = CopyInTile(inQueB_, wsInvRms_, m, 1);
            float invRms = invUb.GetValue(0);
            inQueB_.FreeTensor(invUb);

            float out8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t k0 = 0; k0 < hcD; k0 += kMonoTileK) {
                uint32_t len = (k0 + kMonoTileK <= hcD) ? kMonoTileK : (hcD - k0);
                uint32_t rLen = (len == kMonoTileK) ? kMonoTileK : MonoVecAlign8(len);
                AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, resGm, static_cast<uint64_t>(m) * hcD + k0, len);
                AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(xb);

                for (uint32_t c = 0; c < 8; ++c) {
                    AscendC::LocalTensor<float> wf =
                        CopyInTile(inQueB_, hcFnGm, static_cast<uint64_t>(cBase + c) * hcD + k0, len);
                    AscendC::Mul(acc, xf, wf, len);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueB_.FreeTensor(wf);
                    if (len < rLen) {
                        for (uint32_t i = len; i < rLen; ++i) {
                            acc.SetValue(i, 0.0f);
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    AscendC::ReduceSum<float>(acc, acc, work, rLen);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    out8[c] += acc.GetValue(0);
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
    }

    __aicore__ inline void ComputeSinkhornRows(
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

        for (uint32_t m = blkIdx_; m < meta_.mTokens; m += blkNum_) {
            AscendC::LocalTensor<float> mx = CopyInTile(inQueA_, wsMixes_, static_cast<uint64_t>(m) * 32, kHcMixDim);
            float pre4[4];
            float post4[4];
            for (uint32_t i = 0; i < 4; ++i) {
                float pRaw = mx.GetValue(i) * s0 + bVals[i];
                pre4[i] = 1.0f / (1.0f + ScalarExp(-pRaw)) + meta_.hcEps;
                float poRaw = mx.GetValue(4 + i) * s1 + bVals[4 + i];
                post4[i] = 2.0f / (1.0f + ScalarExp(-poRaw));
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
                float rowSum = 0.0f;
                for (uint32_t c = 0; c < 4; ++c) {
                    cm[r][c] = ScalarExp(cm[r][c] - rowMax);
                    rowSum += cm[r][c];
                }
                float invS = 1.0f / rowSum;
                for (uint32_t c = 0; c < 4; ++c) {
                    cm[r][c] = cm[r][c] * invS + meta_.hcEps;
                }
            }
            inQueA_.FreeTensor(mx);

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
        AscendC::LocalTensor<float> xf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> uAcc = bufAcc_.Get<float>();
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
                uint32_t rLen = (len == kMonoTileK) ? kMonoTileK : MonoVecAlign8(len);

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

                // Store collapsed u BF16 temporarily in wsXNorm_[m, d0..] then compute RMS
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
                if (len < rLen) {
                    for (uint32_t i = len; i < rLen; ++i) {
                        uTile.SetValue(i, 0.0f);
                        sq4d.SetValue(i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::ReduceSum<float>(uAcc, uTile, work, rLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                sumSqU += uAcc.GetValue(0);

                AscendC::ReduceSum<float>(uAcc, sq4d, work, rLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                sumSq4D += uAcc.GetValue(0);
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

    __aicore__ inline void Stage1PreAndAttnMix()
    {
        CollapseNormAndInvRms(resInGm_, preMixInGm_, kHcMult, inputLnWGm_);
        BarrierAllCores();
        ComputeHcMixProject(resInGm_, hcAttnFnGm_);
        BarrierAllCores();
        ComputeSinkhornRows(hcAttnScaleGm_, hcAttnBaseGm_, wsAttnPre_, wsAttnPost_, wsAttnComb_);
    }

    __aicore__ inline void RunGemvOutBlocks(
        const AscendC::GlobalTensor<bfloat16_t> &xGm,
        uint32_t xStride,
        const AscendC::GlobalTensor<bfloat16_t> &wGm,
        const AscendC::GlobalTensor<bfloat16_t> &outGm,
        uint32_t K,
        uint32_t N)
    {
        uint32_t rowBlks = (N + kRowBlk - 1) / kRowBlk;
        uint32_t total = meta_.mTokens * rowBlks;
        AscendC::LocalTensor<float> outF = bufOutRow_.Get<float>();
        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t m = tid / rowBlks;
            uint32_t r0 = (tid % rowBlks) * kRowBlk;
            uint32_t rows = (r0 + kRowBlk <= N) ? kRowBlk : (N - r0);
            GemvRowBlockBf16(
                xGm, static_cast<uint64_t>(m) * xStride,
                wGm, static_cast<uint64_t>(r0) * K, K, rows, outF);
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(outGm, static_cast<uint64_t>(m) * N + r0, ob, rows);
        }
    }

    __aicore__ inline void Stage2QkvProjNormRopeCache()
    {
        uint32_t M = meta_.mTokens;
        uint32_t D = meta_.hidden;
        uint32_t qBlks = (meta_.qLora + kRowBlk - 1) / kRowBlk;
        uint32_t kvBlks = (meta_.headDim + kRowBlk - 1) / kRowBlk;
        uint32_t totalQkv = M * (qBlks + kvBlks);
        AscendC::LocalTensor<float> outF = bufOutRow_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalQkv; tid += blkNum_) {
            uint32_t m = tid / (qBlks + kvBlks);
            uint32_t b = tid % (qBlks + kvBlks);
            if (b < qBlks) {
                uint32_t r0 = b * kRowBlk;
                uint32_t rows = (r0 + kRowBlk <= meta_.qLora) ? kRowBlk : (meta_.qLora - r0);
                GemvRowBlockBf16(wsXNorm_, static_cast<uint64_t>(m) * D, wqAGm_, static_cast<uint64_t>(r0) * D, D, rows, outF);
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(wsQA_, static_cast<uint64_t>(m) * meta_.qLora + r0, ob, rows);
            } else {
                uint32_t r0 = (b - qBlks) * kRowBlk;
                uint32_t rows = (r0 + kRowBlk <= meta_.headDim) ? kRowBlk : (meta_.headDim - r0);
                GemvRowBlockBf16(wsXNorm_, static_cast<uint64_t>(m) * D, wkvGm_, static_cast<uint64_t>(r0) * D, D, rows, outF);
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(wsKV_, static_cast<uint64_t>(m) * meta_.headDim + r0, ob, rows);
            }
        }
        BarrierAllCores();

        // q_norm, kv_norm, kv RoPE, and paged SWA cache scatter
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
        }
        BarrierAllCores();

        // q_b GEMV: wsQR_ [M, qLora] -> wsQ_ [M, nHeads * headDim]
        RunGemvOutBlocks(wsQR_, meta_.qLora, wqBGm_, wsQ_, meta_.qLora, meta_.nHeads * meta_.headDim);
        BarrierAllCores();

        // Apply interleaved partial RoPE across all (m, h) heads
        uint32_t totalHeads = M * meta_.nHeads;
        for (uint32_t tid = blkIdx_; tid < totalHeads; tid += blkNum_) {
            uint32_t m = tid / meta_.nHeads;
            uint32_t h = tid % meta_.nHeads;
            uint64_t ropeOff = (static_cast<uint64_t>(m) * meta_.nHeads + h) * meta_.headDim + meta_.nopeDim;
            ApplyInterleavedRopeInPlace(wsQ_, ropeOff, static_cast<uint64_t>(m) * meta_.ropeDim, meta_.ropeDim);
        }
    }

    __aicore__ inline void Stage3FlashSwaAndOProj()
    {
        uint32_t M = meta_.mTokens;
        uint32_t H = meta_.nHeads;
        uint32_t hd = meta_.headDim;
        uint32_t rLen = (hd == kMonoTileK) ? kMonoTileK : MonoVecAlign8(hd);
        uint32_t totalHeads = M * H;

        AscendC::LocalTensor<float> qf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> kf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> dotRes = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> work = bufWork_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalHeads; tid += blkNum_) {
            uint32_t m = tid / H;
            uint32_t h = tid % H;
            uint64_t qOff = static_cast<uint64_t>(tid) * hd;

            AscendC::LocalTensor<bfloat16_t> qb = CopyInTile(inQueA_, wsQ_, qOff, hd);
            AscendC::Cast(qf, qb, AscendC::RoundMode::CAST_NONE, hd);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(qb);
            if (hd < rLen) {
                for (uint32_t i = hd; i < rLen; ++i) {
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
            for (int32_t t = startT; t < seqLen; ++t) {
                uint32_t blkIdxLogical = static_cast<uint32_t>(t) / meta_.blockSize;
                uint32_t blkOff = static_cast<uint32_t>(t) % meta_.blockSize;
                int32_t physBlk = blockTableGm_.GetValue(static_cast<uint64_t>(m) * meta_.maxBlocks + blkIdxLogical);
                if (physBlk < 0) {
                    continue;
                }
                uint64_t kvOff = (static_cast<uint64_t>(physBlk) * meta_.blockSize + blkOff) * hd;
                AscendC::LocalTensor<bfloat16_t> kb = CopyInTile(inQueB_, swaKvCacheGm_, kvOff, hd);
                AscendC::Cast(kf, kb, AscendC::RoundMode::CAST_NONE, hd);
                AscendC::PipeBarrier<PIPE_V>();
                inQueB_.FreeTensor(kb);

                AscendC::Mul(dotRes, qf, kf, hd);
                AscendC::PipeBarrier<PIPE_V>();
                if (hd < rLen) {
                    for (uint32_t i = hd; i < rLen; ++i) {
                        dotRes.SetValue(i, 0.0f);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::ReduceSum<float>(dotRes, dotRes, work, rLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                float score = dotRes.GetValue(0) * meta_.softmaxScale;

                float newMax = (score > mMax) ? score : mMax;
                float alpha = (lSum > 0.0f) ? ScalarExp(mMax - newMax) : 0.0f;
                float pVal = ScalarExp(score - newMax);
                lSum = lSum * alpha + pVal;
                mMax = newMax;

                AscendC::Muls(acc, acc, alpha, hd);
                AscendC::Muls(kf, kf, pVal, hd);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(acc, acc, kf, hd);
                AscendC::PipeBarrier<PIPE_V>();
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

        // Grouped wo_a: [oGroups, oLora, groupHidden] applied to wsAttnOut_ [M, oGroups, groupHidden]
        uint32_t groupHidden = (meta_.nHeads * meta_.headDim) / meta_.oGroups;
        uint32_t gRowBlks = (meta_.oLora + kRowBlk - 1) / kRowBlk;
        uint32_t totalOA = M * meta_.oGroups * gRowBlks;
        AscendC::LocalTensor<float> outF = bufOutRow_.Get<float>();
        for (uint32_t tid = blkIdx_; tid < totalOA; tid += blkNum_) {
            uint32_t mg = tid / gRowBlks;
            uint32_t m = mg / meta_.oGroups;
            uint32_t g = mg % meta_.oGroups;
            uint32_t r0 = (tid % gRowBlks) * kRowBlk;
            uint32_t rows = (r0 + kRowBlk <= meta_.oLora) ? kRowBlk : (meta_.oLora - r0);
            uint64_t xOff = (static_cast<uint64_t>(m) * meta_.oGroups + g) * groupHidden;
            uint64_t wOff = (static_cast<uint64_t>(g) * meta_.oLora + r0) * groupHidden;
            GemvRowBlockBf16(wsAttnOut_, xOff, woAGm_, wOff, groupHidden, rows, outF);
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(wsOA_, (static_cast<uint64_t>(m) * meta_.oGroups + g) * meta_.oLora + r0, ob, rows);
        }
        BarrierAllCores();

        // wo_b: [hidden, oGroups * oLora]
        RunGemvOutBlocks(wsOA_, meta_.oGroups * meta_.oLora, woBGm_, wsXAttn_, meta_.oGroups * meta_.oLora, meta_.hidden);
    }

    __aicore__ inline void RunHcPostRange(
        const AscendC::GlobalTensor<bfloat16_t> &xGm,
        const AscendC::GlobalTensor<bfloat16_t> &resInGm,
        const AscendC::GlobalTensor<float> &postGm,
        const AscendC::GlobalTensor<float> &combGm,
        const AscendC::GlobalTensor<bfloat16_t> &resOutGm)
    {
        uint32_t D = meta_.hidden;
        uint32_t dTiles = (D + kMonoTileK - 1) / kMonoTileK;
        uint32_t total = meta_.mTokens * dTiles;

        AscendC::LocalTensor<float> xf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> rf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> yBuf = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> tmp = bufAux0_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t m = tid / dTiles;
            uint32_t d0 = (tid % dTiles) * kMonoTileK;
            uint32_t len = (d0 + kMonoTileK <= D) ? kMonoTileK : (D - d0);

            AscendC::LocalTensor<float> pUb = CopyInTile(inQueB_, postGm, static_cast<uint64_t>(m) * 8, kHcMult);
            float post4[4];
            for (uint32_t i = 0; i < 4; ++i) {
                post4[i] = pUb.GetValue(i);
            }
            inQueB_.FreeTensor(pUb);

            AscendC::LocalTensor<float> cUb = CopyInTile(inQueB_, combGm, static_cast<uint64_t>(m) * kHcCombDim, kHcCombDim);
            float comb44[4][4];
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t c = 0; c < 4; ++c) {
                    comb44[r][c] = cUb.GetValue(r * 4 + c);
                }
            }
            inQueB_.FreeTensor(cUb);

            AscendC::LocalTensor<bfloat16_t> xb = CopyInTile(inQueA_, xGm, static_cast<uint64_t>(m) * D + d0, len);
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);

            uint64_t rBase = static_cast<uint64_t>(m) * kHcMult * D + d0;
            for (uint32_t j = 0; j < 4; ++j) {
                AscendC::Muls(yBuf, xf, post4[j], len);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < 4; ++i) {
                    AscendC::LocalTensor<bfloat16_t> rb =
                        CopyInTile(inQueA_, resInGm, rBase + static_cast<uint64_t>(i) * D, len);
                    AscendC::Cast(rf, rb, AscendC::RoundMode::CAST_NONE, len);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueA_.FreeTensor(rb);
                    AscendC::Muls(tmp, rf, comb44[i][j], len);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(yBuf, yBuf, tmp, len);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, yBuf, AscendC::RoundMode::CAST_RINT, len);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(resOutGm, rBase + static_cast<uint64_t>(j) * D, ob, len);
            }
        }
    }

    __aicore__ inline void Stage4MidSeamAndFfnMix()
    {
        RunHcPostRange(wsXAttn_, resInGm_, wsAttnPost_, wsAttnComb_, wsResMid_);
        BarrierAllCores();
        CollapseNormAndInvRms(wsResMid_, wsAttnPre_, 8, postAttnLnWGm_);
        BarrierAllCores();
        ComputeHcMixProject(wsResMid_, hcFfnFnGm_);
        BarrierAllCores();
        ComputeSinkhornRows(hcFfnScaleGm_, hcFfnBaseGm_, wsFfnPre_, wsFfnPost_, wsFfnComb_);
    }

    __aicore__ inline void ApplySiluAndMulRow(
        const AscendC::GlobalTensor<bfloat16_t> &guGm,
        uint64_t guBase,
        const AscendC::GlobalTensor<bfloat16_t> &actGm,
        uint64_t actBase,
        uint32_t I)
    {
        AscendC::LocalTensor<float> gf = bufXF32_.Get<float>();
        AscendC::LocalTensor<float> uf = bufWF32_.Get<float>();
        AscendC::LocalTensor<float> sig = bufAcc_.Get<float>();
        float lim = meta_.swigluLimit;

        for (uint32_t i0 = 0; i0 < I; i0 += kMonoTileK) {
            uint32_t len = (i0 + kMonoTileK <= I) ? kMonoTileK : (I - i0);
            AscendC::LocalTensor<bfloat16_t> gb = CopyInTile(inQueA_, guGm, guBase + i0, len);
            AscendC::LocalTensor<bfloat16_t> ub = CopyInTile(inQueB_, guGm, guBase + I + i0, len);
            AscendC::Cast(gf, gb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(uf, ub, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(gb);
            inQueB_.FreeTensor(ub);

            if (lim > 0.0f) {
                AscendC::Mins(gf, gf, lim, len);
                AscendC::Maxs(uf, uf, -lim, len);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(uf, uf, lim, len);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // sigmoid(gf) = 1 / (1 + exp(-gf))
            AscendC::Muls(sig, gf, -1.0f, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(sig, sig, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sig, sig, 1.0f, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(gf, gf, sig, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(gf, gf, uf, len);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, gf, AscendC::RoundMode::CAST_RINT, len);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(actGm, actBase + i0, ob, len);
        }
    }

    __aicore__ inline void SelectTopKPerToken(uint32_t m)
    {
        uint32_t E = meta_.nExperts;
        uint32_t K = meta_.topK;
        uint64_t lStride = MonoAlign64(E);
        AscendC::LocalTensor<float> lg = CopyInTile(inQueA_, wsRouterLogits_, static_cast<uint64_t>(m) * lStride, E);
        AscendC::LocalTensor<float> scores = bufXF32_.Get<float>();

        if (meta_.useSigmoid) {
            for (uint32_t e = 0; e < E; ++e) {
                scores.SetValue(e, 1.0f / (1.0f + ScalarExp(-lg.GetValue(e))));
            }
        } else {
            float lMax = -1e30f;
            for (uint32_t e = 0; e < E; ++e) {
                float v = lg.GetValue(e);
                if (v > lMax) {
                    lMax = v;
                }
            }
            float lSum = 0.0f;
            for (uint32_t e = 0; e < E; ++e) {
                float ex = ScalarExp(lg.GetValue(e) - lMax);
                scores.SetValue(e, ex);
                lSum += ex;
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
        AscendC::LocalTensor<float> obW = bufAcc_.Get<float>();
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

    __aicore__ inline void Stage5MoEAndFinalPost()
    {
        uint32_t M = meta_.mTokens;
        uint32_t D = meta_.hidden;
        uint32_t E = meta_.nExperts;
        uint32_t K = meta_.topK;
        uint32_t hasShared = (meta_.sharedInter > 0) ? 1U : 0U;

        uint32_t gateBlks = (E + kRowBlk - 1) / kRowBlk;
        uint32_t shared13Blks = hasShared ? ((2 * meta_.sharedInter + kRowBlk - 1) / kRowBlk) : 0U;
        uint32_t totalS5a = M * (gateBlks + shared13Blks);
        uint64_t lStride = MonoAlign64(E);
        AscendC::LocalTensor<float> outF = bufOutRow_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalS5a; tid += blkNum_) {
            uint32_t m = tid / (gateBlks + shared13Blks);
            uint32_t b = tid % (gateBlks + shared13Blks);
            if (b < gateBlks) {
                uint32_t r0 = b * kRowBlk;
                uint32_t rows = (r0 + kRowBlk <= E) ? kRowBlk : (E - r0);
                GemvRowBlockBf16(wsXNorm_, static_cast<uint64_t>(m) * D, gateWGm_, static_cast<uint64_t>(r0) * D, D, rows, outF);
                AscendC::LocalTensor<float> ob = outQue_.AllocTensor<float>();
                for (uint32_t r = 0; r < rows; ++r) {
                    ob.SetValue(r, outF.GetValue(r));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<float>();
                CopyOutTile(wsRouterLogits_, static_cast<uint64_t>(m) * lStride + r0, ob, rows);
            } else {
                uint32_t r0 = (b - gateBlks) * kRowBlk;
                uint32_t rows = (r0 + kRowBlk <= 2 * meta_.sharedInter) ? kRowBlk : (2 * meta_.sharedInter - r0);
                GemvRowBlockBf16(wsXNorm_, static_cast<uint64_t>(m) * D, sharedW13Gm_, static_cast<uint64_t>(r0) * D, D, rows, outF);
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(wsSharedGU_, static_cast<uint64_t>(m) * 2 * meta_.sharedInter + r0, ob, rows);
            }
        }
        BarrierAllCores();

        for (uint32_t m = blkIdx_; m < M; m += blkNum_) {
            SelectTopKPerToken(m);
            if (hasShared) {
                ApplySiluAndMulRow(
                    wsSharedGU_, static_cast<uint64_t>(m) * 2 * meta_.sharedInter,
                    wsSharedAct_, static_cast<uint64_t>(m) * meta_.sharedInter, meta_.sharedInter);
            }
        }
        BarrierAllCores();

        uint32_t shared2Blks = hasShared ? ((D + kRowBlk - 1) / kRowBlk) : 0U;
        uint32_t routed13Blks = K * ((2 * meta_.routedInter + kRowBlk - 1) / kRowBlk);
        uint32_t perTokS5b = shared2Blks + routed13Blks;
        uint32_t totalS5b = M * perTokS5b;
        uint64_t tkStride = MonoAlign64(K);

        for (uint32_t tid = blkIdx_; tid < totalS5b; tid += blkNum_) {
            uint32_t m = tid / perTokS5b;
            uint32_t rem = tid % perTokS5b;
            if (rem < shared2Blks) {
                uint32_t r0 = rem * kRowBlk;
                uint32_t rows = (r0 + kRowBlk <= D) ? kRowBlk : (D - r0);
                GemvRowBlockBf16(wsSharedAct_, static_cast<uint64_t>(m) * meta_.sharedInter, sharedW2Gm_,
                                 static_cast<uint64_t>(r0) * meta_.sharedInter, meta_.sharedInter, rows, outF);
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(wsSharedOut_, static_cast<uint64_t>(m) * D + r0, ob, rows);
            } else {
                uint32_t rr = rem - shared2Blks;
                uint32_t rBlksPerExp = (2 * meta_.routedInter + kRowBlk - 1) / kRowBlk;
                uint32_t kSlot = rr / rBlksPerExp;
                uint32_t r0 = (rr % rBlksPerExp) * kRowBlk;
                uint32_t rows = (r0 + kRowBlk <= 2 * meta_.routedInter) ? kRowBlk : (2 * meta_.routedInter - r0);

                AscendC::LocalTensor<int32_t> idUb =
                    CopyInTile(inQueB_, wsTopkIds_, static_cast<uint64_t>(m) * tkStride, K);
                uint32_t eid = static_cast<uint32_t>(idUb.GetValue(kSlot));
                inQueB_.FreeTensor(idUb);

                uint64_t wOff = (static_cast<uint64_t>(eid) * (2 * meta_.routedInter) + r0) * D;
                GemvRowBlockBf16(wsXNorm_, static_cast<uint64_t>(m) * D, routedW13Gm_, wOff, D, rows, outF);
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, outF, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                uint64_t outOff = ((static_cast<uint64_t>(m) * K + kSlot) * (2 * meta_.routedInter)) + r0;
                CopyOutTile(wsRoutedGU_, outOff, ob, rows);
            }
        }
        BarrierAllCores();

        // Routed SiluAndMulWithClamp
        uint32_t totalPairs = M * K;
        for (uint32_t tid = blkIdx_; tid < totalPairs; tid += blkNum_) {
            ApplySiluAndMulRow(
                wsRoutedGU_, static_cast<uint64_t>(tid) * 2 * meta_.routedInter,
                wsRoutedAct_, static_cast<uint64_t>(tid) * meta_.routedInter, meta_.routedInter);
        }
        BarrierAllCores();

        // Routed W2 combine + shared_out + final hc_post -> resOutGm_ + preMixOutGm_
        uint32_t dRowBlks = (D + kRowBlk - 1) / kRowBlk;
        uint32_t totalFinal = M * dRowBlks;
        for (uint32_t tid = blkIdx_; tid < totalFinal; tid += blkNum_) {
            uint32_t m = tid / dRowBlks;
            uint32_t d0 = (tid % dRowBlks) * kRowBlk;
            uint32_t rows = (d0 + kRowBlk <= D) ? kRowBlk : (D - d0);

            float combOut[kRowBlk];
            if (hasShared) {
                AscendC::LocalTensor<bfloat16_t> sb =
                    CopyInTile(inQueA_, wsSharedOut_, static_cast<uint64_t>(m) * D + d0, rows);
                AscendC::LocalTensor<float> sf = bufAux1_.Get<float>();
                AscendC::Cast(sf, sb, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_ALL>();
                inQueA_.FreeTensor(sb);
                for (uint32_t r = 0; r < rows; ++r) {
                    combOut[r] = sf.GetValue(r);
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    combOut[r] = 0.0f;
                }
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

            for (uint32_t k = 0; k < K; ++k) {
                uint64_t wOff = (static_cast<uint64_t>(ids[k]) * D + d0) * meta_.routedInter;
                uint64_t actOff = (static_cast<uint64_t>(m) * K + k) * meta_.routedInter;
                GemvRowBlockBf16(wsRoutedAct_, actOff, routedW2Gm_, wOff, meta_.routedInter, rows, outF);
                for (uint32_t r = 0; r < rows; ++r) {
                    combOut[r] += ws[k] * outF.GetValue(r);
                }
            }

            // Fuse final hc_post directly using combOut (zero GM roundtrip for x_moe_out!)
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

            float rIn[4][kRowBlk];
            AscendC::LocalTensor<float> rf = bufAux1_.Get<float>();
            uint64_t rBase = static_cast<uint64_t>(m) * kHcMult * D + d0;
            for (uint32_t i = 0; i < 4; ++i) {
                AscendC::LocalTensor<bfloat16_t> rb =
                    CopyInTile(inQueA_, wsResMid_, rBase + static_cast<uint64_t>(i) * D, rows);
                AscendC::Cast(rf, rb, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_ALL>();
                inQueA_.FreeTensor(rb);
                for (uint32_t r = 0; r < rows; ++r) {
                    rIn[i][r] = rf.GetValue(r);
                }
            }

            for (uint32_t j = 0; j < 4; ++j) {
                for (uint32_t r = 0; r < rows; ++r) {
                    rf.SetValue(r, post4[j] * combOut[r] + cm44[0][j] * rIn[0][r] + cm44[1][j] * rIn[1][r] +
                                       cm44[2][j] * rIn[2][r] + cm44[3][j] * rIn[3][r]);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(ob, rf, AscendC::RoundMode::CAST_RINT, rows);
                AscendC::PipeBarrier<PIPE_V>();
                outQue_.EnQue(ob);
                ob = outQue_.DeQue<bfloat16_t>();
                CopyOutTile(resOutGm_, rBase + static_cast<uint64_t>(j) * D, ob, rows);
            }
        }

        // Write preMixOutGm_ [M, 4] from wsFfnPre_ [M, 8]
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
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufXF32_, bufWF32_, bufAcc_, bufAux0_, bufAux1_, bufWork_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufSmall_, bufOutRow_, bufSync_;

    AscendC::GlobalTensor<bfloat16_t> resInGm_, resOutGm_, inputLnWGm_, wqAGm_, qNormWGm_, wqBGm_;
    AscendC::GlobalTensor<bfloat16_t> wkvGm_, kvNormWGm_, woAGm_, woBGm_, postAttnLnWGm_;
    AscendC::GlobalTensor<bfloat16_t> gateWGm_, sharedW13Gm_, sharedW2Gm_, routedW13Gm_, routedW2Gm_, swaKvCacheGm_;
    AscendC::GlobalTensor<float> preMixInGm_, preMixOutGm_, cosGm_, sinGm_, hcAttnFnGm_, hcAttnScaleGm_, hcAttnBaseGm_;
    AscendC::GlobalTensor<float> attnSinkGm_, hcFfnFnGm_, hcFfnScaleGm_, hcFfnBaseGm_, eBiasGm_;
    AscendC::GlobalTensor<int64_t> posGm_, slotGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_, seqLensGm_;

    AscendC::GlobalTensor<bfloat16_t> wsXNorm_, wsResMid_, wsQA_, wsQR_, wsKV_, wsQ_, wsAttnOut_, wsOA_, wsXAttn_;
    AscendC::GlobalTensor<bfloat16_t> wsSharedGU_, wsSharedAct_, wsSharedOut_, wsRoutedGU_, wsRoutedAct_;
    AscendC::GlobalTensor<float> wsInvRms_, wsMixes_, wsAttnPre_, wsAttnPost_, wsAttnComb_;
    AscendC::GlobalTensor<float> wsFfnPre_, wsFfnPost_, wsFfnComb_, wsRouterLogits_, wsTopkW_;
    AscendC::GlobalTensor<int32_t> wsSync_, wsTopkIds_;
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
