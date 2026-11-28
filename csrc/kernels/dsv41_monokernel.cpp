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
constexpr uint32_t kPostBlk = 64;
constexpr uint32_t kRowBlk = 16;
constexpr uint32_t kMonoSyncInts = 1024;
constexpr uint32_t kHcMult = 4;
constexpr uint32_t kHcMixDim = 24;
constexpr uint32_t kHcCombDim = 16;
constexpr uint32_t kMonoAlign = 64;
constexpr uint32_t kUbSkewWf = 16;
constexpr uint32_t kUbSkewProd = 32;

// Cube L1/L0 tiling caps (Atlas A2: L1=512KB, L0A=64KB, L0B=64KB, L0C=128KB):
// L1A: 2 * 48 * 512 * 2B = 96 KB, L1B: 2 * 128 * 512 * 2B = 256 KB (Total L1 = 352 KB)
// L0A: 2 * 48 * 128 * 2B = 24 KB, L0B: 2 * 128 * 64 * 2B = 32 KB, L0C: 2 * 48 * 128 * 4B = 48 KB
constexpr uint32_t kCubeMaxM = 48;
constexpr uint32_t kCubeMaxN = 128;
constexpr uint32_t kCubeL1K = 512;
constexpr uint32_t kCubeL1AStride = kCubeMaxM * kCubeL1K;
constexpr uint32_t kCubeL1BStride = kCubeMaxN * kCubeL1K;
constexpr uint32_t kCubeL0AStride = kCubeMaxM * 128U;
constexpr uint32_t kCubeL0BStride = kCubeMaxN * 64U;
constexpr uint32_t kCubeL0CStride = kCubeMaxM * kCubeMaxN;
constexpr uint32_t kCubeMmadWaitThresh = 10U;

__aicore__ inline uint64_t MonoAlign64(uint64_t val)
{
    return ((val + kMonoAlign - 1) / kMonoAlign) * kMonoAlign;
}

__aicore__ inline uint32_t MonoVecAlign8(uint32_t val)
{
    return ((val + 7U) / 8U) * 8U;
}

__aicore__ inline uint32_t MonoAlign16(uint32_t val)
{
    return ((val + 15U) / 16U) * 16U;
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
        if ASCEND_IS_AIC {
            blkIdx_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
            blkNum_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        }
        if ASCEND_IS_AIV {
            blkIdx_ = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
            blkNum_ = static_cast<uint32_t>(get_block_num() * get_subblockdim());
        }

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
        uint32_t M = meta_.mTokens;
        uint32_t D = meta_.hidden;
        uint32_t E = meta_.nExperts;
        uint32_t K = meta_.topK;
        uint32_t hasShared = (meta_.sharedInter > 0) ? 1U : 0U;

        // Stage 1a: AIV collapses pre-residual and computes input RMSNorm + 4D inv_rms.
        if ASCEND_IS_AIV {
            CollapseNormAndInvRms(resInGm_, preMixInGm_, kHcMult, inputLnWGm_);
        }
        BarrierAllMix();

        // Stage 1b (OVERLAP #1): AIC runs wq_a + wkv Cube GEMMs while AIV runs HcAttnMix.
        if ASCEND_IS_AIC {
            RunAicQkvProj(M, D);
        }
        if ASCEND_IS_AIV {
            RunAivHcMix(resInGm_, hcAttnFnGm_, M);
        }
        BarrierAllMix();

        // Stage 2a: AIV runs q_norm, kv_norm + KV RoPE + SWA cache scatter + Attn Sinkhorn.
        if ASCEND_IS_AIV {
            RunAivQkvNormRopeScatterSinkhorn(M);
        }
        BarrierAllMix();

        // Stage 2b: AIC runs wq_b Cube GEMM -> wsQ_[M, nHeads * headDim].
        if ASCEND_IS_AIC {
            RunAicQbProj(M);
        }
        BarrierAllMix();

        // Stage 3a: AIV runs Flash SWA with fused partial Q RoPE in UB -> wsAttnOut_.
        if ASCEND_IS_AIV {
            Stage3FlashSwa(M);
        }
        BarrierAllMix();

        // Stage 3b: AIC runs grouped wo_a -> AIC barrier -> wo_b -> wsXAttn_.
        if ASCEND_IS_AIC {
            RunAicOProj(M, D);
        }
        BarrierAllMix();

        // Stage 3c + 4a: AIV runs Mid hc_post -> AIV barrier -> PostAttn CollapseNorm.
        if ASCEND_IS_AIV {
            VectorizedMidHcPost(M, D);
            BarrierAivOnly();
            CollapseNormAndInvRms(wsResMid_, wsAttnPre_, 8, postAttnLnWGm_);
        }
        BarrierAllMix();

        // Stage 4b (OVERLAP #2): AIC runs gate (FP32) + shared_w13 (BF16) while AIV runs HcFfnMix.
        if ASCEND_IS_AIC {
            RunAicGateAndSharedUp(M, D, E, hasShared);
        }
        if ASCEND_IS_AIV {
            RunAivHcMix(wsResMid_, hcFfnFnGm_, M);
        }
        BarrierAllMix();

        // Stage 4c: AIV runs Ffn Sinkhorn + TopK + Shared SwiGLU + Packed Expert CSR + Gather wsExpIn_.
        if ASCEND_IS_AIV {
            for (uint32_t m = blkIdx_; m < M; m += blkNum_) {
                ComputeSinkhornOneRow(m, hcFfnScaleGm_, hcFfnBaseGm_, wsFfnPre_, wsFfnPost_, wsFfnComb_);
                SelectTopKPerToken(m);
            }
            if (hasShared) {
                VectorizedClippedSwiglu(wsSharedGU_, wsSharedAct_, M, meta_.sharedInter);
            }
            BarrierAivOnly();
            if (blkIdx_ == 0) {
                BuildPackedExpertCsr(M, K);
            }
            BarrierAivOnly();
            GatherPackedExpertInputs(M, D, K);
        }
        BarrierAllMix();

        // Stage 5a: AIC runs shared_w2 + expert-grouped routed_w13 Cube GEMMs.
        if ASCEND_IS_AIC {
            RunAicSharedDownAndRoutedUp(M, D, K, hasShared);
        }
        BarrierAllMix();

        // Stage 5b: AIV runs vectorized ClippedSwiGLU across all packed MK routed tokens.
        if ASCEND_IS_AIV {
            VectorizedClippedSwiglu(wsRoutedGU_, wsRoutedAct_, M * K, meta_.routedInter);
        }
        BarrierAllMix();

        // Stage 5c: AIC runs expert-grouped routed_w2 Cube GEMMs -> wsRoutedOut_.
        if ASCEND_IS_AIC {
            RunAicRoutedDown(M, D, K);
        }
        BarrierAllMix();

        // Stage 5d: AIV runs Final hc_post reduction -> resOutGm_ + copies preMixOutGm_.
        if ASCEND_IS_AIV {
            VectorizedFinalHcPost(M, D, K, hasShared);
            for (uint32_t m = blkIdx_; m < M; m += blkNum_) {
                AscendC::LocalTensor<float> prUb =
                    CopyInTile(inQueB_, wsFfnPre_, static_cast<uint64_t>(m) * 8, kHcMult);
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
    }

private:
    __aicore__ inline void BarrierAllMix()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BarrierAivOnly()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<true>();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BarrierAicOnly()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(9);
        AscendC::CrossCoreWaitFlag(9);
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
        uint64_t MK = M * meta_.topK;
        uint64_t expStride = MonoAlign64(MK > 0 ? MK : 1);
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
        wsExpIn_.SetGlobalBuffer(AllocWsB(base, off, MK * D));
        wsRoutedGU_.SetGlobalBuffer(AllocWsB(base, off, MK * 2 * meta_.routedInter));
        wsRoutedAct_.SetGlobalBuffer(AllocWsB(base, off, MK * meta_.routedInter));
        wsRoutedOut_.SetGlobalBuffer(AllocWsB(base, off, MK * D));
        wsExpMeta_.SetGlobalBuffer(AllocWsI(base, off, 64 + 6 * expStride));
    }

    __aicore__ inline void SetupPipeBuffers()
    {
        if ASCEND_IS_AIV {
            uint32_t tileBytes = kMonoTileK * sizeof(float);
            pipe_->InitBuffer(inQueA_, 2, tileBytes);
            pipe_->InitBuffer(inQueB_, 2, tileBytes);
            pipe_->InitBuffer(outQue_, 2, tileBytes);

            pipe_->InitBuffer(bufXBatch_, 8 * tileBytes);
            pipe_->InitBuffer(bufWF32_, tileBytes);
            pipe_->InitBuffer(bufProd_, (kMonoTileK + kUbSkewProd) * sizeof(float));
            pipe_->InitBuffer(bufAux0_, tileBytes);
            pipe_->InitBuffer(bufAux1_, tileBytes);
            pipe_->InitBuffer(bufWork_, tileBytes);
            pipe_->InitBuffer(bufSmall_, 256 * sizeof(float));
        }
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(bufL1A_, 2 * kCubeL1AStride * sizeof(bfloat16_t));
            pipe_->InitBuffer(bufL1B_, 2 * kCubeL1BStride * sizeof(bfloat16_t));
            pipe_->InitBuffer(bufL0A_, 2 * kCubeL0AStride * sizeof(bfloat16_t));
            pipe_->InitBuffer(bufL0B_, 2 * kCubeL0BStride * sizeof(bfloat16_t));
            pipe_->InitBuffer(bufL0C_, 2 * kCubeL0CStride * sizeof(float));
        }
    }

    // ==================== AIC Cube GEMM Engine ====================

    __aicore__ inline void CubePoolBegin()
    {
        l1Stage_ = 0;
        l0Stage_ = 0;
        l0cStage_ = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    }

    __aicore__ inline void CubePoolEnd()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    }

    __aicore__ inline void IssueCubeGmToL1(
        const AscendC::GlobalTensor<bfloat16_t> &xGm, uint64_t xBase, uint32_t xStride, uint32_t mAct, uint32_t mPad,
        const AscendC::GlobalTensor<bfloat16_t> &wGm, uint64_t wBase, uint32_t wKStride, uint32_t nAct, uint32_t nPad,
        uint32_t kOff, uint32_t kL1Act, uint32_t stage)
    {
        AscendC::LocalTensor<bfloat16_t> l1aAll = bufL1A_.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> l1bAll = bufL1B_.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> l1a = l1aAll[stage * kCubeL1AStride];
        AscendC::LocalTensor<bfloat16_t> l1b = l1bAll[stage * kCubeL1BStride];

        event_t ev = (stage == 0) ? EVENT_ID0 : EVENT_ID1;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ev);

        AscendC::Nd2NzParams ndA;
        ndA.ndNum = 1;
        ndA.nValue = static_cast<uint16_t>(mAct);
        ndA.dValue = static_cast<uint16_t>(kL1Act);
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint16_t>(xStride);
        ndA.dstNzC0Stride = static_cast<uint16_t>(mPad);
        ndA.dstNzNStride = 1;
        ndA.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1a, xGm[xBase + kOff], ndA);

        AscendC::Nd2NzParams ndB;
        ndB.ndNum = 1;
        ndB.nValue = static_cast<uint16_t>(nAct);
        ndB.dValue = static_cast<uint16_t>(kL1Act);
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint16_t>(wKStride);
        ndB.dstNzC0Stride = static_cast<uint16_t>(nPad);
        ndB.dstNzNStride = 1;
        ndB.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1b, wGm[wBase + kOff], ndB);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ev);
    }

    template <typename DstT>
    __aicore__ inline void CubeGemmTile(
        const AscendC::GlobalTensor<bfloat16_t> &xGm, uint64_t xBase, uint32_t xStride, uint32_t mAct,
        const AscendC::GlobalTensor<bfloat16_t> &wGm, uint64_t wBase, uint32_t K, uint32_t nAct,
        const AscendC::GlobalTensor<DstT> &outGm, uint64_t outBase, uint32_t outStride,
        uint32_t l0KStep)
    {
        uint32_t mPad = MonoAlign16(mAct);
        uint32_t nPad = MonoAlign16(nAct);
        uint32_t M1 = mPad / 16U;
        uint32_t N1 = nPad / 16U;
        uint32_t kTiles = (K + kCubeL1K - 1) / kCubeL1K;

        AscendC::LocalTensor<bfloat16_t> l1aAll = bufL1A_.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> l1bAll = bufL1B_.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> l0aAll = bufL0A_.Get<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> l0bAll = bufL0B_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> l0cAll = bufL0C_.Get<float>();

        uint32_t curL0cIdx = l0cStage_;
        l0cStage_ ^= 1U;
        event_t l0cEv = (curL0cIdx == 0) ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<float> l0c = l0cAll[curL0cIdx * kCubeL0CStride];

        uint32_t firstKLen = (kCubeL1K <= K) ? kCubeL1K : K;
        IssueCubeGmToL1(xGm, xBase, xStride, mAct, mPad, wGm, wBase, K, nAct, nPad, 0, firstKLen, l1Stage_);

        for (uint32_t kIdx = 0; kIdx < kTiles; ++kIdx) {
            uint32_t curL1Idx = l1Stage_;
            l1Stage_ ^= 1U;
            event_t curL1Ev = (curL1Idx == 0) ? EVENT_ID0 : EVENT_ID1;

            uint32_t kOff = kIdx * kCubeL1K;
            uint32_t kL1Act = (kOff + kCubeL1K <= K) ? kCubeL1K : (K - kOff);

            if (kIdx + 1 < kTiles) {
                uint32_t nextOff = (kIdx + 1) * kCubeL1K;
                uint32_t nextLen = (nextOff + kCubeL1K <= K) ? kCubeL1K : (K - nextOff);
                IssueCubeGmToL1(xGm, xBase, xStride, mAct, mPad, wGm, wBase, K, nAct, nPad, nextOff, nextLen,
                                l1Stage_);
            }

            AscendC::LocalTensor<bfloat16_t> l1a = l1aAll[curL1Idx * kCubeL1AStride];
            AscendC::LocalTensor<bfloat16_t> l1b = l1bAll[curL1Idx * kCubeL1BStride];
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(curL1Ev);

            for (uint32_t kSub = 0; kSub < kL1Act; kSub += l0KStep) {
                uint32_t kSubAct = (kSub + l0KStep <= kL1Act) ? l0KStep : (kL1Act - kSub);
                uint32_t k1Start = kSub / 16U;
                uint32_t K0_1 = (kSubAct + 15U) / 16U;

                uint32_t curL0Idx = l0Stage_;
                l0Stage_ ^= 1U;
                event_t l0Ev = (curL0Idx == 0) ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<bfloat16_t> l0a = l0aAll[curL0Idx * kCubeL0AStride];
                AscendC::LocalTensor<bfloat16_t> l0b = l0bAll[curL0Idx * kCubeL0BStride];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Ev);

                AscendC::LoadData2DParams pA;
                pA.startIndex = 0;
                pA.repeatTimes = static_cast<uint16_t>(K0_1);
                pA.srcStride = static_cast<uint16_t>(M1);
                pA.sid = 0;
                pA.dstGap = 0;
                pA.ifTranspose = false;
                pA.addrMode = 0;
                for (uint32_t m1 = 0; m1 < M1; ++m1) {
                    AscendC::LoadData(l0a[m1 * K0_1 * 256U], l1a[k1Start * mPad * 16U + m1 * 256U], pA);
                }

                AscendC::LoadData2DParams pB;
                pB.startIndex = 0;
                pB.repeatTimes = static_cast<uint16_t>(K0_1 * N1);
                pB.srcStride = 1;
                pB.sid = 0;
                pB.dstGap = 0;
                pB.ifTranspose = false;
                pB.addrMode = 0;
                AscendC::LoadData(l0b, l1b[k1Start * nPad * 16U], pB);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Ev);
                if (kIdx == 0 && kSub == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEv);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Ev);

                AscendC::MmadParams mm;
                mm.m = static_cast<uint16_t>(mPad);
                mm.n = static_cast<uint16_t>(nPad);
                mm.k = static_cast<uint16_t>(kSubAct);
                mm.isBias = false;
                mm.cmatrixInitVal = (kIdx == 0 && kSub == 0);
                mm.cmatrixSource = false;
                AscendC::Mmad(l0c, l0a, l0b, mm);
                if (M1 * N1 < kCubeMmadWaitThresh) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Ev);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(curL1Ev);
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cEv);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEv);
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = static_cast<uint16_t>(nAct);
        fp.mSize = static_cast<uint16_t>(mAct);
        fp.srcStride = static_cast<uint16_t>(mPad);
        fp.dstStride = static_cast<uint32_t>(outStride);
        fp.ndNum = 1;
        fp.reluEn = false;
        fp.unitFlag = 0;
        if constexpr (std::is_same_v<DstT, bfloat16_t>) {
            fp.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(outGm[outBase], l0c, fp);
        } else {
            fp.quantPre = QuantMode_t::NoQuant;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(outGm[outBase], l0c, fp);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEv);
    }

    __aicore__ inline void RunAicQkvProj(uint32_t M, uint32_t D)
    {
        uint32_t tileN = 64U;
        uint32_t mTiles = (M + kCubeMaxM - 1) / kCubeMaxM;
        uint32_t qaTiles = (meta_.qLora + tileN - 1) / tileN;
        uint32_t kvTiles = (meta_.headDim + tileN - 1) / tileN;
        uint32_t nSum = qaTiles + kvTiles;
        uint32_t total = mTiles * nSum;

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t mBase = (tid / nSum) * kCubeMaxM;
            uint32_t mAct = (mBase + kCubeMaxM <= M) ? kCubeMaxM : (M - mBase);
            uint32_t t = tid % nSum;
            if (t < qaTiles) {
                uint32_t nOff = t * tileN;
                uint32_t nAct = (nOff + tileN <= meta_.qLora) ? tileN : (meta_.qLora - nOff);
                CubeGemmTile<bfloat16_t>(
                    wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mAct,
                    wqAGm_, static_cast<uint64_t>(nOff) * D, D, nAct,
                    wsQA_, static_cast<uint64_t>(mBase) * meta_.qLora + nOff, meta_.qLora, 128U);
            } else {
                uint32_t nOff = (t - qaTiles) * tileN;
                uint32_t nAct = (nOff + tileN <= meta_.headDim) ? tileN : (meta_.headDim - nOff);
                CubeGemmTile<bfloat16_t>(
                    wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mAct,
                    wkvGm_, static_cast<uint64_t>(nOff) * D, D, nAct,
                    wsKV_, static_cast<uint64_t>(mBase) * meta_.headDim + nOff, meta_.headDim, 128U);
            }
        }
        CubePoolEnd();
    }

    __aicore__ inline void RunAicQbProj(uint32_t M)
    {
        uint32_t qN = meta_.nHeads * meta_.headDim;
        uint32_t tileN = (qN >= 8192U) ? 128U : 64U;
        uint32_t l0K = (tileN == 128U) ? 64U : 128U;
        uint32_t mTiles = (M + kCubeMaxM - 1) / kCubeMaxM;
        uint32_t nTiles = (qN + tileN - 1) / tileN;
        uint32_t total = mTiles * nTiles;

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t mBase = (tid / nTiles) * kCubeMaxM;
            uint32_t mAct = (mBase + kCubeMaxM <= M) ? kCubeMaxM : (M - mBase);
            uint32_t nOff = (tid % nTiles) * tileN;
            uint32_t nAct = (nOff + tileN <= qN) ? tileN : (qN - nOff);
            CubeGemmTile<bfloat16_t>(
                wsQR_, static_cast<uint64_t>(mBase) * meta_.qLora, meta_.qLora, mAct,
                wqBGm_, static_cast<uint64_t>(nOff) * meta_.qLora, meta_.qLora, nAct,
                wsQ_, static_cast<uint64_t>(mBase) * qN + nOff, qN, l0K);
        }
        CubePoolEnd();
    }

    __aicore__ inline void RunAicOProj(uint32_t M, uint32_t D)
    {
        uint32_t mTiles = (M + kCubeMaxM - 1) / kCubeMaxM;
        uint32_t groupHidden = (meta_.nHeads * meta_.headDim) / meta_.oGroups;
        uint32_t xOAStride = meta_.nHeads * meta_.headDim;
        uint32_t outOAStride = meta_.oGroups * meta_.oLora;

        uint32_t tileNA = 64U;
        uint32_t gTiles = (meta_.oLora + tileNA - 1) / tileNA;
        uint32_t totalA = mTiles * meta_.oGroups * gTiles;

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < totalA; tid += blkNum_) {
            uint32_t mtg = tid / gTiles;
            uint32_t mBase = (mtg / meta_.oGroups) * kCubeMaxM;
            uint32_t mAct = (mBase + kCubeMaxM <= M) ? kCubeMaxM : (M - mBase);
            uint32_t g = mtg % meta_.oGroups;
            uint32_t nOff = (tid % gTiles) * tileNA;
            uint32_t nAct = (nOff + tileNA <= meta_.oLora) ? tileNA : (meta_.oLora - nOff);

            uint64_t xOff = static_cast<uint64_t>(mBase) * xOAStride + static_cast<uint64_t>(g) * groupHidden;
            uint64_t wOff = (static_cast<uint64_t>(g) * meta_.oLora + nOff) * groupHidden;
            uint64_t oOff = static_cast<uint64_t>(mBase) * outOAStride + static_cast<uint64_t>(g) * meta_.oLora + nOff;
            CubeGemmTile<bfloat16_t>(
                wsAttnOut_, xOff, xOAStride, mAct,
                woAGm_, wOff, groupHidden, nAct,
                wsOA_, oOff, outOAStride, 128U);
        }
        CubePoolEnd();
        BarrierAicOnly();

        uint32_t tileNB = (D >= 2560U) ? 128U : 64U;
        uint32_t l0KB = (tileNB == 128U) ? 64U : 128U;
        uint32_t bTiles = (D + tileNB - 1) / tileNB;
        uint32_t totalB = mTiles * bTiles;

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < totalB; tid += blkNum_) {
            uint32_t mBase = (tid / bTiles) * kCubeMaxM;
            uint32_t mAct = (mBase + kCubeMaxM <= M) ? kCubeMaxM : (M - mBase);
            uint32_t nOff = (tid % bTiles) * tileNB;
            uint32_t nAct = (nOff + tileNB <= D) ? tileNB : (D - nOff);
            CubeGemmTile<bfloat16_t>(
                wsOA_, static_cast<uint64_t>(mBase) * outOAStride, outOAStride, mAct,
                woBGm_, static_cast<uint64_t>(nOff) * outOAStride, outOAStride, nAct,
                wsXAttn_, static_cast<uint64_t>(mBase) * D + nOff, D, l0KB);
        }
        CubePoolEnd();
    }

    __aicore__ inline void RunAicGateAndSharedUp(uint32_t M, uint32_t D, uint32_t E, uint32_t hasShared)
    {
        uint32_t tileN = 64U;
        uint32_t mTiles = (M + kCubeMaxM - 1) / kCubeMaxM;
        uint32_t gateTiles = (E + tileN - 1) / tileN;
        uint32_t shN = 2U * meta_.sharedInter;
        uint32_t shTiles = hasShared ? ((shN + tileN - 1) / tileN) : 0U;
        uint32_t nSum = gateTiles + shTiles;
        uint32_t total = mTiles * nSum;
        uint64_t lStride = MonoAlign64(E);

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t mBase = (tid / nSum) * kCubeMaxM;
            uint32_t mAct = (mBase + kCubeMaxM <= M) ? kCubeMaxM : (M - mBase);
            uint32_t t = tid % nSum;
            if (t < gateTiles) {
                uint32_t nOff = t * tileN;
                uint32_t nAct = (nOff + tileN <= E) ? tileN : (E - nOff);
                CubeGemmTile<float>(
                    wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mAct,
                    gateWGm_, static_cast<uint64_t>(nOff) * D, D, nAct,
                    wsRouterLogits_, static_cast<uint64_t>(mBase) * lStride + nOff, static_cast<uint32_t>(lStride),
                    128U);
            } else {
                uint32_t nOff = (t - gateTiles) * tileN;
                uint32_t nAct = (nOff + tileN <= shN) ? tileN : (shN - nOff);
                CubeGemmTile<bfloat16_t>(
                    wsXNorm_, static_cast<uint64_t>(mBase) * D, D, mAct,
                    sharedW13Gm_, static_cast<uint64_t>(nOff) * D, D, nAct,
                    wsSharedGU_, static_cast<uint64_t>(mBase) * shN + nOff, shN, 128U);
            }
        }
        CubePoolEnd();
    }

    __aicore__ inline void RunAicSharedDownAndRoutedUp(uint32_t M, uint32_t D, uint32_t K, uint32_t hasShared)
    {
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(wsExpMeta_);
        uint32_t numGroups = static_cast<uint32_t>(wsExpMeta_.GetValue(0));
        uint64_t MK = static_cast<uint64_t>(M) * K;
        uint64_t expStride = MonoAlign64(MK > 0 ? MK : 1);

        uint32_t tileNS = 64U;
        uint32_t mTiles = (M + kCubeMaxM - 1) / kCubeMaxM;
        uint32_t shDTiles = hasShared ? ((D + tileNS - 1) / tileNS) : 0U;
        uint32_t shTasks = mTiles * shDTiles;

        uint32_t tileNR = 128U;
        uint32_t rN = 2U * meta_.routedInter;
        uint32_t rTiles = (rN + tileNR - 1) / tileNR;
        uint32_t rTasks = numGroups * rTiles;
        uint32_t total = shTasks + rTasks;

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            if (tid < shTasks) {
                uint32_t mBase = (tid / shDTiles) * kCubeMaxM;
                uint32_t mAct = (mBase + kCubeMaxM <= M) ? kCubeMaxM : (M - mBase);
                uint32_t nOff = (tid % shDTiles) * tileNS;
                uint32_t nAct = (nOff + tileNS <= D) ? tileNS : (D - nOff);
                CubeGemmTile<bfloat16_t>(
                    wsSharedAct_, static_cast<uint64_t>(mBase) * meta_.sharedInter, meta_.sharedInter, mAct,
                    sharedW2Gm_, static_cast<uint64_t>(nOff) * meta_.sharedInter, meta_.sharedInter, nAct,
                    wsSharedOut_, static_cast<uint64_t>(mBase) * D + nOff, D, 128U);
            } else {
                uint32_t rr = tid - shTasks;
                uint32_t g = rr / rTiles;
                uint32_t nOff = (rr % rTiles) * tileNR;
                uint32_t nAct = (nOff + tileNR <= rN) ? tileNR : (rN - nOff);
                uint32_t eid = static_cast<uint32_t>(wsExpMeta_.GetValue(64ULL + g));
                uint32_t rowStart = static_cast<uint32_t>(wsExpMeta_.GetValue(64ULL + expStride + g));
                uint32_t cnt = static_cast<uint32_t>(wsExpMeta_.GetValue(64ULL + 2ULL * expStride + g));

                CubeGemmTile<bfloat16_t>(
                    wsExpIn_, static_cast<uint64_t>(rowStart) * D, D, cnt,
                    routedW13Gm_, (static_cast<uint64_t>(eid) * rN + nOff) * D, D, nAct,
                    wsRoutedGU_, static_cast<uint64_t>(rowStart) * rN + nOff, rN, 64U);
            }
        }
        CubePoolEnd();
    }

    __aicore__ inline void RunAicRoutedDown(uint32_t M, uint32_t D, uint32_t K)
    {
        uint32_t numGroups = static_cast<uint32_t>(wsExpMeta_.GetValue(0));
        uint64_t MK = static_cast<uint64_t>(M) * K;
        uint64_t expStride = MonoAlign64(MK > 0 ? MK : 1);

        uint32_t tileN = 128U;
        uint32_t dTiles = (D + tileN - 1) / tileN;
        uint32_t total = numGroups * dTiles;

        CubePoolBegin();
        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t g = tid / dTiles;
            uint32_t nOff = (tid % dTiles) * tileN;
            uint32_t nAct = (nOff + tileN <= D) ? tileN : (D - nOff);
            uint32_t eid = static_cast<uint32_t>(wsExpMeta_.GetValue(64ULL + g));
            uint32_t rowStart = static_cast<uint32_t>(wsExpMeta_.GetValue(64ULL + expStride + g));
            uint32_t cnt = static_cast<uint32_t>(wsExpMeta_.GetValue(64ULL + 2ULL * expStride + g));

            CubeGemmTile<bfloat16_t>(
                wsRoutedAct_, static_cast<uint64_t>(rowStart) * meta_.routedInter, meta_.routedInter, cnt,
                routedW2Gm_, (static_cast<uint64_t>(eid) * D + nOff) * meta_.routedInter, meta_.routedInter, nAct,
                wsRoutedOut_, static_cast<uint64_t>(rowStart) * D + nOff, D, 64U);
        }
        CubePoolEnd();
    }

    // ==================== AIV Vector & Attention Engine ====================

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

    __aicore__ inline void RunAivHcMix(
        const AscendC::GlobalTensor<bfloat16_t> &resGm,
        const AscendC::GlobalTensor<float> &hcFnGm,
        uint32_t M)
    {
        uint32_t mixTasks = M * 3U;
        for (uint32_t tid = blkIdx_; tid < mixTasks; tid += blkNum_) {
            ComputeHcMixChunk(resGm, hcFnGm, tid / 3U, (tid % 3U) * 8U);
        }
    }

    __aicore__ inline void RunAivQkvNormRopeScatterSinkhorn(uint32_t M)
    {
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

    __aicore__ inline void Stage3FlashSwa(uint32_t M)
    {
        uint32_t H = meta_.nHeads;
        uint32_t hd = meta_.headDim;
        uint32_t nope = meta_.nopeDim;
        uint32_t rDim = meta_.ropeDim;
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
            AscendC::LocalTensor<float> cs = CopyInTile(inQueB_, cosGm_, static_cast<uint64_t>(m) * rDim, rDim);
            AscendC::LocalTensor<float> sn = CopyInTile(inQueB_, sinGm_, static_cast<uint64_t>(m) * rDim, rDim);
            AscendC::Cast(qf, qb, AscendC::RoundMode::CAST_NONE, hd);
            AscendC::PipeBarrier<PIPE_ALL>();
            inQueA_.FreeTensor(qb);

            // Fused in-UB partial interleaved Q RoPE with BF16 round-trip parity
            for (uint32_t i = 0; i < rDim; i += 2) {
                float x0 = qf.GetValue(nope + i);
                float x1 = qf.GetValue(nope + i + 1);
                float c0 = cs.GetValue(i);
                float c1 = cs.GetValue(i + 1);
                float s0 = sn.GetValue(i);
                float s1 = sn.GetValue(i + 1);
                dotRes.SetValue(i, x0 * c0 - x1 * s0);
                dotRes.SetValue(i + 1, x1 * c1 + x0 * s1);
            }
            inQueB_.FreeTensor(cs);
            inQueB_.FreeTensor(sn);
            AscendC::PipeBarrier<PIPE_ALL>();
            {
                AscendC::LocalTensor<bfloat16_t> rb = outQue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(rb, dotRes, AscendC::RoundMode::CAST_RINT, rDim);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(qf[nope], rb, AscendC::RoundMode::CAST_NONE, rDim);
                AscendC::PipeBarrier<PIPE_ALL>();
                outQue_.FreeTensor(rb);
            }
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
    }

    __aicore__ inline void VectorizedMidHcPost(uint32_t M, uint32_t D)
    {
        uint32_t dBlks = (D + kPostBlk - 1) / kPostBlk;
        uint32_t total = M * dBlks;
        AscendC::LocalTensor<float> rAll = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> xF = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> rf = bufAux1_.Get<float>();
        AscendC::LocalTensor<float> tmp = bufWork_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t m = tid / dBlks;
            uint32_t d0 = (tid % dBlks) * kPostBlk;
            uint32_t rows = (d0 + kPostBlk <= D) ? kPostBlk : (D - d0);
            uint32_t vRows = MonoVecAlign8(rows);

            AscendC::LocalTensor<float> poUb = CopyInTile(inQueB_, wsAttnPost_, static_cast<uint64_t>(m) * 8, kHcMult);
            float post4[4];
            for (uint32_t i = 0; i < 4; ++i) {
                post4[i] = poUb.GetValue(i);
            }
            inQueB_.FreeTensor(poUb);

            AscendC::LocalTensor<float> cmUb =
                CopyInTile(inQueB_, wsAttnComb_, static_cast<uint64_t>(m) * kHcCombDim, kHcCombDim);
            float cm44[4][4];
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t c = 0; c < 4; ++c) {
                    cm44[r][c] = cmUb.GetValue(r * 4 + c);
                }
            }
            inQueB_.FreeTensor(cmUb);

            AscendC::LocalTensor<bfloat16_t> xb =
                CopyInTile(inQueA_, wsXAttn_, static_cast<uint64_t>(m) * D + d0, rows);
            AscendC::Cast(xF, xb, AscendC::RoundMode::CAST_NONE, rows);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(xb);

            uint64_t rBase = static_cast<uint64_t>(m) * kHcMult * D + d0;
            for (uint32_t i = 0; i < 4; ++i) {
                AscendC::LocalTensor<bfloat16_t> rb =
                    CopyInTile(inQueA_, resInGm_, rBase + static_cast<uint64_t>(i) * D, rows);
                AscendC::Cast(rAll[i * kPostBlk], rb, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(rb);
            }

            for (uint32_t j = 0; j < 4; ++j) {
                AscendC::Muls(rf, xF, post4[j], vRows);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < 4; ++i) {
                    AscendC::Muls(tmp, rAll[i * kPostBlk], cm44[i][j], vRows);
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

    __aicore__ inline void VectorizedClippedSwiglu(
        const AscendC::GlobalTensor<bfloat16_t> &guGm,
        const AscendC::GlobalTensor<bfloat16_t> &actGm,
        uint32_t nRows,
        uint32_t inter)
    {
        float lim = meta_.swigluLimit;
        uint32_t cTiles = (inter + kMonoTileK - 1) / kMonoTileK;
        uint32_t total = nRows * cTiles;
        AscendC::LocalTensor<float> gf = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> sig = bufAux1_.Get<float>();
        AscendC::LocalTensor<float> uf = bufProd_.Get<float>()[kUbSkewProd];

        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t row = tid / cTiles;
            uint32_t c0 = (tid % cTiles) * kMonoTileK;
            uint32_t len = (c0 + kMonoTileK <= inter) ? kMonoTileK : (inter - c0);
            uint32_t vLen = MonoVecAlign8(len);
            uint64_t guBase = static_cast<uint64_t>(row) * (2U * inter);

            AscendC::LocalTensor<bfloat16_t> gb = CopyInTile(inQueA_, guGm, guBase + c0, len);
            AscendC::LocalTensor<bfloat16_t> ub = CopyInTile(inQueB_, guGm, guBase + inter + c0, len);
            AscendC::Cast(gf, gb, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(uf, ub, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
            inQueA_.FreeTensor(gb);
            inQueB_.FreeTensor(ub);

            if (lim > 0.0f) {
                AscendC::Mins(gf, gf, lim, vLen);
                AscendC::Mins(uf, uf, lim, vLen);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(uf, uf, -lim, vLen);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Muls(sig, gf, -1.0f, vLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(sig, sig, vLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sig, sig, 1.0f, vLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(gf, gf, sig, vLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(gf, gf, uf, vLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(ob, gf, AscendC::RoundMode::CAST_RINT, len);
            AscendC::PipeBarrier<PIPE_V>();
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(actGm, static_cast<uint64_t>(row) * inter + c0, ob, len);
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

    __aicore__ inline void WriteInt32SliceToMeta(uint64_t dstOff, const AscendC::LocalTensor<int32_t> &src, uint32_t n)
    {
        AscendC::LocalTensor<int32_t> ob = outQue_.AllocTensor<int32_t>();
        for (uint32_t i = 0; i < n; ++i) {
            ob.SetValue(i, src.GetValue(i));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(ob);
        ob = outQue_.DeQue<int32_t>();
        CopyOutTile(wsExpMeta_, dstOff, ob, n);
    }

    __aicore__ inline void BuildPackedExpertCsr(uint32_t M, uint32_t K)
    {
        uint32_t MK = M * K;
        uint64_t tkStride = MonoAlign64(K);
        uint64_t expStride = MonoAlign64(MK > 0 ? MK : 1);

        // bufXBatch_ (32 KB = 8192 int32s): holds up to MK <= 1024 entries safely.
        AscendC::LocalTensor<int32_t> tmpUb = bufXBatch_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> slotIds = tmpUb[0];
        AscendC::LocalTensor<int32_t> slotM = tmpUb[1024];
        AscendC::LocalTensor<int32_t> used = tmpUb[2048];
        AscendC::LocalTensor<int32_t> grpEid = tmpUb[3072];
        AscendC::LocalTensor<int32_t> grpStart = tmpUb[4096];
        AscendC::LocalTensor<int32_t> grpCnt = tmpUb[5120];
        AscendC::LocalTensor<int32_t> srcTokM = tmpUb[6144];
        AscendC::LocalTensor<int32_t> slotToPacked = tmpUb[7168];

        for (uint32_t m = 0; m < M; ++m) {
            AscendC::LocalTensor<int32_t> idUb =
                CopyInTile(inQueA_, wsTopkIds_, static_cast<uint64_t>(m) * tkStride, K);
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t idx = m * K + k;
                slotIds.SetValue(idx, idUb.GetValue(k));
                slotM.SetValue(idx, static_cast<int32_t>(m));
                used.SetValue(idx, 0);
            }
            inQueA_.FreeTensor(idUb);
        }

        uint32_t numGroups = 0;
        uint32_t packedPos = 0;
        for (uint32_t i = 0; i < MK; ++i) {
            if (used.GetValue(i) != 0) {
                continue;
            }
            int32_t eid = slotIds.GetValue(i);
            uint32_t rowStart = packedPos;
            uint32_t cnt = 0;
            for (uint32_t j = i; j < MK && cnt < kCubeMaxM; ++j) {
                if (used.GetValue(j) == 0 && slotIds.GetValue(j) == eid) {
                    used.SetValue(j, 1);
                    srcTokM.SetValue(packedPos, slotM.GetValue(j));
                    slotToPacked.SetValue(j, static_cast<int32_t>(packedPos));
                    ++packedPos;
                    ++cnt;
                }
            }
            grpEid.SetValue(numGroups, eid);
            grpStart.SetValue(numGroups, static_cast<int32_t>(rowStart));
            grpCnt.SetValue(numGroups, static_cast<int32_t>(cnt));
            ++numGroups;
        }

        WriteInt32SliceToMeta(64ULL + 0ULL * expStride, grpEid, numGroups);
        WriteInt32SliceToMeta(64ULL + 1ULL * expStride, grpStart, numGroups);
        WriteInt32SliceToMeta(64ULL + 2ULL * expStride, grpCnt, numGroups);
        WriteInt32SliceToMeta(64ULL + 3ULL * expStride, srcTokM, MK);
        WriteInt32SliceToMeta(64ULL + 4ULL * expStride, slotToPacked, MK);

        AscendC::LocalTensor<int32_t> cntOb = outQue_.AllocTensor<int32_t>();
        cntOb.SetValue(0, static_cast<int32_t>(numGroups));
        AscendC::PipeBarrier<PIPE_ALL>();
        outQue_.EnQue(cntOb);
        cntOb = outQue_.DeQue<int32_t>();
        CopyOutTile(wsExpMeta_, 0, cntOb, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void GatherPackedExpertInputs(uint32_t M, uint32_t D, uint32_t K)
    {
        uint32_t MK = M * K;
        if (MK == 0) {
            return;
        }
        uint64_t expStride = MonoAlign64(MK);
        uint32_t dTiles = (D + kMonoTileK - 1) / kMonoTileK;
        uint32_t total = MK * dTiles;

        AscendC::LocalTensor<int32_t> srcTokUb =
            CopyInTile(inQueB_, wsExpMeta_, 64ULL + 3ULL * expStride, MK);
        AscendC::LocalTensor<int32_t> srcTokLocal = bufXBatch_.Get<int32_t>();
        for (uint32_t p = 0; p < MK; ++p) {
            srcTokLocal.SetValue(p, srcTokUb.GetValue(p));
        }
        inQueB_.FreeTensor(srcTokUb);

        for (uint32_t tid = blkIdx_; tid < total; tid += blkNum_) {
            uint32_t p = tid / dTiles;
            uint32_t d0 = (tid % dTiles) * kMonoTileK;
            uint32_t len = (d0 + kMonoTileK <= D) ? kMonoTileK : (D - d0);
            uint32_t tm = static_cast<uint32_t>(srcTokLocal.GetValue(p));

            AscendC::LocalTensor<bfloat16_t> xb =
                CopyInTile(inQueA_, wsXNorm_, static_cast<uint64_t>(tm) * D + d0, len);
            AscendC::LocalTensor<bfloat16_t> ob = outQue_.AllocTensor<bfloat16_t>();
            AscendC::DataCopy(ob, xb, len);
            AscendC::PipeBarrier<PIPE_ALL>();
            inQueA_.FreeTensor(xb);
            outQue_.EnQue(ob);
            ob = outQue_.DeQue<bfloat16_t>();
            CopyOutTile(wsExpIn_, static_cast<uint64_t>(p) * D + d0, ob, len);
        }
    }

    __aicore__ inline void VectorizedFinalHcPost(uint32_t M, uint32_t D, uint32_t K, uint32_t hasShared)
    {
        uint32_t dBlks = (D + kPostBlk - 1) / kPostBlk;
        uint32_t totalFinal = M * dBlks;
        uint64_t tkStride = MonoAlign64(K);
        uint64_t expStride = MonoAlign64(M * K > 0 ? M * K : 1);

        AscendC::LocalTensor<float> rAll = bufXBatch_.Get<float>();
        AscendC::LocalTensor<float> combUb = bufAux0_.Get<float>();
        AscendC::LocalTensor<float> rf = bufAux1_.Get<float>();
        AscendC::LocalTensor<float> tmp = bufWork_.Get<float>();

        for (uint32_t tid = blkIdx_; tid < totalFinal; tid += blkNum_) {
            uint32_t m = tid / dBlks;
            uint32_t d0 = (tid % dBlks) * kPostBlk;
            uint32_t rows = (d0 + kPostBlk <= D) ? kPostBlk : (D - d0);
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

            AscendC::LocalTensor<int32_t> pUb =
                CopyInTile(inQueA_, wsExpMeta_, 64ULL + 4ULL * expStride + static_cast<uint64_t>(m) * K, K);
            AscendC::LocalTensor<float> wUb = CopyInTile(inQueB_, wsTopkW_, static_cast<uint64_t>(m) * tkStride, K);
            uint32_t pRows[16];
            float ws[16];
            for (uint32_t k = 0; k < K; ++k) {
                pRows[k] = static_cast<uint32_t>(pUb.GetValue(k));
                ws[k] = wUb.GetValue(k);
            }
            inQueA_.FreeTensor(pUb);
            inQueB_.FreeTensor(wUb);

            for (uint32_t k = 0; k < K; ++k) {
                uint64_t roOff = static_cast<uint64_t>(pRows[k]) * D + d0;
                AscendC::LocalTensor<bfloat16_t> rob = CopyInTile(inQueA_, wsRoutedOut_, roOff, rows);
                AscendC::Cast(tmp, rob, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(rob);
                AscendC::Muls(tmp, tmp, ws[k], vRows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(combUb, combUb, tmp, vRows);
                AscendC::PipeBarrier<PIPE_V>();
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
                AscendC::Cast(rAll[i * kPostBlk], rb, AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueA_.FreeTensor(rb);
            }

            for (uint32_t j = 0; j < 4; ++j) {
                AscendC::Muls(rf, combUb, post4[j], vRows);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < 4; ++i) {
                    AscendC::Muls(tmp, rAll[i * kPostBlk], cm44[i][j], vRows);
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

private:
    AscendC::TPipe *pipe_;
    uint32_t blkIdx_{0};
    uint32_t blkNum_{1};
    uint32_t l1Stage_{0};
    uint32_t l0Stage_{0};
    uint32_t l0cStage_{0};
    MonoMeta meta_;

    // AIV UB queues & buffers
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueA_, inQueB_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufXBatch_, bufWF32_, bufProd_, bufAux0_, bufAux1_, bufWork_, bufSmall_;

    // AIC L1/L0 buffers
    AscendC::TBuf<AscendC::QuePosition::A1> bufL1A_;
    AscendC::TBuf<AscendC::QuePosition::B1> bufL1B_;
    AscendC::TBuf<AscendC::QuePosition::A2> bufL0A_;
    AscendC::TBuf<AscendC::QuePosition::B2> bufL0B_;
    AscendC::TBuf<AscendC::QuePosition::CO1> bufL0C_;

    AscendC::GlobalTensor<bfloat16_t> resInGm_, resOutGm_, inputLnWGm_, wqAGm_, qNormWGm_, wqBGm_;
    AscendC::GlobalTensor<bfloat16_t> wkvGm_, kvNormWGm_, woAGm_, woBGm_, postAttnLnWGm_;
    AscendC::GlobalTensor<bfloat16_t> gateWGm_, sharedW13Gm_, sharedW2Gm_, routedW13Gm_, routedW2Gm_, swaKvCacheGm_;
    AscendC::GlobalTensor<float> preMixInGm_, preMixOutGm_, cosGm_, sinGm_, hcAttnFnGm_, hcAttnScaleGm_, hcAttnBaseGm_;
    AscendC::GlobalTensor<float> attnSinkGm_, hcFfnFnGm_, hcFfnScaleGm_, hcFfnBaseGm_, eBiasGm_;
    AscendC::GlobalTensor<int64_t> posGm_, slotGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_, seqLensGm_;

    AscendC::GlobalTensor<bfloat16_t> wsXNorm_, wsResMid_, wsQA_, wsQR_, wsKV_, wsQ_, wsAttnOut_, wsOA_, wsXAttn_;
    AscendC::GlobalTensor<bfloat16_t> wsSharedGU_, wsSharedAct_, wsSharedOut_, wsExpIn_;
    AscendC::GlobalTensor<bfloat16_t> wsRoutedGU_, wsRoutedAct_, wsRoutedOut_;
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
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
