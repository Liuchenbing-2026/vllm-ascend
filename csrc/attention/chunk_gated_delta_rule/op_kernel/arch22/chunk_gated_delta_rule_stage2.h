/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file chunk_gated_delta_rule_stage2.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_matmul_basic.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using StageTwoMT = CGDRMatmulBasic;

struct StageTwoParams {
    GlobalTensor<bfloat16_t> qPrime;    // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> vInner;    // (Nv, Sp, Dv)
    GlobalTensor<float> gCum;   // (Nv, Sp)
    GlobalTensor<bfloat16_t> kCumdecay; // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> curState;  // (Nv, Dv, Dk)
    GlobalTensor<bfloat16_t> finalState;  // (Nv, Dv, Dk)
    GlobalTensor<bfloat16_t> altState;  // (Nv, Dv, Dk) 状态乒乓的第二块缓冲（借用 highState_ 空间）
    GlobalTensor<bfloat16_t> kg;
    GlobalTensor<bfloat16_t> out;
    GM_ADDR ws;
    StageTwoMT *mm1;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage2 {
public:
    __aicore__ inline void Init(StageTwoParams *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        curChunkSize_ = chunkSize_;
        gOptional_ = sTP_->gOptional;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, Std::max((uint64_t)chunkSize_,
                                                             (uint64_t)Dv_) * curDk_ * sizeof(bfloat16_t));
        uint64_t outQueueSize = Std::max((uint64_t)chunkSize_ * chunkSize_ * sizeof(bfloat16_t),
                                         (uint64_t)Dv_ * curDk_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outQueueSize);
        pipe_->InitBuffer(tmpBuff_, (Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ +
                                     BLOCK_FLOAT_NUM) *sizeof(float));
        uint32_t buffOffset = 0;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(Dv_ * curDk_), buffOffset);
        buffOffset += Ceil(Dv_ * curDk_ * sizeof(float), BLOCK_SIZE) * BLOCK_SIZE;
        // 暂存所取的最后一位数
        lastGCum_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(NUM_ONE), buffOffset);
    }

    __aicore__ inline void Process()
    {
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= AIC_AIV_1_1;
        }
        // 门控 Dv 分核：2*Nv 个 (head, Dv/2) 单元放得进 coreNum 时 f=2（nv 小、核大量闲置的场景），
        // 否则 f=1 走与原来完全相同的路径（TP1/nv>=coreNum 不受影响）。
        // Dv_ 必须能被 f 整除，否则 Dv_/f 会静默丢掉尾行（host tiling 只保证 0 < dv <= 128，不保证偶数）
        int64_t f = (2 * Nv_ <= (int64_t)coreNum_ && Dv_ % 2 == 0) ? 2 : 1;
        dvRows_ = Dv_ / f;
        int64_t units = Nv_ * f;
        int64_t unitsPerCore = (units + coreNum_ - 1) / coreNum_;
        int64_t uStart = coreId * unitsPerCore;
        int64_t uEnd = uStart + unitsPerCore;
        uEnd = uEnd > units ? units : uEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        // 状态乒乓：读 buf 与写 buf 分离，砍掉「AIC 读完老状态之前 AIV 不能写」那条 0x2 跨核边，
        // 让 AIV 的 32KB 读-缩放-写与 AIC 的 CalVPrime/CalAttnInter 并行。
        // 相位锚定在 group 末尾：最后一个 chunk 必写 finalState（外层跨 group 用 finalState 接力）。
        bool aliasIn = sTP_->curState.GetPhyAddr() == sTP_->finalState.GetPhyAddr();
        for (int64_t u = uStart; u < uEnd; u++) {
            int64_t nvId = u / f;
            int64_t stateOff = nvId * Dv_ * Dk_ + (u % f) * dvRows_ * Dk_;   // 本单元负责的 Dv 行片
            int64_t rowStart = (u % f) * dvRows_;
            curChunkSize_ = chunkSize_;
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                bool writeFinal = ((chunkNum_ - 1 - cId) % 2) == 0;
                auto writeState = writeFinal ? sTP_->finalState[stateOff]
                                             : sTP_->altState[stateOff];
                auto readState = (cId == 0) ? sTP_->curState[stateOff]
                                : (writeFinal ? sTP_->altState[stateOff]
                                              : sTP_->finalState[stateOff]);
                // 读写同块只可能出现在 cId==0 且入口状态就在 finalState 且 chunkNum_ 为奇数，
                // 此时退回旧握手（AIV 等 AIC 读完再回写）。AIC/AIV 用相同算式判定，flag 收发配对。
                bool sameBuf = (cId == 0) && writeFinal && aliasIn;
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0) {
                        CopyIn(readState, dvRows_, Dk_);
                        CalGCumExp(sTP_->gCum[nvId * Sp_ + length]);
                    }
                    if (sameBuf) {
                        CrossCoreWaitFlag(0x2);
                    }
                    if (GetSubBlockIdx() == 0) {
                        CopyOutState(writeState);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
                    CrossCoreWaitFlag(0x4);
                }
                if ASCEND_IS_AIC {
                    uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
                    uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_ + rowStart;
                    CalVPrime(sTP_->kCumdecay[mm_offset0], readState, sTP_->vInner[mm_offset1]);
                    CalAttnInter(sTP_->qPrime[mm_offset0], readState,
                                 sTP_->out[nvId * Dv_ + rowStart + cId * chunkSize_ * Nv_ * Dv_]);
                    if (sameBuf) {
                        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);   // 读完之前AIV不能写
                    }
                    CrossCoreWaitFlag(0x5);
                    CalStateNew(sTP_->vInner[mm_offset1], sTP_->kg[mm_offset0], writeState);
                    SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
                }
            }
        }
    }

    // 打包模式：group 跨序列（前提：所有序列长度为 chunkSize 整数倍，chunk 起点全局连续），
    // 按 (bid, nv[, Dv片]) 段式链分核。此时 params 的 curState/finalState/altState 传的是
    // 【未加 bid 偏移的基址】，bid 偏移在这里加。
    __aicore__ inline void ProcessPacked(GlobalTensor<int32_t> seqLens, int64_t b, int64_t groupChunkStart,
                                         int64_t totalTokens)
    {
        // 全局最后一个 chunk 的有效长度（只有末序列可能不对齐 chunkSize）
        int64_t totalChunks = (totalTokens + chunkSize_ - 1) / chunkSize_;
        int64_t globalLastLen = totalTokens % chunkSize_ == 0 ? chunkSize_ : totalTokens % chunkSize_;
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= AIC_AIV_1_1;
        }
        int64_t groupChunkEnd = groupChunkStart + chunkNum_;
        // 与本 group 相交的 bid 区间 [bidFirst, bidLast]（所有核算出同一份，标量代价 O(b)）
        int64_t bidFirst = -1;
        int64_t bidLast = -1;
        int64_t acc = 0;
        int64_t prefFirst = 0;   // bidFirst 的起始全局 chunk 号
        for (int64_t bid = 0; bid < b; bid++) {
            int64_t nChunks = ((int64_t)seqLens.GetValue(bid) + chunkSize_ - 1) / chunkSize_;
            if (acc + nChunks > groupChunkStart && acc < groupChunkEnd) {
                if (bidFirst < 0) {
                    bidFirst = bid;
                    prefFirst = acc;
                }
                bidLast = bid;
            }
            acc += nChunks;
        }
        int64_t nBids = bidLast - bidFirst + 1;
        // Dv 分核门：单元数不到核数一半才二分 Dv（单元本来就多时细分只会降 matmul 效率）
        // 同上：Dv_ 为奇数时禁用 Dv 分片
        int64_t f = (2 * nBids * Nv_ <= (int64_t)coreNum_ && Dv_ % 2 == 0) ? 2 : 1;
        dvRows_ = Dv_ / f;
        int64_t units = nBids * Nv_ * f;
        int64_t unitsPerCore = (units + coreNum_ - 1) / coreNum_;
        int64_t uStart = coreId * unitsPerCore;
        int64_t uEnd = uStart + unitsPerCore;
        uEnd = uEnd > units ? units : uEnd;
        // 增量维护当前 bid 的起始全局 chunk 号，避免每 unit 重扫前缀和（O(units+nBids) 而非 O(units*nBids)）
        int64_t prevBid = -1;
        int64_t bidChunkStart = 0;
        int64_t bidChunkEnd = 0;
        for (int64_t u = uStart; u < uEnd; u++) {
            int64_t bid = bidFirst + u / (Nv_ * f);
            int64_t nvId = (u / f) % Nv_;
            int64_t rowStart = (u % f) * dvRows_;
            if (bid != prevBid) {
                if (prevBid < 0) {
                    bidChunkStart = prefFirst;
                    for (int64_t x = bidFirst; x < bid; x++) {
                        bidChunkStart += (int64_t)seqLens.GetValue(x) / chunkSize_;
                    }
                } else {
                    bidChunkStart = bidChunkEnd;   // 上一 bid 的末尾即本 bid 的开头（bid 单调递增）
                }
                bidChunkEnd = bidChunkStart + ((int64_t)seqLens.GetValue(bid) + chunkSize_ - 1) / chunkSize_;
                prevBid = bid;
            }
            int64_t c0 = bidChunkStart > groupChunkStart ? bidChunkStart : groupChunkStart;
            int64_t c1 = bidChunkEnd < groupChunkEnd ? bidChunkEnd : groupChunkEnd;
            int64_t subLen = c1 - c0;
            if (subLen <= 0) {
                continue;
            }
            bool carried = c0 > bidChunkStart;   // 该序列更早的 chunk 在之前的 group 里
            int64_t stateOff = bid * Nv_ * Dv_ * Dk_ + nvId * Dv_ * Dk_ + rowStart * Dk_;
            for (int64_t j = 0; j < subLen; j++) {
                int64_t groupRow = c0 - groupChunkStart + j;
                // 只有全局最后一个 chunk 可能不满（前 b-1 个序列均对齐 chunkSize）
                curChunkSize_ = ((c0 + j) == totalChunks - 1) ? globalLastLen : chunkSize_;
                bool writeFinal = ((subLen - 1 - j) % 2) == 0;
                auto writeState = writeFinal ? sTP_->finalState[stateOff] : sTP_->altState[stateOff];
                auto readState = (j == 0)
                    ? (carried ? sTP_->finalState[stateOff] : sTP_->curState[stateOff])
                    : (writeFinal ? sTP_->altState[stateOff] : sTP_->finalState[stateOff]);
                bool sameBuf = (j == 0) && writeFinal && carried;
                int64_t length = groupRow * chunkSize_;
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0) {
                        CopyIn(readState, dvRows_, Dk_);
                        CalGCumExp(sTP_->gCum[nvId * Sp_ + length]);
                    }
                    if (sameBuf) {
                        CrossCoreWaitFlag(0x2);
                    }
                    if (GetSubBlockIdx() == 0) {
                        CopyOutState(writeState);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
                    CrossCoreWaitFlag(0x4);
                }
                if ASCEND_IS_AIC {
                    uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
                    uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_ + rowStart;
                    CalVPrime(sTP_->kCumdecay[mm_offset0], readState, sTP_->vInner[mm_offset1]);
                    CalAttnInter(sTP_->qPrime[mm_offset0], readState,
                                 sTP_->out[nvId * Dv_ + rowStart + groupRow * chunkSize_ * Nv_ * Dv_]);
                    if (sameBuf) {
                        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);   // 读完之前AIV不能写
                    }
                    CrossCoreWaitFlag(0x5);
                    CalStateNew(sTP_->vInner[mm_offset1], sTP_->kg[mm_offset0], writeState);
                    SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
                }
            }
        }
    }

    __aicore__ inline void CalGCumExp(GlobalTensor<float> gCum)
    {
        if (gOptional_) {
            // 刷新cache
            DataCacheCleanAndInvalid<float,
                                     CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(gCum[curChunkSize_ - 1]);
            float tmpFloat = gCum.GetValue(curChunkSize_ - 1);
            lastGCum_.SetValue(0, tmpFloat);
            SetFlag<HardEvent::S_V>(S_V_EVENT);
            WaitFlag<HardEvent::S_V>(S_V_EVENT);
            Exp<float, 0, true>(lastGCum_, lastGCum_, 1);
        } else {
            lastGCum_.SetValue(0, 1.0f);
        }
        float tmpFloat = lastGCum_.GetValue(0);
        auto stateIn = inQueue_.DeQue<bfloat16_t>();
        auto stateOut = outQueue_.AllocTensor<bfloat16_t>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        Cast(tmpBuffer1_, stateIn, RoundMode::CAST_NONE, dvRows_ * curDk_);
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        Muls(tmpBuffer1_, tmpBuffer1_, tmpFloat, dvRows_ * curDk_);
        Cast(stateOut, tmpBuffer1_, RoundMode::CAST_RINT, dvRows_ * curDk_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(stateOut);
        inQueue_.FreeTensor(stateIn);
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<bfloat16_t> qPrime,
                                        GlobalTensor<bfloat16_t> state,
                                        GlobalTensor<bfloat16_t> out)
    {
        // q_prime @ state.transpose(0, 1)
        // Reuse：state 已被 CalVPrime 拷进对面 set 的 L1B。安全性：MTE2 队列序保证 CalVPrime
        // 的载入先完成；下一笔 CalStateNew 带 serial，不会提前覆写这块 L1B。
        sTP_->mm1->Execute<false, true, false, bfloat16_t, BSource::Reuse>(
            qPrime, state, out, curChunkSize_, dvRows_, Dk_, 0, 0, Nv_ * Dv_);
    }

    __aicore__ inline void CalVPrime(GlobalTensor<bfloat16_t> kCumdecay,
                                     GlobalTensor<bfloat16_t> state,
                                     GlobalTensor<bfloat16_t> vPrime)
    {
        // v_inner += k_cumdecay @ state.transpose(0, 1)（N 维只算本单元的 Dv 行片，输出行距 Dv）
        // serial：state 由上一 chunk 的 CalStateNew（相邻一笔）写出，须等其 Fixpipe
        sTP_->mm1->Execute<false, true, true, bfloat16_t, BSource::Copy, true>(
            kCumdecay, state, vPrime, curChunkSize_, dvRows_, Dk_, 0, 0, Dv_);
    }

    __aicore__ inline void CalStateNew(GlobalTensor<bfloat16_t> vInner,
                                       GlobalTensor<bfloat16_t> kg,
                                       GlobalTensor<bfloat16_t> state)
    {
        // state_out += v_new.transpose(0, 1) @ kg（M 维只算本单元的 Dv 行片，A 源行距 Dv）
        // serial：与 CalAttnInter 的 Reuse 配套（保证不提前覆写其正在读的 L1B）
        sTP_->mm1->Execute<true, false, true, bfloat16_t, BSource::Copy, true>(
            vInner, kg, state, dvRows_, Dk_, curChunkSize_, Dv_);
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)), // 非对齐情况需要补0
                                   static_cast<uint32_t>(0),
                                   0, 0};
        int padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void CopyOutState(GlobalTensor<bfloat16_t> stateNew)
    {
        CopyOut<bfloat16_t>(stateNew, dvRows_, Dk_, false);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>(0);
        if (setAtomic) {
            SetAtomicAdd<outType>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageTwoParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> lastGCum_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int64_t Sp_;
    int64_t dvRows_ = 0;   // 本核每单元负责的 Dv 行数（f=1 时等于 Dv_）
    int32_t chunkNum_;
    int32_t coreNum_;
    bool gOptional_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_H