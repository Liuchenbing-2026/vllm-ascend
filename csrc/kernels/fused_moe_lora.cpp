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
 *
 * ------------------------------------------------------------------------
 * Fused MoE-LoRA kernel: shrink + expand-slice + accumulate in one kernel.
 *
 * Designed as the AscendC equivalent of the combined-index two-call bgmv
 * path in PunicaWrapperNPU._add_lora_fused_moe_bgmv. Per token:
 *
 *   1) Read combined index `vid = expert * max_loras + lora_id`.
 *   2) If vid < 0 → skip the entire row (predicate, no compute).
 *   3) Shrink: rank_buf[r] = scale * sum_k x[i,k] * A[vid,r,k]
 *      (mirrors bgmv_shrink dot-product loop using Mul + ReduceSum;
 *       result stays in UB rather than being written back to GM)
 *   4) Expand: y[i, col_start:col_end] += rank_buf @ B[vid].T
 *      (mirrors bgmv_expand BlockReduceSum / PairReduceSum pattern;
 *       reads rank_buf straight from UB)
 *
 * Compared to running bgmv_shrink + buffer.mul_(mask) + bgmv_expand_slice
 * separately, this version saves:
 *   - 1 kernel launch (3 → 1)
 *   - one round-trip of the rank buffer through DDR
 *   - the dedicated valid_mask multiply pass
 *
 * Status: Framework drafted, awaits NPU verification. The host wrapper
 * lives in csrc/torch_binding.cpp; Python entry is
 * vllm_ascend/lora/punica_npu.py::_add_lora_fused_moe_ascendc.
 * ------------------------------------------------------------------------
 */

#include "kernel_operator.h"
#include "types.h"

template <typename scalar_t>
class FusedMoeLora {
public:
    using X_T = scalar_t;      // hidden states (bf16 / half)
    using WA_T = scalar_t;     // lora_a weights
    using R_T = float;         // rank buffer (intermediate, kept in UB)
    using WB_T = scalar_t;     // lora_b weights
    using Y_T = scalar_t;      // output accumulator

    // ----- Shrink tile (mirrors bgmv_shrink.cpp:TILE_LENGTH) -----
    static constexpr uint64_t SHRINK_BUFFER_NUM = 1;
    static constexpr uint64_t SHRINK_TILE_LENGTH = 11776;

    // ----- Expand layout (mirrors bgmv_expand.cpp constants) -----
    static constexpr int32_t EXPAND_BUFFER_NUM = 2;
    static constexpr int32_t NUM_BYTES_PER_REPEAT = 256;
    static constexpr int32_t NUM_BLOCKS_PER_REPEAT = 8;
    static constexpr int32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);
    static constexpr int32_t MASK_COUNT = NUM_BYTES_PER_REPEAT / sizeof(float);
    static constexpr int32_t W_IN_TILE_NUM_ELEMENTS = 8192;
    static constexpr int32_t Y_OUT_TILE_NUM_ELEMENTS = 4096;
    static constexpr int32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE_NUM_ELEMENTS / NUM_ELEMENTS_PER_REPEAT;
    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_16 =
        (BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) /
        NUM_ELEMENTS_PER_REPEAT;
    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_32 = (PAIR_REDUCE_NUM_REPEATS_16 + 1) / 2;

    // ----- Supported ranks (mirrors bgmv_expand.cpp) -----
    static constexpr uint64_t LORA_RANK_8 = 8;
    static constexpr uint64_t LORA_RANK_16 = 16;
    static constexpr uint64_t LORA_RANK_32 = 32;
    static constexpr uint64_t LORA_RANK_64 = 64;

public:
    __aicore__ inline FusedMoeLora(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *x, __gm__ void *loraA, __gm__ void *loraB,
                                __gm__ void *indices, uint32_t indicesSize, __gm__ void *y,
                                uint32_t batchSize, uint32_t numTokensPerCore,
                                uint32_t inputHiddenDim, uint32_t maxLoRARank,
                                uint32_t outputHiddenDim, uint32_t sliceOffset,
                                uint32_t outputFullDim, float scale)
    {
        // ----- Save parameters -----
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        outputHiddenDim_ = outputHiddenDim;
        sliceOffset_ = sliceOffset;
        outputFullDim_ = outputFullDim;
        scale_ = scale;
        singleLoRAAWeightLen_ = inputHiddenDim_ * maxLoRARank_;
        singleLoRABWeightLen_ = maxLoRARank_ * outputHiddenDim_;
        shrinkIncremental_ = inputHiddenDim_ > SHRINK_TILE_LENGTH;

        // ----- GM buffers -----
        xGm_.SetGlobalBuffer((__gm__ X_T *)x);
        loraAGm_.SetGlobalBuffer((__gm__ WA_T *)loraA);
        loraBGm_.SetGlobalBuffer((__gm__ WB_T *)loraB);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, indicesSize);
        yGm_.SetGlobalBuffer((__gm__ Y_T *)y);

        // ----- Shrink-side UB allocation -----
        pipe_->InitBuffer(inQueueShrinkX_, SHRINK_BUFFER_NUM, SHRINK_TILE_LENGTH * sizeof(X_T));
        pipe_->InitBuffer(inQueueShrinkW_, SHRINK_BUFFER_NUM, SHRINK_TILE_LENGTH * sizeof(WA_T));
        pipe_->InitBuffer(tmpBufferShrinkX_, SHRINK_TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(tmpBufferShrinkW_, SHRINK_TILE_LENGTH * sizeof(float));

        // ----- Rank buffer (resident in UB, NOT going through DDR) -----
        pipe_->InitBuffer(rankBuffer_, maxLoRARank_ * sizeof(float));

        // ----- Expand-side UB allocation -----
        pipe_->InitBuffer(dupBufferX_, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(inQueueExpandW_, EXPAND_BUFFER_NUM, W_IN_TILE_NUM_ELEMENTS * sizeof(WB_T));
        pipe_->InitBuffer(inQueueExpandY_, EXPAND_BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(outQueueExpandY_, EXPAND_BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferExpandW_, W_IN_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(inBufferExpandY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBufferExpandY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));

        // ----- Expand layout helpers (per bgmv_expand) -----
        numOutputElementsPerInputTile_ = BLOCK_REDUCE_NUM_REPEATS * (NUM_ELEMENTS_PER_REPEAT / maxLoRARank_);
        numStreamInPerOutputTile_ = Y_OUT_TILE_NUM_ELEMENTS / numOutputElementsPerInputTile_;
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIdx = blockIdx * numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        if (endIdx > batchSize_) {
            endIdx = batchSize_;
        }
        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            // ----- Combined-index lookup (predicate skip) -----
            reqLoRAIndex_ = indicesGm_.GetValue(idx);
            if (reqLoRAIndex_ < 0) {
                continue;  // invalid / no-LoRA row: device-side skip, zero compute
            }
            reqLoRAAOffset_ = reqLoRAIndex_ * singleLoRAAWeightLen_;
            reqLoRABOffset_ = reqLoRAIndex_ * singleLoRABWeightLen_;
            yOffset_ = outputFullDim_ * idx + sliceOffset_;

            // ----- Stage 1: Shrink. Produces rankBuffer_ in UB. -----
            if (shrinkIncremental_) {
                DoShrink<true>(idx);
            } else {
                DoShrink<false>(idx);
            }

            // ----- Stage 2: Prepare duplicated rank buffer for expand -----
            // BGMVExpand replicates the rank vector to fill NUM_ELEMENTS_PER_REPEAT
            // so that one Mul covers multiple output rows at once.
            BuildDupBuffer();

            // ----- Stage 3: Expand and accumulate into y. -----
            DoExpand();
        }
    }

private:
    // =====================================================================
    // Shrink: x[i] (1 x in_dim) @ A[vid] (rank x in_dim).T -> rank_buf (rank,)
    // Mirrors BGMVShrink::ProcessImpl + ScaleOutput, but writes to rankBuffer_
    // (UB-resident) instead of yOutGm_.
    // =====================================================================
    template <bool INCREMENTAL_MODE>
    __aicore__ inline void DoShrink(const int64_t idx)
    {
        AscendC::LocalTensor<float> rankLocal = rankBuffer_.Get<float>();
        if constexpr (!INCREMENTAL_MODE) {
            // x fits in one tile - load once, cast to fp32, reuse for all ranks.
            CopyInShrinkX(idx, 0, inputHiddenDim_);
            AscendC::LocalTensor<float> xTmp = tmpBufferShrinkX_.Get<float>();
            AscendC::LocalTensor<X_T> xLocal = inQueueShrinkX_.DeQue<X_T>();
            Cast(xTmp, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueShrinkX_.FreeTensor(xLocal);
        }

        for (uint32_t r = 0; r < maxLoRARank_; r++) {
            float acc(0);
            for (uint32_t j = 0; j < inputHiddenDim_ / SHRINK_TILE_LENGTH; j++) {
                if constexpr (INCREMENTAL_MODE) {
                    CopyInShrinkX(idx, j);
                }
                CopyInShrinkW(r, j);
                ShrinkCompute<INCREMENTAL_MODE>(acc);
            }
            ShrinkCopyAndComputeLastTile<INCREMENTAL_MODE>(idx, r, acc);
            // Apply scale here so the duplicated rank vector below is final.
            rankLocal.SetValue(r, acc * scale_);
        }
    }

    __aicore__ inline void CopyInShrinkX(const int64_t idx, int32_t colIdx,
                                         int32_t numElements = SHRINK_TILE_LENGTH)
    {
        AscendC::LocalTensor<X_T> xLocal = inQueueShrinkX_.AllocTensor<X_T>();
        DataCopy(xLocal, xGm_[inputHiddenDim_ * idx + colIdx * SHRINK_TILE_LENGTH], numElements);
        inQueueShrinkX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInShrinkW(int32_t rowIdx, int32_t colIdx,
                                         int32_t numElements = SHRINK_TILE_LENGTH)
    {
        AscendC::LocalTensor<WA_T> wLocal = inQueueShrinkW_.AllocTensor<WA_T>();
        DataCopy(wLocal,
                 loraAGm_[reqLoRAAOffset_ + rowIdx * inputHiddenDim_ + colIdx * SHRINK_TILE_LENGTH],
                 numElements);
        inQueueShrinkW_.EnQue(wLocal);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ShrinkCompute(float &acc, int32_t numElements = SHRINK_TILE_LENGTH)
    {
        AscendC::LocalTensor<WA_T> wLocal = inQueueShrinkW_.DeQue<WA_T>();
        AscendC::LocalTensor<float> xTmp = tmpBufferShrinkX_.Get<float>();
        AscendC::LocalTensor<float> wTmp = tmpBufferShrinkW_.Get<float>();

        if constexpr (INCREMENTAL_MODE) {
            AscendC::LocalTensor<X_T> xLocal = inQueueShrinkX_.DeQue<X_T>();
            Cast(xTmp, xLocal, AscendC::RoundMode::CAST_NONE, numElements);
            Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueShrinkX_.FreeTensor(xLocal);
            inQueueShrinkW_.FreeTensor(wLocal);
        } else {
            Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueShrinkW_.FreeTensor(wLocal);
        }
        Mul(wTmp, xTmp, wTmp, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum<float>(wTmp, wTmp, wTmp, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        acc += wTmp.GetValue(0);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ShrinkCopyAndComputeLastTile(const int64_t idx, int32_t r, float &acc)
    {
        int32_t colIdx = inputHiddenDim_ / SHRINK_TILE_LENGTH;
        int32_t remaining = inputHiddenDim_ % SHRINK_TILE_LENGTH;
        if (remaining == 0) {
            return;
        }
        if constexpr (INCREMENTAL_MODE) {
            CopyInShrinkX(idx, colIdx, remaining);
        }
        CopyInShrinkW(r, colIdx, remaining);
        ShrinkCompute<INCREMENTAL_MODE>(acc, remaining);
    }

    // =====================================================================
    // BuildDupBuffer: replicate rank_buf to fill NUM_ELEMENTS_PER_REPEAT
    // floats so that one Mul covers multiple output rows of B at once.
    // Mirrors the latter half of BGMVExpand::CopyInX.
    // =====================================================================
    __aicore__ inline void BuildDupBuffer()
    {
        AscendC::LocalTensor<float> rankLocal = rankBuffer_.Get<float>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();

        // First copy of rank_buf occupies positions [0, maxLoRARank_).
        for (uint32_t j = 0; j < maxLoRARank_; j++) {
            xDup.SetValue(j, rankLocal.GetValue(j));
        }
        // Replicate to fill NUM_ELEMENTS_PER_REPEAT.
        for (uint32_t i = maxLoRARank_; i < NUM_ELEMENTS_PER_REPEAT; i += maxLoRARank_) {
            for (uint32_t j = 0; j < maxLoRARank_; j++) {
                xDup.SetValue(i + j, rankLocal.GetValue(j));
            }
        }
    }

    // =====================================================================
    // Expand: y[i, col_start:col_end] += rank_buf @ B[vid].T
    // Mirrors BGMVExpand::Process (the inner per-token portion).
    // =====================================================================
    __aicore__ inline void DoExpand()
    {
        int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
        for (int32_t i = 0; i < numStreamOut; i++) {
            CopyInExpandY(i);
            for (int32_t j = 0; j < numStreamInPerOutputTile_; j++) {
                CopyInExpandW(i * numStreamInPerOutputTile_ + j);
                ExpandCompute(j * numOutputElementsPerInputTile_);
            }
            ExpandScaleOutput();
            CopyOutExpand(i);
        }
        ExpandLastTile();
    }

    __aicore__ inline void ExpandLastTile()
    {
        int32_t remainingY = outputHiddenDim_ % Y_OUT_TILE_NUM_ELEMENTS;
        if (remainingY == 0) {
            return;
        }
        int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
        int32_t remainingW = remainingY * maxLoRARank_;
        int32_t numCompleteWTile = remainingW / W_IN_TILE_NUM_ELEMENTS;
        int32_t remainingWTail = remainingW % W_IN_TILE_NUM_ELEMENTS;

        CopyInExpandY(numStreamOut, remainingY);

        int32_t outIdx = 0;
        for (outIdx = 0; outIdx < numCompleteWTile; outIdx++) {
            CopyInExpandW(numStreamOut * numStreamInPerOutputTile_ + outIdx);
            ExpandCompute(outIdx * numOutputElementsPerInputTile_);
        }
        if (remainingWTail != 0) {
            CopyInExpandW(numStreamOut * numStreamInPerOutputTile_ + numCompleteWTile, remainingWTail);
            int32_t lastRepeatCount = remainingWTail / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduce16 =
                (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) /
                NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduce32 = (pairReduce16 + 1) / 2;
            int32_t lastOutEl = outIdx * numOutputElementsPerInputTile_;
            ExpandCompute(lastOutEl, lastRepeatCount, pairReduce16, pairReduce32);
        }
        ExpandScaleOutput(remainingY);
        CopyOutExpand(numStreamOut, remainingY);
    }

    __aicore__ inline void CopyInExpandY(int32_t progress,
                                         int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yLocal = inQueueExpandY_.AllocTensor<Y_T>();
        DataCopy(yLocal, yGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], numElements);
        inQueueExpandY_.EnQue(yLocal);
    }

    __aicore__ inline void CopyInExpandW(int32_t progress,
                                         int32_t numElements = W_IN_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<WB_T> wLocal = inQueueExpandW_.AllocTensor<WB_T>();
        DataCopy(wLocal, loraBGm_[reqLoRABOffset_ + progress * W_IN_TILE_NUM_ELEMENTS], numElements);
        inQueueExpandW_.EnQue(wLocal);
    }

    __aicore__ inline void ExpandCompute(int32_t progress,
                                         int32_t blockReduceRepeat = BLOCK_REDUCE_NUM_REPEATS,
                                         int32_t pairReduce16 = PAIR_REDUCE_NUM_REPEATS_16,
                                         int32_t pairReduce32 = PAIR_REDUCE_NUM_REPEATS_32)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferExpandY_.Get<float>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();
        AscendC::LocalTensor<WB_T> wLocal = inQueueExpandW_.DeQue<WB_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferExpandW_.Get<float>();

        Cast(wTmp, wLocal, AscendC::RoundMode::CAST_NONE, MASK_COUNT, blockReduceRepeat, castParams_);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueExpandW_.FreeTensor(wLocal);

        Mul(wTmp, xDup, wTmp, MASK_COUNT, blockReduceRepeat, dotProductParams_);
        AscendC::PipeBarrier<PIPE_V>();

        if (maxLoRARank_ == LORA_RANK_8) {
            BlockReduceSum(yLocal[progress], wTmp, blockReduceRepeat, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                           reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (maxLoRARank_ == LORA_RANK_16) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeat, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                           reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[progress], wTmp, pairReduce16, MASK_COUNT,
                          reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                          reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (maxLoRARank_ == LORA_RANK_32) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeat, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                           reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(wTmp, wTmp, pairReduce16, MASK_COUNT,
                          reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                          reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[progress], wTmp, pairReduce32, MASK_COUNT,
                          reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                          reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (maxLoRARank_ == LORA_RANK_64) {
            BlockReduceSum(wTmp, wTmp, blockReduceRepeat, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                           reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            BlockReduceSum(yLocal[progress], wTmp, pairReduce16, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride,
                           reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ExpandScaleOutput(int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferExpandY_.Get<float>();
        AscendC::LocalTensor<Y_T> yInLocal = inQueueExpandY_.DeQue<Y_T>();
        AscendC::LocalTensor<float> yInFp32 = inBufferExpandY_.Get<float>();

        Cast(yInFp32, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueExpandY_.FreeTensor(yInLocal);

        Add(yLocal, yLocal, yInFp32, numElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<Y_T> yOutLocal = outQueueExpandY_.AllocTensor<Y_T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();

        outQueueExpandY_.EnQue<Y_T>(yOutLocal);
    }

    __aicore__ inline void CopyOutExpand(int32_t progress,
                                         int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueExpandY_.DeQue<Y_T>();
        DataCopy(yGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], yOutLocal, numElements);
        outQueueExpandY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;

    // Shrink-side queues / buffers
    AscendC::TQue<AscendC::QuePosition::VECIN, SHRINK_BUFFER_NUM> inQueueShrinkX_, inQueueShrinkW_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferShrinkX_, tmpBufferShrinkW_;

    // Shared rank buffer (this is the fusion point: UB-resident, not GM)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rankBuffer_;

    // Expand-side queues / buffers
    AscendC::TQue<AscendC::QuePosition::VECIN, EXPAND_BUFFER_NUM> inQueueExpandW_, inQueueExpandY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, EXPAND_BUFFER_NUM> outQueueExpandY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dupBufferX_, tmpBufferExpandW_, inBufferExpandY_,
        tmpBufferExpandY_;

    // GM views
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<WA_T> loraAGm_;
    AscendC::GlobalTensor<WB_T> loraBGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    AscendC::GlobalTensor<Y_T> yGm_;

    // Scalars
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t outputHiddenDim_;
    uint32_t sliceOffset_;
    uint32_t outputFullDim_;
    float scale_;
    uint32_t singleLoRAAWeightLen_;
    uint32_t singleLoRABWeightLen_;
    int64_t reqLoRAIndex_;
    uint64_t reqLoRAAOffset_;
    uint64_t reqLoRABOffset_;
    uint64_t yOffset_;
    bool shrinkIncremental_;
    uint32_t numOutputElementsPerInputTile_;
    uint32_t numStreamInPerOutputTile_;

    // Repeat params (copied from bgmv_expand.cpp)
    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::UnaryRepeatParams reduceSumParams_ = {1, 1, 1, 8};
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};
};

#define FUSED_MOE_LORA_TYPE_DECLARE(TYPE)                                                              \
    extern "C" __global__ __aicore__ void fused_moe_lora_##TYPE(                                       \
        __gm__ void *x, __gm__ void *loraA, __gm__ void *loraB, __gm__ void *indices,                  \
        uint32_t indicesSize, __gm__ void *y, uint32_t batchSize, uint32_t numTokensPerCore,           \
        uint32_t inputHiddenDim, uint32_t maxLoRARank, uint32_t outputHiddenDim,                       \
        uint32_t sliceOffset, uint32_t outputFullDim, float scale)                                     \
    {                                                                                                  \
        AscendC::TPipe pipe;                                                                           \
        FusedMoeLora<TYPE> op(&pipe);                                                                  \
        op.Init(x, loraA, loraB, indices, indicesSize, y, batchSize, numTokensPerCore,                 \
                inputHiddenDim, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim, scale);      \
        op.Process();                                                                                  \
    }

FUSED_MOE_LORA_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
FUSED_MOE_LORA_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void fused_moe_lora_impl(AscendType type, void *stream, void *x, void *loraA, void *loraB,
                                void *indices, uint32_t indicesSize, void *y, uint32_t batchSize,
                                uint32_t numTokensPerCore, uint32_t inputHiddenDim,
                                uint32_t maxLoRARank, uint32_t outputHiddenDim,
                                uint32_t sliceOffset, uint32_t outputFullDim, float scale)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == AscendType::FP16) {
        fused_moe_lora_half<<<blockDim, nullptr, stream>>>(
            x, loraA, loraB, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim,
            maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim, scale);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        fused_moe_lora_bfloat16_t<<<blockDim, nullptr, stream>>>(
            x, loraA, loraB, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim,
            maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim, scale);
#endif
    } else {
        return;
    }
}

}  // namespace vllm_ascend
