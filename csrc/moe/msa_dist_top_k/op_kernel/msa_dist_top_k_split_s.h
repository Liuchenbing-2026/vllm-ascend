/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

/*!
 * \file msa_dist_top_k_split_s.h
 * \brief MiniMax-M3 MSA decode block-selection kernel (tiling key 10).
 *
 * Per (decode query token b, kv-group g=1):
 *   1. SCORE   S_j = (iq . ik_j) * 1/sqrt(128)  (exact bf16 dot, dim=128),
 *              computed by the Cube via a bf16 K=128 matmul C[1,L] = iq[1,128] . ikf[L,128]^T,
 *              keys gathered per-physical-block through block_table.
 *   2. CAUSAL  positions j > seq_len-1 forced to MIN_HALF_VALUE before pooling.
 *   3. POOL    max over each 128-token block -> M_b (WholeReduceMax-128).
 *   4. TOPK    top-16 blocks by M_b via the AscendC TopK API.
 *   5. LOCAL   force-keep block (seq_len-1)/128 (recent=1), no sink (init=0).
 *   6. WRITE   selected logical block ids ascending, local block written LAST,
 *              int32 indices[B, G, topk_total].
 *
 * This is the batch-parallel decode clone of the hamming Parallel kernel, kept
 * under TILING_KEY_IS(10) per the task spec. All binarization / popcount /
 * int4b_t / Select / CumSum / rope / offload machinery has been deleted.
 */

#ifndef MSA_DIST_TOP_K_SPLIT_S_H
#define MSA_DIST_TOP_K_SPLIT_S_H

#include "msa_dist_top_k_base.h"

namespace AscendC {

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t RESET_NUM = 0U;
constexpr uint32_t VECTOR_CUBE_RATIO = 2;
constexpr uint32_t MAX_MSA_BATCH = 256;

class MsaDistTopKSplitSKernel {
public:
    __aicore__ inline MsaDistTopKSplitSKernel() {}

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR iq, GM_ADDR idxkCache, GM_ADDR seqLen, GM_ADDR keyBlockTable,
                                GM_ADDR indices, GM_ADDR workSpace, const TilingT &tilingData, TPipe *pipe)
    {
        const TCubeTiling &tiling = tilingData.matmulTiling;
        pipe_ = pipe;
        InitTilingParams(tilingData);
        InitParams();
        InitGlobalBuffers(iq, idxkCache, seqLen, keyBlockTable, indices, workSpace);

        mm_.SetSubBlockIdx(0);
        mm_.Init(&tiling, pipe_);
    }

    template <typename TilingT>
    __aicore__ inline void Process(const TilingT &tilingData)
    {
        // One core handles a contiguous slice of (batch * head) rows. With G=1
        // for decode, batchN == batch. Cube produces scores into matmulGm_, the
        // paired Vector (subBlockIdx 0) pools + TopK + writes indices.
        //
        // Row assignment must be identical on AIC and its paired AIV so they
        // share the same matmulGm_ score region. The MIX_AIC_1_2 launch gives
        // VECTOR_CUBE_RATIO (=2) vector cores per cube core; both vector cores
        // of a pair derive the same coreRow as their cube via blockIdx_/ratio.
        uint32_t coreRow = blockIdx_;
        if ASCEND_IS_AIV {
            coreRow = blockIdx_ / VECTOR_CUBE_RATIO;
        }
        uint32_t coreStride = param_.usedCoreNum;

        if ASCEND_IS_AIV {
            InitLocalBuffers();
        }

        for (uint32_t row = coreRow; row < param_.batchN; row += coreStride) {
            uint32_t batchIdx = row / param_.head;
            uint32_t headIdx = row % param_.head;

            uint32_t curSeqLen = static_cast<uint32_t>(seqLenGm_.GetValue(batchIdx));
            uint32_t numBlocks = (curSeqLen == 0) ? 0 : ((curSeqLen + MSA_BLOCK_SIZE - 1) / MSA_BLOCK_SIZE);

            if ASCEND_IS_AIC {
                if (curSeqLen != 0) {
                    ComputeMM(batchIdx, headIdx, numBlocks);
                }
                CubeNotifyVector(SYNC_AIC_AIV_FLAG);
            }
            if ASCEND_IS_AIV {
                VectorWaitCube(SYNC_AIC_AIV_FLAG);
                // Only the first vector of each pair does the (cheap) selection
                // for v1; the second idles to keep the handshake balanced.
                if (subBlockIdx_ == 0) {
                    if (curSeqLen == 0) {
                        WriteEmpty(row);
                    } else {
                        ComputeTopK(row, batchIdx, headIdx, curSeqLen, numBlocks, tilingData);
                    }
                }
            }
        }
    }

protected:
    // ------------------------------------------------------------------
    // Cube: scores[1, numBlocks*128] = iq[1,128] . keys[numBlocks*128, 128]^T
    // keys are the gathered idxk_cache physical blocks for this seq.
    // The B operand is declared transposed so the per-token-major key layout
    // [n_keys, dim] yields C[1, n_keys] directly.
    // ------------------------------------------------------------------
    __aicore__ inline void ComputeMM(uint32_t batchIdx, uint32_t headIdx, uint32_t numBlocks)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // A: iq for this (batch, head). Decode -> 1 query token, G=1.
        uint64_t offA = (batchIdx * param_.head + headIdx) * param_.ka;
        // C: this row's score region in workspace (maxSeqLen wide per row).
        uint64_t offC = static_cast<uint64_t>(batchIdx * param_.head + headIdx) * param_.maxSeqLen;

        // Score each logical block directly against its physical idxk_cache block.
        // Each physical block is contiguous [block_size, dim] in idxk_cache, so the
        // matmul reads B straight from the paged cache (no GM->GM gather needed):
        //   C[1, block_size] = iq[1, dim] . ik_block[block_size, dim]^T   (B transposed)
        // written to the contiguous per-row score region at offset b*block_size.
        for (uint32_t b = 0; b < numBlocks; ++b) {
            int32_t phys = keyBlockTableGm_.GetValue(batchIdx * param_.blockCount + b);
            uint64_t bOff = static_cast<uint64_t>(phys) * MSA_BLOCK_SIZE * param_.dimension;
            mm_.SetOrgShape(param_.M, MSA_BLOCK_SIZE, param_.ka);
            mm_.SetSingleShape(param_.M, MSA_BLOCK_SIZE, param_.ka);
            mm_.SetTensorA(iqGm_[offA], AMatmulType::isTrans);
            mm_.SetTensorB(idxkCacheGm_[bOff], BMatmulType::isTrans);
            mm_.IterateAll(matmulGm_[offC + static_cast<uint64_t>(b) * MSA_BLOCK_SIZE]);
        }
    }

    // ------------------------------------------------------------------
    // Vector: causal mask -> block max-pool -> force local -> TopK -> write.
    // ------------------------------------------------------------------
    template <typename TilingT>
    __aicore__ inline void ComputeTopK(uint32_t row, uint32_t batchIdx, uint32_t headIdx, uint32_t curSeqLen,
        uint32_t numBlocks, const TilingT &tilingData)
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint64_t scoreOff = static_cast<uint64_t>(row) * param_.maxSeqLen;

        // 1) Causal mask: positions j >= curSeqLen forced to MIN_HALF_VALUE so
        //    future tokens never win the per-block max. Only the local
        //    (partial) block has a tail to mask; full blocks are entirely valid.
        uint32_t nKeysPadded = numBlocks * MSA_BLOCK_SIZE;
        uint32_t invalidTail = nKeysPadded - curSeqLen; // tokens beyond seq_len in the last block
        if (invalidTail > 0) {
            LocalTensor<half> maskTmp = scratchQueue_.AllocTensor<half>();
            Duplicate(maskTmp, static_cast<half>(MIN_HALF_VALUE), invalidTail);
            scratchQueue_.EnQue(maskTmp);
            maskTmp = scratchQueue_.DeQue<half>();
            DataCopyExtParams cpTail{1, static_cast<uint32_t>(invalidTail * sizeof(half)), 0, 0, 0};
            DataCopyPad(matmulGm_[scoreOff + curSeqLen], maskTmp, cpTail);
            scratchQueue_.FreeTensor(maskTmp);
            SetFlag<HardEvent::MTE3_MTE2>(1);
            WaitFlag<HardEvent::MTE3_MTE2>(1);
        }

        // 2) Block max-pool over each 128-token block -> M_b (numBlocks values).
        //    Pre-fill the full TopK inner range [0, innerN) with MIN from an
        //    aligned (offset 0) pre-fill, then let the pool overwrite the leading
        //    numBlocks values. This avoids the unaligned VEC Duplicate at
        //    poolValue[numBlocks] (numBlocks is rarely a multiple of 16 -> the
        //    half offset numBlocks*2B violates the 32B VEC UB-address alignment).
        uint32_t innerN = matmul::CeilDiv(numBlocks, 32) * 32;
        LocalTensor<half> poolValue = topKValueInQueue_.AllocTensor<half>();
        LocalTensor<half> reduceScratch = scratchQueue_.AllocTensor<half>();
        Duplicate(poolValue, static_cast<half>(MIN_HALF_VALUE), innerN);
        PipeBarrier<PIPE_V>();
        ReduceMaxBlock128(matmulGm_[scoreOff], reduceScratch, poolValue, static_cast<uint16_t>(numBlocks));
        scratchQueue_.FreeTensor(reduceScratch);

        // 3) Determine k and the local block index.
        uint32_t localBlk = (curSeqLen - 1) / MSA_BLOCK_SIZE;     // forced block
        uint32_t kBlocks = Min(param_.maxK, numBlocks);            // top-k, capped to valid blocks

        // 4) Force-keep the local block. The local block is the LAST valid block
        //    (numBlocks-1) in the pooled tensor, so reuse the tail-fill helper.
        if (param_.localBlocks > 0) {
            FillMaxValueFromTail(poolValue, numBlocks, param_.localBlocks);
        }
        PipeBarrier<PIPE_V>();

        topKValueInQueue_.EnQue(poolValue);
        poolValue = topKValueInQueue_.DeQue<half>();

        // 5) TopK over the numBlocks pooled scores. Candidate indices are the
        //    logical block ids 0..numBlocks-1.
        LocalTensor<int32_t> topKInIndex = topKIndexInQueue_.AllocTensor<int32_t>();
        ArithProgression(topKInIndex, 0, 1, static_cast<int32_t>(numBlocks));
        topKIndexInQueue_.EnQue(topKInIndex);
        topKInIndex = topKIndexInQueue_.DeQue<int32_t>();

        LocalTensor<half> topKOutValue = topKValueOutQueue_.AllocTensor<half>();
        LocalTensor<int32_t> topKOutIndex = topKIndexOutQueue_.AllocTensor<int32_t>();
        TopKCustom(topKOutValue, topKOutIndex, poolValue, topKInIndex, static_cast<int32_t>(kBlocks),
            tilingData, numBlocks);
        topKValueInQueue_.FreeTensor(poolValue);
        topKIndexInQueue_.FreeTensor(topKInIndex);
        topKValueOutQueue_.FreeTensor(topKOutValue);

        // 6) Write selected logical block ids: the local block is force-kept and
        //    appears among the kBlocks winners; we drop it from the ascending
        //    set, sort the rest ascending, then append the local block LAST.
        WriteSelected(row, batchIdx, topKOutIndex, kBlocks, localBlk);
        topKIndexOutQueue_.FreeTensor(topKOutIndex);
    }

    // Writes the canonical ordered logical-block list for this row:
    //   [ ascending(top-k winners minus local), local_block ]
    // padded to topkTotal with the local block id (dedup-safe duplicate of the
    // already-selected local block is harmless because FIA reads it once via the
    // gathered block_table; the python glue dedups for actual_seq_lengths_kv).
    __aicore__ inline void WriteSelected(uint32_t row, uint32_t batchIdx, LocalTensor<int32_t> &topKOutIndex,
        uint32_t kBlocks, uint32_t localBlk)
    {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<int32_t> outBuf = outIndexBuf_.template Get<int32_t>();
        __ubuf__ const int32_t *winners = reinterpret_cast<__ubuf__ const int32_t *>(topKOutIndex.GetPhyAddr());
        __ubuf__ int32_t *out = reinterpret_cast<__ubuf__ int32_t *>(outBuf.GetPhyAddr());

        // Collect winners excluding the local block (it is placed last).
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < kBlocks; ++i) {
            int32_t blk = winners[i];
            if (blk < 0) {
                continue;
            }
            if (static_cast<uint32_t>(blk) == localBlk && param_.localBlocks > 0) {
                continue; // drop, will append last
            }
            out[cnt++] = blk;
        }

        // Sort the non-local winners ascending.
        SortInt32AscendingUB(outBuf, cnt);

        // Append the local block last (recent=1).
        uint32_t writeLen = cnt;
        if (param_.localBlocks > 0) {
            out[cnt] = static_cast<int32_t>(localBlk);
            writeLen = cnt + param_.localBlocks; // local appended (+1)
        }

        // Pad up to topkTotal with the local block id so the output shape is
        // static [B, G, topkTotal] for graph capture. Extra slots repeat the
        // local block; python dedups when building actual_seq_lengths_kv.
        for (uint32_t i = writeLen; i < param_.topkTotal; ++i) {
            out[i] = static_cast<int32_t>(localBlk);
        }

        outIndexBuf_.template EnQue<int32_t>(outBuf);
        outBuf = outIndexBuf_.template DeQue<int32_t>();
        uint64_t outOff = static_cast<uint64_t>(row) * param_.topkTotal;
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(param_.topkTotal * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indicesGm_[outOff], outBuf, cpOut);
    }

    // For a zero-length / padded row, emit a deterministic block-0 list.
    __aicore__ inline void WriteEmpty(uint32_t row)
    {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<int32_t> outBuf = outIndexBuf_.template Get<int32_t>();
        Duplicate(outBuf, static_cast<int32_t>(0), param_.topkTotal);
        outIndexBuf_.template EnQue<int32_t>(outBuf);
        outBuf = outIndexBuf_.template DeQue<int32_t>();
        uint64_t outOff = static_cast<uint64_t>(row) * param_.topkTotal;
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(param_.topkTotal * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indicesGm_[outOff], outBuf, cpOut);
    }

    // ---- sync helpers (simplified single Cube : Vector handshake) ----
    __aicore__ inline void CubeNotifyVector(uint16_t evt)
    {
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(evt);
    }
    __aicore__ inline void VectorWaitCube(uint16_t evt)
    {
        CrossCoreWaitFlag(evt);
    }

    // ---- init ----
    __aicore__ inline void InitParams()
    {
        blockIdx_ = GetBlockIdx();
        subBlockIdx_ = GetSubBlockIdx();
    }

    template <typename TilingT>
    __aicore__ inline void InitTilingParams(const TilingT &tilingData)
    {
        const TCubeTiling &tiling = tilingData.matmulTiling;
        const auto &p = tilingData.params;
        param_.usedCoreNum = p.usedCoreNum;
        param_.batch = p.batch;
        param_.head = p.head;
        param_.batchN = p.batchN;
        param_.dimension = p.dimension;
        param_.maxSeqLen = p.maxSeqLen;
        param_.maxK = p.maxK;
        param_.blockCount = p.blockCount;
        param_.topkTotal = p.topkTotal;
        param_.localBlocks = p.localBlocks;
        param_.initBlocks = p.initBlocks;
        param_.topKInnerSize = p.topKInnerSize;
        param_.matmulResultSize = p.matmulResultSize;
        param_.M = tiling.M;
        param_.N = tiling.N;
        param_.ka = tiling.Ka;
        param_.kb = tiling.Kb;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR iq, GM_ADDR idxkCache, GM_ADDR seqLen, GM_ADDR keyBlockTable,
        GM_ADDR indices, GM_ADDR workSpace)
    {
        iqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(iq));
        idxkCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(idxkCache));
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen));
        keyBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(keyBlockTable));
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));

        // Workspace layout (host tiling must size user workspace to match):
        //   [0]        gathered keys, per-cube-core: usedCoreNum * maxSeqLen * dim (bf16)
        //   [keyBytes] matmul scores: batchN * maxSeqLen (half)
        uint64_t keyBytes = static_cast<uint64_t>(param_.usedCoreNum) * param_.maxSeqLen * param_.dimension
            * sizeof(bfloat16_t);
        keyGatherGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workSpace));
        matmulGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workSpace + keyBytes));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        // Pooled value tensor: one half per block, padded to topKInnerSize.
        uint32_t innerSize = Max(param_.topKInnerSize, static_cast<uint32_t>(MAX_FP16_PROCESS_NUM));
        pipe_->InitBuffer(topKValueInQueue_, 1, innerSize * sizeof(half));
        pipe_->InitBuffer(topKIndexInQueue_, 1, innerSize * sizeof(int32_t));
        pipe_->InitBuffer(topKValueOutQueue_, 1, param_.maxK * sizeof(half) + DATABLOCK_BYTES);
        pipe_->InitBuffer(topKIndexOutQueue_, 1, param_.maxK * sizeof(int32_t) + DATABLOCK_BYTES);
        pipe_->InitBuffer(scratchQueue_, 1, (param_.maxSeqLen + MAX_FP16_PROCESS_NUM) * sizeof(half));
        pipe_->InitBuffer(outIndexBuf_, (param_.topkTotal + 16) * sizeof(int32_t));
    }

protected:
    TPipe *pipe_;

    GlobalTensor<bfloat16_t> iqGm_;
    GlobalTensor<bfloat16_t> idxkCacheGm_;
    GlobalTensor<bfloat16_t> keyGatherGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<int32_t> keyBlockTableGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<half> matmulGm_;

    TQue<TPosition::VECIN, 1> topKValueInQueue_;
    TQue<TPosition::VECIN, 1> topKIndexInQueue_;
    TQue<TPosition::VECOUT, 1> topKValueOutQueue_;
    TQue<TPosition::VECOUT, 1> topKIndexOutQueue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> scratchQueue_;
    TBuf<TPosition::VECCALC> outIndexBuf_;

    MsaTilingParam param_;
    uint32_t blockIdx_ = 0;
    uint32_t subBlockIdx_ = 0;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint16_t SYNC_AIC_AIV_FLAG = 4;

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_CFG_NO_PRELOAD> mm_;
};

} // namespace AscendC
#endif // MSA_DIST_TOP_K_SPLIT_S_H

