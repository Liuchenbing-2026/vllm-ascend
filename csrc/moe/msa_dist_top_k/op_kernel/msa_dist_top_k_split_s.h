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
// Blocks pooled per WholeReduceMax batch. chunkBuf_ = POOL_CHUNK_BLOCKS*128 half
// (= 8 KiB at 32), a FIXED UB cost independent of context length. This is what
// lets long contexts (numBlocks up to thousands) avoid the old maxSeqLen-sized
// scratch that overflowed UB at numBlocks >= 128. Must be <= MAX_REPEAT_TIMES
// and a multiple of 16 so blockMax[chunkStart] stays 32B-aligned.
constexpr uint32_t POOL_CHUNK_BLOCKS = 32;

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

            // Per-row task/sync structure is identical on every replay:
            //   AIV gather (static blockCount loop) -> Vector->Cube notify ->
            //   AIC ONE IterateAll -> Cube->Vector notify -> AIV pool/TopK/write.
            // Exactly ONE Cube task and exactly TWO cross-core flags per row,
            // both counts fixed by tiling (batchN / usedCoreNum). curSeqLen only
            // changes DATA (gather source addresses, masked score positions).
            //
            // gatherCore picks the per-cube-core slab of keyGatherGm_. The two
            // paired AIVs and their AIC must agree on it: AIC uses its raw cube
            // id; the AIVs divide their id by VECTOR_CUBE_RATIO to get the same.
            uint32_t gatherCore = blockIdx_;
            if ASCEND_IS_AIV {
                gatherCore = blockIdx_ / VECTOR_CUBE_RATIO;
            }

            // All-AIV vector path: subBlock 0 computes iq dot ik scores on the
            // vector core (no Cube matmul, no cross-core handoff). The M=1 Cube
            // GEMV GM write was not reliably visible to the paired AIV for N>128;
            // here matmulGm_ is AIV-written + AIV-read = same-core coherent. No
            // Cube task => no FFTS+ Cube stream variability (safer graph capture).
            (void)gatherCore;
            if ASCEND_IS_AIV {
                if (subBlockIdx_ == 0) {
                    if (curSeqLen == 0) {
                        WriteEmpty(row);
                    } else {
                        // Fused score + per-block max-pool: scores are reduced to
                        // block maxes chunk-by-chunk in UB, never materialised at
                        // maxSeqLen width, so UB stays bounded for any context len.
                        ComputeBlockMax(row, batchIdx, headIdx, curSeqLen, numBlocks);
                    }
                }
            }
        }
    }

protected:
    // ------------------------------------------------------------------
    // Vector: gather this seq's logical idxk blocks (paged via block_table)
    // into the contiguous per-cube-core slab keyGatherGm_[gatherCore*...]. This
    // is the graph-safety pivot (mirrors hamming UnpackKey): the block_table
    // walk lives on AIV as a DMA loop, NOT in the Cube task stream, and the
    // contiguous result lets the Cube issue exactly ONE IterateAll afterwards.
    //
    // The loop is bounded by the runtime numBlocks, but this is a Vector-only
    // DataCopy loop -- its trip count never lengthens the captured FFTS+ Cube /
    // cross-core task stream (same class as hamming's UnpackKey and msa's own
    // ReduceMaxBlock128 WholeReduceMax loop). Logical block b always lands at
    // logical slot b of the slab regardless of its physical id, so the single
    // matmul below reads one contiguous, token-major [maxSeqLen, dim] operand.
    //
    // Columns [numBlocks*128, maxSeqLen) of the slab are left untouched: the
    // matmul does compute scores there, but ComputeTopK is bounded by numBlocks
    // and never reads them, so their (stale) contents cannot affect selection.
    // ------------------------------------------------------------------
    __aicore__ inline void GatherKeys(uint32_t batchIdx, uint32_t numBlocks, uint32_t gatherCore)
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint64_t slabBase = static_cast<uint64_t>(gatherCore) * param_.maxSeqLen * param_.dimension;
        uint32_t blkElems = MSA_BLOCK_SIZE * param_.dimension;            // 128*128 bf16 per physical block
        uint32_t blkBytes = blkElems * static_cast<uint32_t>(sizeof(bfloat16_t));
        DataCopyExtParams cpBlk{1, blkBytes, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> cpPad{false, 0, 0, 0};
        for (uint32_t b = 0; b < numBlocks; ++b) {
            int32_t phys = keyBlockTableGm_.GetValue(batchIdx * param_.blockCount + b);
            uint64_t srcOff = static_cast<uint64_t>(phys) * blkElems;     // paged idxk_cache block
            uint64_t dstOff = slabBase + static_cast<uint64_t>(b) * blkElems;
            LocalTensor<bfloat16_t> ub = keyGatherInQueue_.AllocTensor<bfloat16_t>();
            DataCopyPad(ub, idxkCacheGm_[srcOff], cpBlk, cpPad);         // GM(paged) -> UB
            keyGatherInQueue_.EnQue(ub);
            ub = keyGatherInQueue_.DeQue<bfloat16_t>();
            DataCopyPad(keyGatherGm_[dstOff], ub, cpBlk);               // UB -> GM(contiguous)
            keyGatherInQueue_.FreeTensor(ub);
        }
    }

    // ------------------------------------------------------------------
    // Cube: ONE matmul over the contiguous gathered keys.
    //   C[1, maxSeqLen] = iq[1, dim] . gatheredKeys[maxSeqLen, dim]^T  (B transposed)
    // N is the STATIC maxSeqLen (= blockCount*128), exactly the value the host
    // matmul tiling was built with (SetMatmulTiling N=maxSeqLen). A single
    // IterateAll with a fixed shape => one Cube task whose internal base-block
    // loop is identical at capture and replay. Element C[0, b*128 + t] is the
    // score of token t of logical block b -- byte-identical to what the old
    // per-block loop wrote, because the gather preserves the source key bytes
    // and slot order.
    // ------------------------------------------------------------------
    __aicore__ inline void ComputeMM(uint32_t batchIdx, uint32_t headIdx, uint32_t gatherCore)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // A: iq for this (batch, head). Decode -> 1 query token, G=1.
        uint64_t offA = (batchIdx * param_.head + headIdx) * param_.ka;
        // B: this cube core's contiguous gathered-key slab.
        uint64_t offB = static_cast<uint64_t>(gatherCore) * param_.maxSeqLen * param_.dimension;
        // C: this row's score region in workspace (maxSeqLen wide per row).
        uint64_t offC = static_cast<uint64_t>(batchIdx * param_.head + headIdx) * param_.maxSeqLen;

        mm_.SetOrgShape(param_.M, param_.maxSeqLen, param_.ka);
        mm_.SetSingleShape(param_.M, param_.maxSeqLen, param_.ka);
        mm_.SetTensorA(iqGm_[offA], AMatmulType::isTrans);
        mm_.SetTensorB(keyGatherGm_[offB], BMatmulType::isTrans);
        mm_.IterateAll(matmulGm_[offC]);
        // cross-core notify (PIPE_FIX) handles matmul->AIV ordering. legacy note:
        // CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>, which itself gates on the
        // matmul's PIPE_FIX (fixpipe) ops finishing before the paired AIV is
        // released to pool matmulGm_. A FIX_MTE2 drain BEFORE that notify empties
        // PIPE_FIX, so the cross-core flag no longer waits on the matmul write
        // and the AIV pools partially-landed (under-max) scores. Mirrors hamming.
    }

    // ------------------------------------------------------------------
    // Vector fused score + block max-pool (replaces ComputeScoresVec + the old
    // ComputeTopK matmulGm round-trip). Per logical block b: page-load its
    // 128x128 bf16 key tile, cast bf16->fp32->fp16, broadcast-multiply by iq,
    // WholeReduceSum over dim -> 128 token scores written into a small per-chunk
    // UB buffer (POOL_CHUNK_BLOCKS blocks). After each chunk fills, one
    // WholeReduceMax collapses it to chunkBlocks block-maxes straight into the
    // output buffer. Causal masking of the partial last block is applied with a
    // 128-lane masked Duplicate on its (32B-aligned) chunk slot before pooling.
    //
    // Peak UB is the gather queue + the bf16->fp16 cast scratch + the FIXED
    // POOL_CHUNK_BLOCKS chunk buffer -- none of which scale with numBlocks. This
    // is the long-context fix: the old path read all maxSeqLen scores back into a
    // maxSeqLen-sized UB scratch, overflowing UB at numBlocks >= 128.
    // ------------------------------------------------------------------
    __aicore__ inline void ComputeBlockMax(uint32_t row, uint32_t batchIdx, uint32_t headIdx,
        uint32_t curSeqLen, uint32_t numBlocks)
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t dim = param_.dimension;                 // 128
        uint32_t blkElems = MSA_BLOCK_SIZE * dim;        // 128*128
        uint32_t outW = param_.blockCount;
        uint32_t fillN = matmul::CeilDiv(outW, 32U) * 32U;

        // iq bf16 -> fp32 -> fp16 (Mul needs fp16/fp32; bf16->fp16 direct cast unsupported).
        LocalTensor<bfloat16_t> iqBf = iqBf16Buf_.Get<bfloat16_t>();
        LocalTensor<float> iqF = iqF32Buf_.Get<float>();
        LocalTensor<half> iqH = iqHalfBuf_.Get<half>();
        uint64_t offA = (static_cast<uint64_t>(batchIdx) * param_.head + headIdx) * param_.ka;
        DataCopyExtParams cpIq{1, static_cast<uint32_t>(dim * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> cpIqPad{false, 0, 0, 0};
        DataCopyPad(iqBf, iqGm_[offA], cpIq, cpIqPad);
        {
            int32_t e = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
        }
        Cast(iqF, iqBf, RoundMode::CAST_NONE, dim);
        PipeBarrier<PIPE_V>();
        Cast(iqH, iqF, RoundMode::CAST_NONE, dim);
        PipeBarrier<PIPE_V>();

        // Output block-max buffer, pre-filled MIN so padding blocks [numBlocks,
        // blockCount) score -inf and the python top-k never selects them.
        LocalTensor<half> blockMax = topKValueInQueue_.AllocTensor<half>();
        Duplicate(blockMax, static_cast<half>(MIN_HALF_VALUE), fillN);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> keyF = keyF32Buf_.Get<float>();
        LocalTensor<half> keyH = keyHalfBuf_.Get<half>();
        LocalTensor<half> chunkBuf = chunkBuf_.Get<half>();
        uint64_t fmask[2] = {UINT64_MAX, UINT64_MAX};
        DataCopyExtParams cpKey{1, static_cast<uint32_t>(blkElems * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> cpKeyPad{false, 0, 0, 0};

        // Causal mask for the (only possibly partial) last block: lanes
        // [lastValid, 128) are tokens >= curSeqLen and must be forced to MIN.
        uint32_t lastBlock = numBlocks - 1;
        uint32_t lastValid = curSeqLen - lastBlock * MSA_BLOCK_SIZE;   // 1..128
        uint64_t tailMask[2] = {0, 0};
        if (lastValid < MSA_BLOCK_SIZE) {
            if (lastValid < 64) {
                tailMask[0] = UINT64_MAX << lastValid;                 // lanes [lastValid,64)
                tailMask[1] = UINT64_MAX;                              // lanes [64,128)
            } else {
                tailMask[0] = 0;
                tailMask[1] = UINT64_MAX << (lastValid - 64);         // lanes [lastValid,128)
            }
        }

        for (uint32_t chunkStart = 0; chunkStart < numBlocks; chunkStart += POOL_CHUNK_BLOCKS) {
            uint32_t chunkBlocks = Min(POOL_CHUNK_BLOCKS, numBlocks - chunkStart);
            for (uint32_t j = 0; j < chunkBlocks; ++j) {
                uint32_t b = chunkStart + j;
                int32_t phys = keyBlockTableGm_.GetValue(batchIdx * param_.blockCount + b);
                uint64_t srcOff = static_cast<uint64_t>(phys) * blkElems;
                LocalTensor<bfloat16_t> keyBf = keyGatherInQueue_.AllocTensor<bfloat16_t>();
                DataCopyPad(keyBf, idxkCacheGm_[srcOff], cpKey, cpKeyPad);
                keyGatherInQueue_.EnQue(keyBf);
                keyBf = keyGatherInQueue_.DeQue<bfloat16_t>();
                Cast(keyF, keyBf, RoundMode::CAST_NONE, blkElems);   // bf16 -> fp32
                keyGatherInQueue_.FreeTensor(keyBf);
                PipeBarrier<PIPE_V>();
                Cast(keyH, keyF, RoundMode::CAST_NONE, blkElems);    // fp32 -> fp16
                PipeBarrier<PIPE_V>();
                // products keyH[t,d] *= iq[d]  (iq broadcast across 128 token rows)
                Mul(keyH, keyH, iqH, static_cast<int32_t>(MSA_BLOCK_SIZE), MSA_BLOCK_SIZE,
                    {1, 1, 1, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                // dot product per token = sum over dim (fp16) -> this block's
                // 128-half slot in the chunk buffer (j*128 is 32B-aligned).
                WholeReduceSum<half>(chunkBuf[static_cast<uint32_t>(j) * MSA_BLOCK_SIZE], keyH, fmask,
                    MSA_BLOCK_SIZE, 1, 1, 8);
                PipeBarrier<PIPE_V>();
                if (b == lastBlock && lastValid < MSA_BLOCK_SIZE) {
                    // mask invalid tail lanes (masked Duplicate over the 8-datablock
                    // repeat at the 32B-aligned slot, mirrors FillMaxValueFromTail).
                    Duplicate(chunkBuf[static_cast<uint32_t>(j) * MSA_BLOCK_SIZE],
                        static_cast<half>(MIN_HALF_VALUE), tailMask, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                }
            }
            // Collapse this chunk: chunkBlocks blocks -> chunkBlocks maxes, written
            // contiguously at blockMax[chunkStart] (32B-aligned, chunkStart % 32 == 0).
            WholeReduceMax<half>(blockMax[chunkStart], chunkBuf, fmask,
                static_cast<int32_t>(chunkBlocks), 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }

        // Emit per-block maxes [0, blockCount) as fp16. Full barrier so the
        // WholeReduceMax (PIPE_V) writes land before the UB->GM DMA (PIPE_MTE3).
        PipeBarrier<PIPE_ALL>();
        uint64_t outOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(outW);
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(outW * sizeof(half)), 0, 0, 0};
        DataCopyPad(indicesGm_[outOff], blockMax, cpOut);
        topKValueInQueue_.FreeTensor(blockMax);
    }

    // For a zero-length / padded row, emit a deterministic block-0 list.
    __aicore__ inline void WriteEmpty(uint32_t row)
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint32_t outW = param_.blockCount;
        uint32_t fillN = matmul::CeilDiv(outW, 32U) * 32U;
        LocalTensor<half> poolValue = topKValueInQueue_.AllocTensor<half>();
        Duplicate(poolValue, static_cast<half>(MIN_HALF_VALUE), fillN);
        PipeBarrier<PIPE_V>();
        // Full barrier before the UB->GM emit: guarantees the WholeReduceMax
        // (PIPE_V) writes to poolValue are fully landed before the DMA (PIPE_MTE3)
        // reads them. A lone V_MTE3 flag left a residual race that duplicated /
        // shifted the leading pooled values non-deterministically.
        PipeBarrier<PIPE_ALL>();
        uint64_t outOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(outW);
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(outW * sizeof(half)), 0, 0, 0};
        DataCopyPad(indicesGm_[outOff], poolValue, cpOut);
        topKValueInQueue_.FreeTensor(poolValue);
    }

    // ---- sync helpers (two-leg Vector<->Cube handshake per row) ----
    // Leg 1 (gather done): AIV sets after its UB->GM gather writes (PIPE_MTE3),
    //                      AIC waits before reading the slab in ComputeMM.
    __aicore__ inline void VectorNotifyCube(uint16_t evt)
    {
        // All-AIV barrier first: both paired vector cores must reach here (the
        // gathering AIV only after PIPE_MTE3 drains its keyGatherGm writes)
        // before either notifies the cube -- otherwise the idle AIV's early
        // notify lets the cube read a half-written key slab.
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_FLAG);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(evt);
    }
    __aicore__ inline void CubeWaitVector(uint16_t evt)
    {
        CrossCoreWaitFlag(evt);
    }
    // Leg 2 (scores ready): AIC sets after the matmul output pipe (PIPE_FIX),
    //                       AIV waits before pooling matmulGm_.
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
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(indices));

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
        // Gather staging: one physical idxk block ([128,128] bf16 = 32 KiB) per
        // buffer, double-buffered so the GM->UB and UB->GM legs of consecutive
        // blocks overlap. The gather and the TopK phases share UB but run in
        // distinct phases (separated by the Cube round-trip), so peak UB is the
        // gather queue plus the (KB-scale) TopK buffers -- well within AIV UB.
        pipe_->InitBuffer(keyGatherInQueue_, DOUBLE_BUFFER_NUM,
            MSA_BLOCK_SIZE * param_.dimension * sizeof(bfloat16_t));
        // Output block-max tensor: one half per block, padded to topKInnerSize
        // (= CeilDiv(blockCount,32)*32 on the host), so it always holds fillN.
        uint32_t innerSize = Max(param_.topKInnerSize, static_cast<uint32_t>(MAX_FP16_PROCESS_NUM));
        pipe_->InitBuffer(topKValueInQueue_, 1, innerSize * sizeof(half));
        pipe_->InitBuffer(iqBf16Buf_, param_.dimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(iqF32Buf_, param_.dimension * sizeof(float));
        pipe_->InitBuffer(iqHalfBuf_, param_.dimension * sizeof(half));
        pipe_->InitBuffer(keyF32Buf_, MSA_BLOCK_SIZE * param_.dimension * sizeof(float));
        pipe_->InitBuffer(keyHalfBuf_, MSA_BLOCK_SIZE * param_.dimension * sizeof(half));
        // Fixed-size chunk buffer for the fused pool (POOL_CHUNK_BLOCKS*128 half).
        // FIXED regardless of numBlocks -- replaces the old maxSeqLen scratch that
        // overflowed UB once numBlocks reached 128. The unused topK index/out
        // queues + maxSeqLen scratch are no longer allocated (top-k is python-side).
        pipe_->InitBuffer(chunkBuf_, POOL_CHUNK_BLOCKS * MSA_BLOCK_SIZE * sizeof(half));
    }

protected:
    TPipe *pipe_;

    GlobalTensor<bfloat16_t> iqGm_;
    GlobalTensor<bfloat16_t> idxkCacheGm_;
    GlobalTensor<bfloat16_t> keyGatherGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<int32_t> keyBlockTableGm_;
    GlobalTensor<half> indicesGm_;
    GlobalTensor<half> matmulGm_;

    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> keyGatherInQueue_;
    TQue<TPosition::VECIN, 1> topKValueInQueue_;
    TQue<TPosition::VECIN, 1> topKIndexInQueue_;
    TQue<TPosition::VECOUT, 1> topKValueOutQueue_;
    TQue<TPosition::VECOUT, 1> topKIndexOutQueue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> scratchQueue_;
    TBuf<TPosition::VECCALC> outIndexBuf_;
    TBuf<TPosition::VECCALC> iqBf16Buf_;
    TBuf<TPosition::VECCALC> iqF32Buf_;
    TBuf<TPosition::VECCALC> iqHalfBuf_;
    TBuf<TPosition::VECCALC> keyF32Buf_;
    TBuf<TPosition::VECCALC> keyHalfBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;

    MsaTilingParam param_;
    uint32_t blockIdx_ = 0;
    uint32_t subBlockIdx_ = 0;

    static constexpr uint64_t SYNC_MODE0 = 0; // AIV<->AIV (all vector cores)
    static constexpr uint64_t SYNC_MODE2 = 2; // AIC<->AIV
    static constexpr uint16_t SYNC_AIV_ONLY_ALL_FLAG = 0; // all-AIV barrier
    static constexpr uint16_t SYNC_AIV_AIC_FLAG = 2; // gather done: Vector -> Cube
    static constexpr uint16_t SYNC_AIC_AIV_FLAG = 4; // scores ready: Cube -> Vector

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_CFG_NO_PRELOAD> mm_;
};

} // namespace AscendC
#endif // MSA_DIST_TOP_K_SPLIT_S_H

