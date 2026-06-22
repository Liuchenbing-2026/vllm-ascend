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
 * \file msa_dist_top_k_base.h
 * \brief Shared constants / helpers for the MiniMax-M3 MSA block-selection op.
 *
 * Adapted from hamming_dist_top_k_base.h. All hamming/popcount/int4b_t/Select
 * machinery has been deleted. The only retained pieces are:
 *   - the ReduceMax block max-pool helper (chunkSize==128 path only),
 *   - the AscendC TopK wrapper,
 *   - the int32 ascending insertion sort used to canonicalise block ids,
 *   - the FillMaxValueFromTail helper used to force-keep the local block,
 *   - the MAX/MIN half sentinels.
 */

#ifndef MSA_DIST_TOP_K_BASE_H
#define MSA_DIST_TOP_K_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace AscendC {

constexpr uint32_t MAX_FP16_PROCESS_NUM = 128;
constexpr uint32_t MAX_INT32_PROCESS_NUM = 64;
constexpr float MIN_HALF_VALUE = -65535;
constexpr half MAX_HALF_VALUE = (half)65504;

// datablock bytes = 32bytes
constexpr uint32_t DATABLOCK_BYTES = 32;
// half 4 datablocks element size
constexpr uint32_t FOUR_DATABLOCKS_ELEMENT_SIZE = 64;
// half 8 datablocks element size
constexpr uint32_t EIGHT_DATABLOCKS_ELEMENT_SIZE = 128;

// MSA decode constants (mirror get_msa_decode_params on the python side).
constexpr uint32_t MSA_BLOCK_SIZE = 128;   // block_size / chunk_size
constexpr uint32_t MSA_DIM = 128;          // index head dim (iq / ik)
constexpr float MSA_SCALE = 0.08838834764831845f; // 1 / sqrt(128)

// Cube matmul config: no preload, single-shot iterate, bias enabled flag last.
constexpr MatmulConfig MM_CFG_NO_PRELOAD{false, false, true, 0, 0, 0, false, false, false, false, false,
                                         0, 0, 0, 0, 0, 0, 0, true};

// Flattened tiling fields the host drafter emits (mirrors hamming minus
// rope/offload/sink). Field names intentionally generic.
struct MsaTilingParam {
    uint32_t usedCoreNum = 0;
    uint32_t M = 0;          // matmul M (= G = 1 for decode)
    uint32_t N = 0;
    uint32_t ka = 0;         // = dimension = 128
    uint32_t kb = 0;         // = dimension = 128
    uint32_t batch = 0;      // B decode sequences
    uint32_t head = 0;       // G (kv group), = 1 for decode
    uint32_t batchN = 0;     // batch * head
    uint32_t dimension = 0;  // 128
    uint32_t maxSeqLen = 0;  // padded max KV length (multiple of 128)
    uint32_t maxK = 0;       // topk_blocks = 16
    uint32_t blockCount = 0; // MAXB: logical blocks per seq in block_table
    uint32_t topkTotal = 0;  // topk_blocks + local_blocks + init_blocks = 17
    uint32_t localBlocks = 0; // recent = 1
    uint32_t initBlocks = 0;  // sink = 0
    uint32_t topKInnerSize = 0;
    uint32_t matmulResultSize = 0;
};

template <typename T>
__aicore__ inline T Min(const T a, const T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline T Max(const T a, const T b)
{
    return a > b ? a : b;
}

// ----------------------------------------------------------------------------
// TopK wrapper. n is the number of valid candidate scores (chunk count), k is
// the number of winners to keep. The TilingData type is templated so the host
// struct name can change without touching this header.
// ----------------------------------------------------------------------------
template <typename TilingT>
__aicore__ inline void TopKCustom(const LocalTensor<half> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<half> &srcValueLocal, const LocalTensor<int32_t> &srcIndexLocal, const int32_t k,
    const TilingT &tiling, uint32_t n)
{
    LocalTensor<bool> finishLocal;
    AscendC::TopKInfo topkInfo;
    topkInfo.outter = tiling.params.outer;
    topkInfo.n = n;
    topkInfo.inner = matmul::CeilDiv(n, 32) * 32; /* 32: inner must be aligned to 32 */
    TopK<half, true, false, false, TopKMode::TOPK_NORMAL>(dstValueLocal, dstIndexLocal, srcValueLocal, srcIndexLocal,
        finishLocal, k, tiling.topkTiling, topkInfo, true);
}

// ----------------------------------------------------------------------------
// Block max-pool: matmul score row in GM -> one max per 128-element block.
// MSA only ever uses chunkSize == MSA_BLOCK_SIZE (128), so the BlockReduceMax /
// chunk 1/8/16/64 branches of the hamming op are deleted.
//   reduceInputLocal : scratch UB (>= ceilAligned scores)
//   reduceOutputLocal: max values (chunkNum entries)
//   chunkNum         : number of 128-elem blocks to pool
// ----------------------------------------------------------------------------
__aicore__ inline void ReduceMaxBlock128(const GlobalTensor<half> &inputGm, const LocalTensor<half> &reduceInputLocal,
    const LocalTensor<half> &reduceOutputLocal, const uint16_t chunkNum)
{
    constexpr uint8_t chunkSize = MSA_BLOCK_SIZE; // 128
    uint32_t dataBlockNum = (static_cast<uint32_t>(chunkNum) * static_cast<uint32_t>(chunkSize) + 15) / 16; // 16 half / 32B block
    uint32_t blockLen = static_cast<uint32_t>(16 * sizeof(half));

    // chunkSize==128: copy one contiguous block.
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(dataBlockNum * blockLen), 0, 0, 0};
    DataCopyPadExtParams<half> copyInPadParams{false, 0, 0, 0};
    DataCopyPad(reduceInputLocal, inputGm, copyInParams, copyInPadParams);

    // Pad tail to a multiple of 8 datablocks (one WholeReduceMax repeat = 128 half).
    uint32_t dataBlockNumAligned = matmul::CeilDiv(dataBlockNum, 8) * 8;
    if (dataBlockNumAligned > dataBlockNum) {
        Duplicate(reduceInputLocal[dataBlockNum * 16], static_cast<half>(MIN_HALF_VALUE),
            (dataBlockNumAligned - dataBlockNum) * 16);
    }

    {
        int32_t evtM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evtM2V);
        WaitFlag<HardEvent::MTE2_V>(evtM2V);
    }
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_ALL>();

    int32_t totalRepeat = dataBlockNumAligned / 8;            // 1 repeat == 128 half == 1 block
    int32_t repeat = Min(MAX_REPEAT_TIMES, totalRepeat);
    int32_t loopNum = matmul::CeilDiv(totalRepeat, repeat);
    int32_t tailRepeat = totalRepeat - (loopNum - 1) * repeat;
    uint64_t mask[2];
    mask[0] = UINT64_MAX;
    mask[1] = UINT64_MAX; // all 128 lanes participate

    uint32_t srcOffset = 0;
    uint32_t dstOffset = 0;
    for (int32_t i = 0; i < loopNum - 1; i++) {
        WholeReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], mask, repeat, 1, 1, 8,
            ReduceOrder::ORDER_ONLY_VALUE);
        srcOffset += repeat * 8 * 16; // repeat * 128 half
        dstOffset += repeat;          // 1 max per repeat
    }
    WholeReduceMax<half>(reduceOutputLocal[dstOffset], reduceInputLocal[srcOffset], mask, tailRepeat, 1, 1, 8,
        ReduceOrder::ORDER_ONLY_VALUE);
}

// In-place ascending insertion sort of int32 logical block ids in UB.
__aicore__ inline void SortInt32AscendingUB(LocalTensor<int32_t> &buf, uint32_t len)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (len <= 1) {
        return;
    }
    __ubuf__ int32_t *data = reinterpret_cast<__ubuf__ int32_t *>(buf.GetPhyAddr());
    for (uint32_t i = 1; i < len; ++i) {
        int32_t key = data[i];
        int32_t j = static_cast<int32_t>(i) - 1;
        while (j >= 0 && data[j] > key) {
            data[j + 1] = data[j];
            --j;
        }
        data[j + 1] = key;
    }
}

// Force the last copyLen pooled scores to MAX_HALF_VALUE so TopK keeps them.
// MSA uses copyLen = localBlocks = 1 to force-keep the local block.
__aicore__ inline void FillMaxValueFromTail(
    LocalTensor<half> &topKValueInTensor, uint32_t tensorSize, uint32_t copyLen)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (copyLen == 0 || copyLen > tensorSize) {
        return;
    }
    uint32_t alignedElements = DATABLOCK_BYTES / sizeof(half);
    uint32_t offset = tensorSize - copyLen;
    if (offset % alignedElements == 0) {
        Duplicate(topKValueInTensor[offset], static_cast<half>(MAX_HALF_VALUE), copyLen);
        return;
    }

    uint32_t offsetAligned = offset / alignedElements * alignedElements;
    uint32_t alignedAddCopyElements = tensorSize - offsetAligned;
    uint64_t mask[2] = {0, 0};
    uint32_t needSkipElements = alignedAddCopyElements - copyLen;
    int32_t lastCopyLen = alignedAddCopyElements - EIGHT_DATABLOCKS_ELEMENT_SIZE;
    if (lastCopyLen > 0) {
        alignedAddCopyElements = EIGHT_DATABLOCKS_ELEMENT_SIZE;
    }
    if (alignedAddCopyElements <= FOUR_DATABLOCKS_ELEMENT_SIZE) {
        mask[0] = (UINT64_MAX << needSkipElements) & (UINT64_MAX >> (FOUR_DATABLOCKS_ELEMENT_SIZE - alignedAddCopyElements));
    } else if (alignedAddCopyElements <= EIGHT_DATABLOCKS_ELEMENT_SIZE) {
        mask[0] = (UINT64_MAX << needSkipElements);
        mask[1] = UINT64_MAX >> (EIGHT_DATABLOCKS_ELEMENT_SIZE - alignedAddCopyElements);
    }
    Duplicate(topKValueInTensor[offsetAligned], static_cast<half>(MAX_HALF_VALUE), mask, 1, 1, 8);
    if (lastCopyLen > 0) {
        Duplicate(topKValueInTensor[offsetAligned + EIGHT_DATABLOCKS_ELEMENT_SIZE], static_cast<half>(MAX_HALF_VALUE),
            lastCopyLen);
    }
}

} // namespace AscendC
#endif // MSA_DIST_TOP_K_BASE_H

