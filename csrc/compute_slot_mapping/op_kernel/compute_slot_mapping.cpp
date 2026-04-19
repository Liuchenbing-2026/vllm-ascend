/**
 * ComputeSlotMapping AscendC Kernel
 *
 * Computes slot_mapping for each token:
 *   block_table_idx = req_indices[i] * block_table_stride + positions[i] / block_size
 *   block_number = block_table[block_table_idx]
 *   slot_mapping[i] = block_number * block_size + positions[i] % block_size
 *
 * Multi-core: tokens distributed across cores, each core processes its range.
 * Block table gather uses GlobalTensor scalar access (random access pattern).
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t TILE_SIZE = 256;

class ComputeSlotMappingKernel {
public:
    __aicore__ inline ComputeSlotMappingKernel() {}

    __aicore__ inline void Init(GM_ADDR reqIndices, GM_ADDR positions,
                                GM_ADDR blockTable, GM_ADDR slotMapping,
                                const ComputeSlotMappingTilingData* tilingData)
    {
        numTokens_ = tilingData->numTokens;
        blockSize_ = tilingData->blockSize;
        blockTableStride_ = tilingData->blockTableStride;
        cpSize_ = tilingData->cpSize;
        cpRank_ = tilingData->cpRank;
        cpInterleave_ = tilingData->cpInterleave;
        blockTableSize_ = tilingData->blockTableSize;

        uint32_t coreId = GetBlockIdx();
        uint32_t tokensPerCore = tilingData->tokensPerCore;
        uint32_t remainderTokens = tilingData->remainderTokens;

        if (coreId < remainderTokens) {
            myStart_ = coreId * (tokensPerCore + 1);
            myCount_ = tokensPerCore + 1;
        } else {
            myStart_ = remainderTokens * (tokensPerCore + 1)
                     + (coreId - remainderTokens) * tokensPerCore;
            myCount_ = tokensPerCore;
        }

        gmReqIndices_.SetGlobalBuffer((__gm__ int32_t*)reqIndices, numTokens_);
        gmPositions_.SetGlobalBuffer((__gm__ int32_t*)positions, numTokens_);
        gmBlockTable_.SetGlobalBuffer((__gm__ int32_t*)blockTable, blockTableSize_);
        gmSlotMapping_.SetGlobalBuffer((__gm__ int32_t*)slotMapping, numTokens_);

        // Allocate UB buffers for tiled processing
        uint32_t alignedTile = AlignUp(TILE_SIZE * sizeof(int32_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(reqIdxBuf_, alignedTile);
        pipe_.InitBuffer(posBuf_, alignedTile);
        pipe_.InitBuffer(slotBuf_, alignedTile);
    }

    // Common path: cp_size == 1
    __aicore__ inline void Process()
    {
        for (uint32_t off = 0; off < myCount_; off += TILE_SIZE) {
            uint32_t tileLen = (myCount_ - off < TILE_SIZE) ? (myCount_ - off) : TILE_SIZE;
            uint32_t gmOff = myStart_ + off;

            // Load req_indices and positions into UB
            LocalTensor<int32_t> localReq = reqIdxBuf_.Get<int32_t>();
            LocalTensor<int32_t> localPos = posBuf_.Get<int32_t>();
            DataCopyIn(localReq, gmReqIndices_, (int32_t)gmOff, (int32_t)tileLen);
            DataCopyIn(localPos, gmPositions_, (int32_t)gmOff, (int32_t)tileLen);

            // Compute slot mapping per element
            LocalTensor<int32_t> localSlot = slotBuf_.Get<int32_t>();
            for (uint32_t j = 0; j < tileLen; j++) {
                int32_t reqIdx = localReq.GetValue(j);
                int32_t pos = localPos.GetValue(j);

                // Integer division without modulo: blockIdx = pos / blockSize
                int32_t blockIdx = pos / blockSize_;
                int32_t blockTableIdx = reqIdx * blockTableStride_ + blockIdx;
                int32_t blockNumber = gmBlockTable_.GetValue(blockTableIdx);

                // Block offset = pos - blockIdx * blockSize (avoids modulo)
                int32_t blockOffset = pos - blockIdx * blockSize_;
                localSlot.SetValue(j, blockNumber * blockSize_ + blockOffset);
            }

            // Write back
            DataCopyOut(localSlot, gmOff, tileLen);
        }
    }

    // CP path: cp_size > 1
    __aicore__ inline void ProcessCP()
    {
        int32_t virtualBlockSize = blockSize_ * cpSize_;

        for (uint32_t off = 0; off < myCount_; off += TILE_SIZE) {
            uint32_t tileLen = (myCount_ - off < TILE_SIZE) ? (myCount_ - off) : TILE_SIZE;
            uint32_t gmOff = myStart_ + off;

            LocalTensor<int32_t> localReq = reqIdxBuf_.Get<int32_t>();
            LocalTensor<int32_t> localPos = posBuf_.Get<int32_t>();
            DataCopyIn(localReq, gmReqIndices_, (int32_t)gmOff, (int32_t)tileLen);
            DataCopyIn(localPos, gmPositions_, (int32_t)gmOff, (int32_t)tileLen);

            LocalTensor<int32_t> localSlot = slotBuf_.Get<int32_t>();
            for (uint32_t j = 0; j < tileLen; j++) {
                int32_t reqIdx = localReq.GetValue(j);
                int32_t pos = localPos.GetValue(j);

                int32_t blockIdx = pos / virtualBlockSize;
                int32_t blockTableIdx = reqIdx * blockTableStride_ + blockIdx;
                int32_t blockNumber = gmBlockTable_.GetValue(blockTableIdx);

                int32_t virtualOffset = pos - blockIdx * virtualBlockSize;
                int32_t isLocal = (virtualOffset / cpInterleave_) % cpSize_ == cpRank_;

                if (isLocal) {
                    int32_t localOffset =
                        (virtualOffset / (cpSize_ * cpInterleave_)) * cpInterleave_
                        + virtualOffset % cpInterleave_;
                    localSlot.SetValue(j, blockNumber * blockSize_ + localOffset);
                } else {
                    localSlot.SetValue(j, -1);
                }
            }

            DataCopyOut(localSlot, gmOff, tileLen);
        }
    }

private:
    static __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
    {
        return (x + a - 1) / a * a;
    }

    __aicore__ inline void DataCopyIn(LocalTensor<int32_t>& dst,
                                       GlobalTensor<int32_t>& src,
                                       int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int32_t);
        int32_t aligned = (count + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
        DataCopy(dst, src[gmOffset], aligned);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void DataCopyOut(LocalTensor<int32_t>& src,
                                        uint32_t gmOffset, uint32_t count)
    {
        if (count == 0) return;
        uint32_t totalBytes = count * static_cast<uint32_t>(sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
        DataCopyPad(gmSlotMapping_[gmOffset], src, DataCopyExtParams(1, totalBytes, 0, 0, 0));
        pipe_barrier(PIPE_ALL);
    }

private:
    GlobalTensor<int32_t> gmReqIndices_, gmPositions_, gmBlockTable_, gmSlotMapping_;
    uint32_t numTokens_, blockTableSize_;
    int32_t blockSize_, blockTableStride_;
    int32_t cpSize_, cpRank_, cpInterleave_;
    uint32_t myStart_, myCount_;

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> reqIdxBuf_, posBuf_, slotBuf_;
};

extern "C" __global__ __aicore__ void compute_slot_mapping(
    GM_ADDR reqIndices, GM_ADDR positions,
    GM_ADDR blockTable, GM_ADDR slotMapping,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    ComputeSlotMappingKernel op;
    op.Init(reqIndices, positions, blockTable, slotMapping, &tilingData);

    if (TILING_KEY_IS(0)) {
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        op.ProcessCP();
    }
}
