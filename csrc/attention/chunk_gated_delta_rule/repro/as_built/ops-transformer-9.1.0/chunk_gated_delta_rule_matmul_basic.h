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
 * \file chunk_gated_delta_rule_matmul_basic.h
 * \brief CGDR basic API matmul for arch22 (910B/v220):
 *        manually controls GM->L1->L0A/L0B->Mmad->L0C->GM pipeline.
 *        All matmuls in this operator fit a single 128x128 tile, so Execute is single-shot,
 *        with 2-deep software pipelining across calls (all on-chip buffers ping-pong by call
 *        parity; call i's top gate waits call i-2 on the same buffer set, which transitively
 *        protects L1/L0A/L0B/L0C).
 *        Adjacent calls with a GM read-after-write dependency must pass serial=true, which
 *        additionally waits the other set's Fixpipe (i.e. call i-1) before loading.
 */

#ifndef CHUNK_GATED_DELTA_RULE_MATMUL_BASIC_H
#define CHUNK_GATED_DELTA_RULE_MATMUL_BASIC_H

#include "kernel_operator.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

constexpr uint32_t MM_BLOCK_CUBE = 16;
constexpr uint32_t MM_B16_BLK = 16;       // b16: 32B fractal width / sizeof(b16)
constexpr uint32_t MM_FRACTAL_ELEM = 256; // 16x16 elements per 512B fractal
constexpr uint32_t MM_MAX_DIM = 128;
constexpr uint32_t MM_L1_ELEM_CNT = MM_MAX_DIM * MM_MAX_DIM;  // 16384 b16 = 32KB
constexpr uint32_t MM_SMALL_MN = 10;      // v220: (m/16)*(n/16) < 10 needs PIPE_M barrier after Mmad
// FIX_MTE2 ids 4/5 (id 6 is FIX_MTE2_EVENT in stage2, ids 0-3 kept clear);
// the three forward flags use ids 6/7 on their own channels (no other AIC users).
constexpr int32_t MM_BACK_ID[2] = {4, 5};
constexpr int32_t MM_FWD_ID[2] = {6, 7};

enum class BSource {
    Copy,
    SameAsA,
    Reuse
};

__aicore__ inline uint32_t MmAlign(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

class CGDRMatmulBasic {
public:
    __aicore__ inline CGDRMatmulBasic() {}

    // Re-init after every pipe Reset (per stage). AIC only.
    __aicore__ inline void Init(TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            return;
        }
        pipe->InitBuffer(l1Buf_, 4 * MM_L1_ELEM_CNT * sizeof(bfloat16_t));  // 128KB: (A+B) x2 sets
        pipe->InitBuffer(l0aBuf_, 2 * MM_L1_ELEM_CNT * sizeof(bfloat16_t)); // 64KB = full L0A
        pipe->InitBuffer(l0bBuf_, 2 * MM_L1_ELEM_CNT * sizeof(bfloat16_t)); // 64KB = full L0B
        pipe->InitBuffer(l0cBuf_, 2 * MM_L1_ELEM_CNT * sizeof(float));      // 128KB = full L0C
        for (int32_t p = 0; p < 2; p++) {
            l1aTensor_[p] = l1Buf_.GetWithOffset<bfloat16_t>(
                MM_L1_ELEM_CNT, (2 * p) * MM_L1_ELEM_CNT * sizeof(bfloat16_t));
            l1bTensor_[p] = l1Buf_.GetWithOffset<bfloat16_t>(
                MM_L1_ELEM_CNT, (2 * p + 1) * MM_L1_ELEM_CNT * sizeof(bfloat16_t));
            l0aTensor_[p] = l0aBuf_.GetWithOffset<bfloat16_t>(MM_L1_ELEM_CNT,
                                                              p * MM_L1_ELEM_CNT * sizeof(bfloat16_t));
            l0bTensor_[p] = l0bBuf_.GetWithOffset<bfloat16_t>(MM_L1_ELEM_CNT,
                                                              p * MM_L1_ELEM_CNT * sizeof(bfloat16_t));
            l0cTensor_[p] = l0cBuf_.GetWithOffset<float>(MM_L1_ELEM_CNT, p * MM_L1_ELEM_CNT * sizeof(float));
            SetFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[p]);
        }
        cnt_ = 0;
    }

    // Must be called before the owning pipe is Reset. AIC only.
    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            return;
        }
        WaitFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[0]);
        WaitFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[1]);
    }

    template <bool transA, bool transB, bool accum, typename dstType = bfloat16_t, BSource bSrc = BSource::Copy,
              bool serial = false>
    __aicore__ inline void Execute(GlobalTensor<bfloat16_t> aGm, GlobalTensor<bfloat16_t> bGm,
                                   GlobalTensor<dstType> cGm, uint32_t m, uint32_t n, uint32_t k,
                                   uint32_t aGmRowStride = 0, uint32_t bGmRowStride = 0, uint32_t cGmRowStride = 0)
    {
        if ASCEND_IS_AIV {
            return;
        }

        uint32_t p = cnt_ & 1;
        cnt_++;
        uint32_t aRows = transA ? k : m;
        uint32_t aCols = transA ? m : k;
        uint32_t bRows = transB ? n : k;
        uint32_t bCols = transB ? k : n;
        uint32_t aSrcD = (aGmRowStride == 0) ? aCols : aGmRowStride;
        uint32_t bSrcD = (bGmRowStride == 0) ? bCols : bGmRowStride;
        uint32_t cDstStride = (cGmRowStride == 0) ? n : cGmRowStride;

        // (1) GM -> L1 (nd2nz)
        if constexpr (serial) {
            // 依赖上一笔（对面 set）的 Fixpipe 写回：等它、再补回一枚保持配对
            WaitFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[1 - p]);
            SetFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[1 - p]);
        }
        WaitFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[p]);

        CopyGmToL1(aGm, aRows, aCols, aSrcD, l1aTensor_[p]);
        if constexpr (bSrc == BSource::Copy) {
            CopyGmToL1(bGm, bRows, bCols, bSrcD, l1bTensor_[p]);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MM_FWD_ID[p]);

        // (2) L1 -> L0
        WaitFlag<HardEvent::MTE2_MTE1>(MM_FWD_ID[p]);

        if constexpr (transA) {
            LoadAKm(l0aTensor_[p], l1aTensor_[p], MmAlign(aRows, MM_BLOCK_CUBE), MmAlign(m, MM_BLOCK_CUBE));
        } else {
            LoadAMk(l0aTensor_[p], l1aTensor_[p], MmAlign(m, MM_BLOCK_CUBE), MmAlign(k, MM_BLOCK_CUBE));
        }
        // Reuse 读的是上一笔（对面 set）拷入的 B；仅在上一笔与本笔及下一笔都 serial 时安全
        LocalTensor<bfloat16_t> &l1bUse = (bSrc == BSource::SameAsA)
                                              ? l1aTensor_[p]
                                              : ((bSrc == BSource::Reuse) ? l1bTensor_[1 - p] : l1bTensor_[p]);
        if constexpr (transB) {
            LoadBNk(l0bTensor_[p], l1bUse, MmAlign(n, MM_BLOCK_CUBE), MmAlign(k, MM_BLOCK_CUBE));
        } else {
            LoadBKn(l0bTensor_[p], l1bUse, MmAlign(k, MM_BLOCK_CUBE), MmAlign(n, MM_BLOCK_CUBE));
        }

        SetFlag<HardEvent::MTE1_M>(MM_FWD_ID[p]);

        // (3) Mmad
        WaitFlag<HardEvent::MTE1_M>(MM_FWD_ID[p]);

        DoMmad(l0cTensor_[p], l0aTensor_[p], l0bTensor_[p], m, n, k);

        SetFlag<HardEvent::M_FIX>(MM_FWD_ID[p]);

        // (4) L0C -> GM
        WaitFlag<HardEvent::M_FIX>(MM_FWD_ID[p]);

        CopyL0CToGm<dstType, accum>(cGm, l0cTensor_[p], m, n, cDstStride);

        SetFlag<HardEvent::FIX_MTE2>(MM_BACK_ID[p]);
    }

private:
    TBuf<TPosition::A1> l1Buf_;
    TBuf<TPosition::A2> l0aBuf_;
    TBuf<TPosition::B2> l0bBuf_;
    TBuf<TPosition::CO1> l0cBuf_;
    LocalTensor<bfloat16_t> l1aTensor_[2];
    LocalTensor<bfloat16_t> l1bTensor_[2];
    LocalTensor<bfloat16_t> l0aTensor_[2];
    LocalTensor<bfloat16_t> l0bTensor_[2];
    LocalTensor<float> l0cTensor_[2];
    uint32_t cnt_ = 0;

    __aicore__ inline void CopyGmToL1(GlobalTensor<bfloat16_t> &gm, uint32_t rows, uint32_t cols, uint32_t srcDStride,
                                      LocalTensor<bfloat16_t> &l1Tensor)
    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = rows;
        nd2nz.dValue = cols;
        nd2nz.srcDValue = srcDStride;
        nd2nz.dstNzC0Stride = MmAlign(rows, MM_BLOCK_CUBE);
        nd2nz.dstNzNStride = 1;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, gm, nd2nz);
    }

    // A stored (m, k) in L1 as Nz with mAlign rows: plain 2D loads, one per 16-col k block.
    __aicore__ inline void LoadAMk(LocalTensor<bfloat16_t> &l0Tensor, LocalTensor<bfloat16_t> &l1Tensor,
                                   uint32_t mAlign, uint32_t kAlign)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.srcStride = 1;
        params.dstGap = kAlign / MM_B16_BLK - 1;
        params.repeatTimes = static_cast<uint8_t>(mAlign / MM_BLOCK_CUBE);
        params.ifTranspose = false;
        uint32_t loopTimes = kAlign / MM_B16_BLK;
        uint64_t l1Offset = mAlign * MM_B16_BLK;
        uint64_t l0Offset = MM_FRACTAL_ELEM;
        for (uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadData(l0Tensor[loop * l0Offset], l1Tensor[loop * l1Offset], params);
        }
    }

    // A stored (k, m) in L1 as Nz with kAlign rows: transpose loads into L0A.
    // L0A layout = m-major with k fractals contiguous (same as LoadAMk); per m column-block the
    // k/16 source fractals are consecutive (srcStride=1) and written out in order (dstGap=0).
    __aicore__ inline void LoadAKm(LocalTensor<bfloat16_t> &l0Tensor, LocalTensor<bfloat16_t> &l1Tensor,
                                   uint32_t kAlign, uint32_t mAlign)
    {
        LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.srcStride = 1;
        params.dstFracGap = 0;
        params.dstGap = 0;
        params.repeatTimes = static_cast<uint8_t>(kAlign / MM_B16_BLK);
        uint32_t loopTimes = mAlign / MM_B16_BLK;
        uint64_t l1Offset = kAlign * MM_B16_BLK;
        uint64_t l0Offset = kAlign * MM_B16_BLK;
        for (uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadDataWithTranspose(l0Tensor[loop * l0Offset], l1Tensor[loop * l1Offset], params);
        }
    }

    // B stored (k, n) in L1 as Nz with kAlign rows: transpose loads into L0B (Zn layout).
    __aicore__ inline void LoadBKn(LocalTensor<bfloat16_t> &l0Tensor, LocalTensor<bfloat16_t> &l1Tensor,
                                   uint32_t kAlign, uint32_t nAlign)
    {
        LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.srcStride = 1;
        params.dstFracGap = 0;
        params.dstGap = static_cast<uint16_t>(nAlign / MM_BLOCK_CUBE - 1);
        params.repeatTimes = static_cast<uint8_t>(kAlign / MM_B16_BLK);
        uint32_t loopTimes = nAlign / MM_B16_BLK;
        uint64_t l1Offset = kAlign * MM_B16_BLK;
        uint64_t l0Offset = MM_FRACTAL_ELEM;
        for (uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadDataWithTranspose(l0Tensor[loop * l0Offset], l1Tensor[loop * l1Offset], params);
        }
    }

    // B stored (n, k) in L1 as Nz with nAlign rows: plain loads (already K-major for L0B).
    __aicore__ inline void LoadBNk(LocalTensor<bfloat16_t> &l0Tensor, LocalTensor<bfloat16_t> &l1Tensor,
                                   uint32_t nAlign, uint32_t kAlign)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;
        if (nAlign == kAlign) {
            params.repeatTimes = static_cast<uint8_t>((nAlign / MM_BLOCK_CUBE) * (kAlign / MM_B16_BLK));
            LoadData(l0Tensor, l1Tensor, params);
        } else {
            params.repeatTimes = static_cast<uint8_t>(nAlign / MM_BLOCK_CUBE);
            uint32_t loopTimes = kAlign / MM_B16_BLK;
            uint64_t l1Offset = nAlign * MM_B16_BLK;
            uint64_t l0Offset = nAlign * MM_B16_BLK;
            for (uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadData(l0Tensor[loop * l0Offset], l1Tensor[loop * l1Offset], params);
            }
        }
    }

    __aicore__ inline void DoMmad(LocalTensor<float> &l0cTensor, LocalTensor<bfloat16_t> &l0aTensor,
                                  LocalTensor<bfloat16_t> &l0bTensor, uint32_t m, uint32_t n, uint32_t k)
    {
        MmadParams mmadParams;
        mmadParams.m = static_cast<uint16_t>((m < MM_BLOCK_CUBE) ? MM_BLOCK_CUBE : m);
        mmadParams.n = static_cast<uint16_t>(n);
        mmadParams.k = static_cast<uint16_t>(k);
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);
        if ((mmadParams.m / MM_BLOCK_CUBE) * (mmadParams.n / MM_BLOCK_CUBE) < MM_SMALL_MN) {
            PipeBarrier<PIPE_M>();
        }
    }

    template <typename dstType, bool accum>
    __aicore__ inline void CopyL0CToGm(GlobalTensor<dstType> &gm, LocalTensor<float> &l0cTensor, uint32_t m, uint32_t n,
                                       uint32_t dstStride)
    {
        FixpipeParamsV220 fixParams;
        fixParams.mSize = static_cast<uint16_t>(m);
        fixParams.nSize = static_cast<uint16_t>(n);
        fixParams.srcStride = static_cast<uint16_t>(MmAlign(m, MM_BLOCK_CUBE));
        fixParams.dstStride = dstStride;
        fixParams.ndNum = 1;
        fixParams.unitFlag = 0;

        if constexpr (std::is_same<dstType, bfloat16_t>::value) {
            fixParams.quantPre = QuantMode_t::F322BF16;
        }

        if constexpr (accum) {
            SetAtomicAdd<dstType>();
        }
        Fixpipe(gm, l0cTensor, fixParams);
        if constexpr (accum) {
            SetAtomicNone();
        }
    }
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_MATMUL_BASIC_H
