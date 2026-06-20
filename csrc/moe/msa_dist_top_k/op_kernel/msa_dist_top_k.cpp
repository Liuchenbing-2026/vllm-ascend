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
 * \file msa_dist_top_k.cpp
 * \brief Entry point for the MiniMax-M3 MSA decode block-selection op.
 *
 * MIX_AIC_1_2 (1 Cube : 2 Vector). Inputs (decode, per TP rank, G=1):
 *   iq           bf16  [B, G=1, 128]   index query (1 token / seq)
 *   idxkCache    bf16  [num_blocks, 128, 128]  paged index-key cache
 *   seqLen       int32 [B]
 *   keyBlockTable int32 [B, MAXB]       logical -> physical
 * Output:
 *   indices      int32 [B, G, topk_total]  selected logical block ids,
 *                ascending with the local block LAST.
 *
 * Single tiling-key path is implemented (key 10, the sequential "split_s"
 * clone). Tiling key 1 (parallel) is stubbed to the same kernel for v1.
 */

#include "msa_dist_top_k_split_s.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void msa_dist_top_k(GM_ADDR iq, GM_ADDR idxkCache, GM_ADDR seqLen,
        GM_ADDR keyBlockTable, GM_ADDR indicesIn, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)indicesIn;  // ABI alias of `indices` (in-place persistent out buffer); kernel writes via `indices`.
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(10)) {
        MsaDistTopKSplitSKernel op;
        op.Init(iq, idxkCache, seqLen, keyBlockTable, indices, user1, tilingData, &tPipe);
        op.Process(tilingData);
        tPipe.Destroy();
    } else if (TILING_KEY_IS(1)) {
        // v1: parallel decode path reuses the sequential kernel (correctness
        // first; a dedicated batch-parallel variant can replace this later).
        MsaDistTopKSplitSKernel op;
        op.Init(iq, idxkCache, seqLen, keyBlockTable, indices, user1, tilingData, &tPipe);
        op.Process(tilingData);
        tPipe.Destroy();
    }
}

