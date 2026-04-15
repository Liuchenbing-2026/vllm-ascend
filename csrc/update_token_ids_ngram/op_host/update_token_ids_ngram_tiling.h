/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef UPDATE_TOKEN_IDS_NGRAM_TILING_H
#define UPDATE_TOKEN_IDS_NGRAM_TILING_H

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UpdateTokenIdsNgramTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, reqsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderReqs);
    TILING_DATA_FIELD_DEF(uint32_t, maxNewTokens);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
    TILING_DATA_FIELD_DEF(int32_t, vocabSize);
END_TILING_DATA_DEF;

struct UpdateTokenIdsNgramCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(UpdateTokenIdsNgram, UpdateTokenIdsNgramTilingData)

}  // namespace optiling

#endif  // UPDATE_TOKEN_IDS_NGRAM_TILING_H
