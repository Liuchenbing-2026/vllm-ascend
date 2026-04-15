/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "update_token_ids_ngram_tiling.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

#include <algorithm>

namespace optiling {

static void GetCompileParameters(gert::TilingContext* context, uint32_t& coreNum)
{
    auto ptrCompileInfo = reinterpret_cast<const UpdateTokenIdsNgramCompileInfo*>(
        context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = ascendcPlatform.GetCoreNum();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_I(context, "Enter TilingFunc for UpdateTokenIdsNgram");

    uint32_t coreNum;
    GetCompileParameters(context, coreNum);

    // sampled_token_ids: input 0, shape [B, max_new]
    auto sampledShape = context->GetInputShape(0);
    uint32_t numReqs = 0;
    uint32_t maxNewTokens = 0;
    if (sampledShape != nullptr && sampledShape->GetStorageShape().GetDimNum() >= 2) {
        numReqs = static_cast<uint32_t>(sampledShape->GetStorageShape().GetDim(0));
        maxNewTokens = static_cast<uint32_t>(sampledShape->GetStorageShape().GetDim(1));
    }

    // token_ids_gpu: input 1, shape [B, max_len]
    auto tokenIdsShape = context->GetInputShape(1);
    uint32_t maxSeqLen = 0;
    if (tokenIdsShape != nullptr && tokenIdsShape->GetStorageShape().GetDimNum() >= 2) {
        maxSeqLen = static_cast<uint32_t>(tokenIdsShape->GetStorageShape().GetDim(1));
    }

    // Attribute: vocab_size
    auto attrs = context->GetAttrs();
    int32_t vocabSize = *(attrs->GetAttrPointer<int32_t>(0));

    // Core distribution
    uint32_t usedCoreNum = std::min(coreNum, numReqs);
    if (usedCoreNum == 0) usedCoreNum = 1;
    uint32_t reqsPerCore = numReqs / usedCoreNum;
    uint32_t remainderReqs = numReqs % usedCoreNum;

    context->SetTilingKey(1);

    UpdateTokenIdsNgramTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numReqs(numReqs);
    tiling.set_reqsPerCore(reqsPerCore);
    tiling.set_remainderReqs(remainderReqs);
    tiling.set_maxNewTokens(maxNewTokens);
    tiling.set_maxSeqLen(maxSeqLen);
    tiling.set_vocabSize(vocabSize);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(usedCoreNum);

    OPS_LOG_I(context, "numReqs=%u reqsPerCore=%u remainder=%u maxNew=%u maxSeqLen=%u vocabSize=%d",
              numReqs, reqsPerCore, remainderReqs, maxNewTokens, maxSeqLen, vocabSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<UpdateTokenIdsNgramCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpdateTokenIdsNgram)
    .Tiling(TilingFunc)
    .TilingParse<UpdateTokenIdsNgramCompileInfo>(TilingPrepare);

}  // namespace optiling
