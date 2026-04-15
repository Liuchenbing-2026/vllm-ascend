/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "ngram_match_extract_tiling.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

#include <algorithm>

namespace optiling {

static void GetCompileParameters(gert::TilingContext* context, uint32_t& coreNum)
{
    auto ptrCompileInfo = reinterpret_cast<const NgramMatchExtractCompileInfo*>(
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
    OPS_LOG_I(context, "Enter TilingFunc for NgramMatchExtract");

    uint32_t coreNum;
    GetCompileParameters(context, coreNum);

    // token_ids: input 0, shape [B, max_len]
    auto tokenIdsShape = context->GetInputShape(0);
    uint32_t numReqs = 0;
    uint32_t maxSeqLen = 0;
    if (tokenIdsShape != nullptr && tokenIdsShape->GetStorageShape().GetDimNum() >= 2) {
        numReqs = static_cast<uint32_t>(tokenIdsShape->GetStorageShape().GetDim(0));
        maxSeqLen = static_cast<uint32_t>(tokenIdsShape->GetStorageShape().GetDim(1));
    }

    // Attributes: min_n, max_n, k
    auto attrs = context->GetAttrs();
    int32_t minN = *(attrs->GetAttrPointer<int32_t>(0));
    int32_t maxN = *(attrs->GetAttrPointer<int32_t>(1));
    int32_t k = *(attrs->GetAttrPointer<int32_t>(2));

    // Core distribution
    uint32_t usedCoreNum = std::min(coreNum, numReqs);
    if (usedCoreNum == 0) usedCoreNum = 1;
    uint32_t reqsPerCore = numReqs / usedCoreNum;
    uint32_t remainderReqs = numReqs % usedCoreNum;

    context->SetTilingKey(1);

    NgramMatchExtractTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numReqs(numReqs);
    tiling.set_reqsPerCore(reqsPerCore);
    tiling.set_remainderReqs(remainderReqs);
    tiling.set_maxSeqLen(maxSeqLen);
    tiling.set_minN(static_cast<uint32_t>(minN));
    tiling.set_maxN(static_cast<uint32_t>(maxN));
    tiling.set_k(static_cast<uint32_t>(k));

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(usedCoreNum);

    OPS_LOG_I(context, "numReqs=%u reqsPerCore=%u remainder=%u maxSeqLen=%u minN=%d maxN=%d k=%d",
              numReqs, reqsPerCore, remainderReqs, maxSeqLen, minN, maxN, k);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NgramMatchExtractCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NgramMatchExtract)
    .Tiling(TilingFunc)
    .TilingParse<NgramMatchExtractCompileInfo>(TilingPrepare);

}  // namespace optiling
