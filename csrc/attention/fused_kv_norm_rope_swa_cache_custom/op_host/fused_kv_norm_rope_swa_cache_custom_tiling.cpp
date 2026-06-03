/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

/*!
 * \file fused_kv_norm_rope_swa_cache_custom_tiling.cpp
 * \brief task=token tiling for the DSA SWA-kv prep fusion.
 */
#include "fused_kv_norm_rope_swa_cache_custom_tiling.h"

namespace optiling {

static ge::graphStatus TilingFuncFusedKvNormRopeSwa(gert::TilingContext* ctx)
{
    FusedKvNormRopeSwaCacheCustomTilingData td;

    auto kvShape    = ctx->GetInputShape(0);   // [num_tokens, head_dim]
    auto cosShape   = ctx->GetInputShape(2);   // [num_tokens, rope_dim]
    auto cacheShape = ctx->GetInputShape(5);   // [num_blocks, block_size, head_dim]
    if (kvShape == nullptr || cosShape == nullptr || cacheShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t nt        = static_cast<uint32_t>(kvShape->GetStorageShape().GetDim(0));
    uint32_t headDim   = static_cast<uint32_t>(kvShape->GetStorageShape().GetDim(1));
    uint32_t ropeDim   = static_cast<uint32_t>(cosShape->GetStorageShape().GetDim(1));
    uint32_t numBlocks = static_cast<uint32_t>(cacheShape->GetStorageShape().GetDim(0));
    uint32_t blockSize = static_cast<uint32_t>(cacheShape->GetStorageShape().GetDim(1));

    if (ropeDim == 0 || headDim < ropeDim || blockSize == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t nopeDim = headDim - ropeDim;

    auto attrs = ctx->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const float* epsPtr = attrs->GetFloat(0);
    float eps = (epsPtr != nullptr) ? *epsPtr : 1e-6f;

    // Platform info is not always available in the Tiling phase; prefer the parsed CompileInfo,
    // fall back to GetPlatformInfo (mirrors add_rms_norm_bias_tiling.cpp:211-218).
    uint32_t aivNum = 0;
    auto ptrCompileInfo = reinterpret_cast<const FusedKvNormRopeSwaCacheCustomCompileInfo*>(ctx->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto plat = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
        aivNum = plat.GetCoreNumAiv();
    } else {
        aivNum = ptrCompileInfo->totalCoreNum;
    }
    if (aivNum == 0) aivNum = 40;

    // Always use all AIV cores for maximal parallelism.
    uint32_t usedCoreNum = aivNum;
    if (usedCoreNum == 0) usedCoreNum = 1;

    td.set_numTokens(nt);
    td.set_headDim(headDim);
    td.set_nopeDim(nopeDim);
    td.set_ropeDim(ropeDim);
    td.set_blockSize(blockSize);
    td.set_numBlocks(numBlocks);
    td.set_eps(eps);
    td.set_headDimF(static_cast<float>(headDim));

    uint32_t isBf16 = 0;
    auto kvDesc = ctx->GetInputDesc(0);
    if (kvDesc != nullptr && kvDesc->GetDataType() == ge::DT_BF16) {
        isBf16 = 1;
    }
    td.set_isBf16(isBf16);
    td.set_tasksPerCore(1);       // unused by v2 kernel (core range from GetBlockNum)
    td.set_usedCoreNum(usedCoreNum);

    td.SaveToBuffer(ctx->GetRawTilingData()->GetData(),
                    ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(td.GetDataSize());

    ctx->SetTilingKey(1);   // always key 1; dtype selected at runtime from td.isBf16

    size_t* currentWs = ctx->GetWorkspaceSizes(1);
    currentWs[0] = 16 * 1024 * 1024;

    ctx->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareFusedKvNormRopeSwa(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<FusedKvNormRopeSwaCacheCustomCompileInfo>();
    if (compileInfo == nullptr) return ge::GRAPH_FAILED;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) return ge::GRAPH_FAILED;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedKvNormRopeSwaCacheCustom)
    .Tiling(TilingFuncFusedKvNormRopeSwa)
    .TilingParse<FusedKvNormRopeSwaCacheCustomCompileInfo>(TilingPrepareFusedKvNormRopeSwa);

}  // namespace optiling
