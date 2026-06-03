/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#ifndef FUSED_KV_NORM_ROPE_SWA_CACHE_CUSTOM_TILING_H
#define FUSED_KV_NORM_ROPE_SWA_CACHE_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedKvNormRopeSwaCacheCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, nopeDim);
    TILING_DATA_FIELD_DEF(uint32_t, ropeDim);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
    TILING_DATA_FIELD_DEF(float,    eps);
    TILING_DATA_FIELD_DEF(float,    headDimF);
    TILING_DATA_FIELD_DEF(uint32_t, isBf16);
    TILING_DATA_FIELD_DEF(uint32_t, tasksPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

struct FusedKvNormRopeSwaCacheCustomCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

REGISTER_TILING_DATA_CLASS(FusedKvNormRopeSwaCacheCustom, FusedKvNormRopeSwaCacheCustomTilingData)
}  // namespace optiling

#endif  // FUSED_KV_NORM_ROPE_SWA_CACHE_CUSTOM_TILING_H
