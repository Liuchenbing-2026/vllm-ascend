/**
 * @file compute_slot_mapping_tiling.cpp
 * @brief ComputeSlotMapping TilingFunc implementation
 */

#include "compute_slot_mapping_tiling.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

#include <algorithm>

namespace optiling {

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& coreNum)
{
    auto ptrCompileInfo = reinterpret_cast<const ComputeSlotMappingCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = ascendcPlatform.GetCoreNum();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_I(context, "Enter TilingFunc for ComputeSlotMapping");

    // ========== 1. Get hardware core count ==========
    uint32_t coreNum;
    GetCompileParameters(context, coreNum);

    // ========== 2. Derive numTokens from req_indices shape ==========
    auto reqIndicesShape = context->GetInputShape(0);
    uint32_t numTokens = 0;
    if (reqIndicesShape != nullptr &&
        reqIndicesShape->GetStorageShape().GetDimNum() > 0) {
        numTokens = static_cast<uint32_t>(reqIndicesShape->GetStorageShape().GetDim(0));
    }

    // ========== 3. Derive blockTableSize from block_table shape ==========
    auto blockTableShape = context->GetInputShape(2);
    uint32_t blockTableSize = 0;
    if (blockTableShape != nullptr &&
        blockTableShape->GetStorageShape().GetDimNum() > 0) {
        blockTableSize = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(0));
    }

    // ========== 4. Get operator attributes ==========
    auto attrs = context->GetAttrs();
    int32_t blockSize = *(attrs->GetAttrPointer<int32_t>(0));
    int32_t blockTableStride = *(attrs->GetAttrPointer<int32_t>(1));
    int32_t cpSize = *(attrs->GetAttrPointer<int32_t>(2));
    int32_t cpRank = *(attrs->GetAttrPointer<int32_t>(3));
    int32_t cpInterleave = *(attrs->GetAttrPointer<int32_t>(4));

    // ========== 5. Compute core distribution ==========
    uint32_t usedCoreNum = std::min(coreNum, numTokens);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    uint32_t tokensPerCore = numTokens / usedCoreNum;
    uint32_t remainderTokens = numTokens % usedCoreNum;

    // ========== 6. Set tiling_key ==========
    // key=0: cp_size == 1 (common path), key=1: cp_size > 1 (CP path)
    context->SetTilingKey(cpSize > 1 ? 1 : 0);

    // ========== 7. Fill TilingData ==========
    ComputeSlotMappingTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numTokens(numTokens);
    tiling.set_tokensPerCore(tokensPerCore);
    tiling.set_remainderTokens(remainderTokens);
    tiling.set_blockSize(blockSize);
    tiling.set_blockTableStride(blockTableStride);
    tiling.set_cpSize(cpSize);
    tiling.set_cpRank(cpRank);
    tiling.set_cpInterleave(cpInterleave);
    tiling.set_blockTableSize(blockTableSize);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ========== 8. Set block_dim ==========
    context->SetBlockDim(usedCoreNum);

    OPS_LOG_I(context, "Block Dim: %u, numTokens: %u, blockSize: %d, cpSize: %d",
        usedCoreNum, numTokens, blockSize, cpSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ComputeSlotMapping(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ComputeSlotMappingCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ComputeSlotMapping)
    .Tiling(TilingFunc)
    .TilingParse<ComputeSlotMappingCompileInfo>(TilingPrepare4ComputeSlotMapping);

}  // namespace optiling
