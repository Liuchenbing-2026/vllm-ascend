/**
 * @file update_num_computed_tokens_tiling.cpp
 * @brief UpdateNumComputedTokens TilingFunc implementation
 */

#include "update_num_computed_tokens_tiling.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

#include <algorithm>

namespace optiling {

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& coreNum)
{
    auto ptrCompileInfo = reinterpret_cast<const UpdateNumComputedTokensCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = ascendcPlatform.GetCoreNum();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_I(context, "Enter TilingFunc for UpdateNumComputedTokens");

    // ========== 1. Get hardware core count ==========
    uint32_t coreNum;
    GetCompileParameters(context, coreNum);

    // ========== 2. Derive numReqs from cpu_values shape (input 3) ==========
    auto cpuValuesShape = context->GetInputShape(3);
    uint32_t numReqs = 0;
    if (cpuValuesShape != nullptr &&
        cpuValuesShape->GetStorageShape().GetDimNum() > 0) {
        numReqs = static_cast<uint32_t>(cpuValuesShape->GetStorageShape().GetDim(0));
    }

    // ========== 3. Get buffer sizes ==========
    // valid_sampled_token_count (input 1)
    auto validShape = context->GetInputShape(1);
    uint32_t validSampledSize = 0;
    if (validShape != nullptr &&
        validShape->GetStorageShape().GetDimNum() > 0) {
        validSampledSize = static_cast<uint32_t>(validShape->GetStorageShape().GetDim(0));
    }

    // prev_num_draft_tokens (input 2)
    auto draftShape = context->GetInputShape(2);
    uint32_t prevDraftSize = 0;
    if (draftShape != nullptr &&
        draftShape->GetStorageShape().GetDimNum() > 0) {
        prevDraftSize = static_cast<uint32_t>(draftShape->GetStorageShape().GetDim(0));
    }

    // ========== 4. Compute core distribution ==========
    uint32_t usedCoreNum = std::min(coreNum, numReqs);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    uint32_t reqsPerCore = numReqs / usedCoreNum;
    uint32_t remainderReqs = numReqs % usedCoreNum;

    // ========== 5. Set tiling_key ==========
    context->SetTilingKey(1);

    // ========== 6. Fill TilingData ==========
    UpdateNumComputedTokensTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numReqs(numReqs);
    tiling.set_reqsPerCore(reqsPerCore);
    tiling.set_remainderReqs(remainderReqs);
    tiling.set_validSampledSize(validSampledSize);
    tiling.set_prevDraftSize(prevDraftSize);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ========== 7. Set block_dim ==========
    context->SetBlockDim(usedCoreNum);

    OPS_LOG_I(context, "Block Dim: %u, numReqs: %u", usedCoreNum, numReqs);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4UpdateNumComputedTokens(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<UpdateNumComputedTokensCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpdateNumComputedTokens)
    .Tiling(TilingFunc)
    .TilingParse<UpdateNumComputedTokensCompileInfo>(TilingPrepare4UpdateNumComputedTokens);

}  // namespace optiling
