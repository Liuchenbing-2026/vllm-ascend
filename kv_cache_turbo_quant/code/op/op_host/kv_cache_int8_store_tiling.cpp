#include "../op_kernel/kv_cache_int8_store_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus TilingKvI8Store(gert::TilingContext *context)
{
    KvCacheInt8StoreTilingData *tiling = context->GetTilingData<KvCacheInt8StoreTilingData>();

    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *scaleShape = context->GetInputShape(1);
    if (xShape == nullptr || scaleShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (xShape->GetStorageShape().GetDimNum() != 3 || scaleShape->GetStorageShape().GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = xShape->GetStorageShape().GetDim(0);
    const int64_t numKvHeads = xShape->GetStorageShape().GetDim(1);
    const int64_t headDim = xShape->GetStorageShape().GetDim(2);
    if (numKvHeads != scaleShape->GetStorageShape().GetDim(0) ||
        headDim != scaleShape->GetStorageShape().GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    if (headDim != 64 && headDim != 128 && headDim != 256 && headDim != 512) {
        return ge::GRAPH_FAILED;
    }
    int64_t numBlocks = 0;
    int64_t blockSize = 0;
    if (context->GetAttrs() != nullptr) {
        if (context->GetAttrs()->GetInt(0) != nullptr) {
            numBlocks = *context->GetAttrs()->GetInt(0);
        }
        if (context->GetAttrs()->GetInt(1) != nullptr) {
            blockSize = *context->GetAttrs()->GetInt(1);
        }
    }
    if (numTokens < 0 || numKvHeads <= 0 || numBlocks <= 0 || blockSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalRows = static_cast<uint64_t>(numTokens) * static_cast<uint64_t>(numKvHeads);
    if (totalRows > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    uint32_t used = totalRows < aivNum ? static_cast<uint32_t>(totalRows) : aivNum;
    if (used == 0) {
        used = 1;
    }

    tiling->numTokens = static_cast<uint32_t>(numTokens);
    tiling->numKvHeads = static_cast<uint32_t>(numKvHeads);
    tiling->headDim = static_cast<uint32_t>(headDim);
    tiling->numBlocks = static_cast<uint32_t>(numBlocks);
    tiling->blockSize = static_cast<uint32_t>(blockSize);
    tiling->usedCores = used;

    context->SetBlockDim(used);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling