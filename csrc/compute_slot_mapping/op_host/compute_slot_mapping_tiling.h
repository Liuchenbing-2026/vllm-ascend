#ifndef COMPUTE_SLOT_MAPPING_TILING_H
#define COMPUTE_SLOT_MAPPING_TILING_H

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ComputeSlotMappingTilingData)
    // ---- core distribution ----
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderTokens);

    // ---- operator attributes ----
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, blockTableStride);
    TILING_DATA_FIELD_DEF(int32_t, cpSize);
    TILING_DATA_FIELD_DEF(int32_t, cpRank);
    TILING_DATA_FIELD_DEF(int32_t, cpInterleave);

    // ---- block table total size (for GM binding) ----
    TILING_DATA_FIELD_DEF(uint32_t, blockTableSize);
END_TILING_DATA_DEF;

struct ComputeSlotMappingCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(ComputeSlotMapping, ComputeSlotMappingTilingData)

}  // namespace optiling

#endif  // COMPUTE_SLOT_MAPPING_TILING_H
