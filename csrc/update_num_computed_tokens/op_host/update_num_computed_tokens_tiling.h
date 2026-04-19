#ifndef UPDATE_NUM_COMPUTED_TOKENS_TILING_H
#define UPDATE_NUM_COMPUTED_TOKENS_TILING_H

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UpdateNumComputedTokensTilingData)
    // ---- core distribution ----
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, numReqs);
    TILING_DATA_FIELD_DEF(uint32_t, reqsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderReqs);

    // ---- buffer sizes for GM binding ----
    TILING_DATA_FIELD_DEF(uint32_t, validSampledSize);   // valid_sampled_token_count total size
    TILING_DATA_FIELD_DEF(uint32_t, prevDraftSize);       // prev_num_draft_tokens total size
END_TILING_DATA_DEF;

struct UpdateNumComputedTokensCompileInfo {
    uint32_t totalCoreNum = 0;
};

REGISTER_TILING_DATA_CLASS(UpdateNumComputedTokens, UpdateNumComputedTokensTilingData)

}  // namespace optiling

#endif  // UPDATE_NUM_COMPUTED_TOKENS_TILING_H
