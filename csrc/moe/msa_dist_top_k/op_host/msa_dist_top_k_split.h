#ifndef MSA_DIST_TOP_K_SPLIT_H
#define MSA_DIST_TOP_K_SPLIT_H

#include "msa_dist_top_k.h"
#include "msa_dist_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
// MSA decode uses a single tiling path (key 10, the sequential "split_s" clone).
// The split-S class derives from the base tiling and overrides only IsCapable()
// (always true so key 10 is emitted) and GetTilingKey() (returns 10). All the
// shape / matmul / topk math is inherited from MsaDistTopKTiling::DoOpTiling.
class MsaDistTopKSplitSTiling : public MsaDistTopKTiling {
public:
    MsaDistTopKSplitSTiling(gert::TilingContext *context) : MsaDistTopKTiling(context) {}

    bool IsCapable();

    uint64_t GetTilingKey();
};
}
#endif
