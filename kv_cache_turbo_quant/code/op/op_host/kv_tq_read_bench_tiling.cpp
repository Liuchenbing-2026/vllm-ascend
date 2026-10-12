#include "../op_kernel/kv_tq_read_bench_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus TilingKvTqReadBench(gert::TilingContext *context)
{
    KvTqReadBenchTilingData *tiling = context->GetTilingData<KvTqReadBenchTilingData>();

    int64_t mode = 0;
    int64_t rows = 0;
    if (context->GetAttrs() != nullptr) {
        if (context->GetAttrs()->GetInt(0) != nullptr) {
            mode = *context->GetAttrs()->GetInt(0);
        }
        if (context->GetAttrs()->GetInt(1) != nullptr) {
            rows = *context->GetAttrs()->GetInt(1);
        }
    }
    if (mode < 0 || mode > 2 || rows <= 0) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();

    tiling->mode = static_cast<uint32_t>(mode);
    tiling->rows = static_cast<uint32_t>(rows);
    tiling->usedCores = aivNum;
    tiling->pad = 0;

    context->SetBlockDim(aivNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling