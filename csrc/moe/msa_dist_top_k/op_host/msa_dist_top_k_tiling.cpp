#include "msa_dist_top_k_tiling.h"
#include "msa_dist_top_k.h"
#include "msa_dist_top_k_split.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"


namespace optiling {
static ge::graphStatus TilingPrepareForMsaDistTopK(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MsaDistTopKSplitSTiling msaDistTopKSplitSTiling(context);
    msaDistTopKSplitSTiling.GetShapeAttrsInfo();
    msaDistTopKSplitSTiling.GetPlatformInfo();
    auto can_split = msaDistTopKSplitSTiling.IsCapable();
    if (can_split) {
        msaDistTopKSplitSTiling.DoOpTiling();
        msaDistTopKSplitSTiling.DoLibApiTiling();
        msaDistTopKSplitSTiling.GetWorkspaceSize();
        msaDistTopKSplitSTiling.PostTiling();
        context->SetTilingKey(msaDistTopKSplitSTiling.GetTilingKey());
        return ge::GRAPH_SUCCESS;
    }

    MsaDistTopKTiling msaDistTopKTiling(context);
    msaDistTopKTiling.GetShapeAttrsInfo();
    msaDistTopKTiling.GetPlatformInfo();
    msaDistTopKTiling.IsCapable();
    msaDistTopKTiling.DoOpTiling();
    msaDistTopKTiling.GetWorkspaceSize();
    msaDistTopKTiling.PostTiling();
    context->SetTilingKey(msaDistTopKTiling.GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MsaDistTopK)
    .Tiling(TilingFunc)
    .TilingParse<MsaDistTopKCompileInfo>(TilingPrepareForMsaDistTopK);
}
