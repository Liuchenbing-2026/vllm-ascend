#include "../op_kernel/tq_paged_read_pair_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t ROW_BYTES = 66;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t JOB_FIELDS = 4;
constexpr int64_t LOOKUP_ELEMENTS = 5696;  // LUT plus fixed uint32 offset tables, as BF16 bits
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *cache = context->GetInputShape(0);
    const auto *table = context->GetInputShape(2);
    const auto *jobs = context->GetInputShape(3);
    const auto *lookup = context->GetInputShape(4);
    if (cache == nullptr || table == nullptr || jobs == nullptr || lookup == nullptr ||
        context->GetAttrs() == nullptr || context->GetAttrs()->GetInt(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *value = context->GetInputShape(1);
    if (value == nullptr || value->GetStorageShape().GetDimNum() != 4) return ge::GRAPH_FAILED;
    const auto &c = cache->GetStorageShape();
    if (c.GetDimNum() != 4) return ge::GRAPH_FAILED;
    for (int i = 0; i < 4; ++i) {
        if (value->GetStorageShape().GetDim(i) != c.GetDim(i)) return ge::GRAPH_FAILED;
    }
    const auto &t = table->GetStorageShape();
    const auto &j = jobs->GetStorageShape();
    if (c.GetDimNum() != 4 || t.GetDimNum() != 2 || j.GetDimNum() != 2 ||
        c.GetDim(3) != ROW_BYTES || j.GetDim(1) != JOB_FIELDS ||
        lookup->GetStorageShape().GetShapeSize() != LOOKUP_ELEMENTS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = *context->GetAttrs()->GetInt(0);
    if (tokens <= 0 || c.GetDim(0) <= 0 || c.GetDim(1) <= 0 || c.GetDim(2) <= 0 ||
        t.GetDim(0) <= 0 || t.GetDim(1) <= 0 || j.GetDim(0) <= 0 ||
        j.GetDim(0) > UINT32_MAX / 2 || c.GetDim(1) > UINT32_MAX || c.GetDim(2) > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<TqPagedReadPairTilingData>();
    data->jobs = static_cast<uint32_t>(j.GetDim(0));
    data->heads = static_cast<uint32_t>(c.GetDim(2));
    data->blockTokens = static_cast<uint32_t>(c.GetDim(1));
    data->tableWidth = static_cast<uint32_t>(t.GetDim(1));
    data->cacheBytes = static_cast<uint64_t>(c.GetShapeSize());
    data->outputElements = static_cast<uint64_t>(tokens) * data->heads * HEAD_DIM;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    context->SetBlockDim(data->jobs * 2 < cores ? data->jobs * 2 : cores);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *cache = context->GetInputShape(0);
    if (cache == nullptr || cache->GetDimNum() != 4 || context->GetAttrs() == nullptr) {
        return GRAPH_FAILED;
    }
    const auto *tokens = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (tokens == nullptr || *tokens < 0) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape({*tokens, cache->GetDim(2), HEAD_DIM});
    *context->GetOutputShape(1) = *context->GetOutputShape(0);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DT_BF16);
    context->SetOutputDataType(1, DT_BF16);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class TqPagedReadPair : public OpDef {
public:
    explicit TqPagedReadPair(const char *name) : OpDef(name)
    {
        this->Input("cache").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_cache").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("jobs").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lookup").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dense").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dense_value").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("tokens").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
    }
};
OP_ADD(TqPagedReadPair);
}
