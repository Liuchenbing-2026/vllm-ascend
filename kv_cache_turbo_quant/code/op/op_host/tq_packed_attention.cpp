// Prepared host interface and tiling. CANN compilation is still unverified.
#include "tq_packed_attention_plan.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus PackedAttentionTiling(gert::TilingContext *context)
{
    for (uint32_t i = 0; i < 6; ++i) {
        if (context->GetInputShape(i) == nullptr) return ge::GRAPH_FAILED;
    }
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &k = context->GetInputShape(1)->GetStorageShape();
    const auto &v = context->GetInputShape(2)->GetStorageShape();
    const auto &table = context->GetInputShape(3)->GetStorageShape();
    const auto &lengths = context->GetInputShape(4)->GetStorageShape();
    const auto &centroids = context->GetInputShape(5)->GetStorageShape();
    const auto *attrs = context->GetAttrs();
    if (q.GetDimNum() != 3 || k.GetDimNum() != 4 || v.GetDimNum() != 4 ||
        table.GetDimNum() != 2 || lengths.GetDimNum() != 1 || centroids.GetDimNum() != 1 ||
        q.GetDim(2) != 128 || k.GetDim(3) != 66 || centroids.GetDim(0) != 16 ||
        table.GetDim(0) != q.GetDim(0) || lengths.GetDim(0) != q.GetDim(0) ||
        attrs == nullptr || attrs->GetFloat(0) == nullptr || attrs->GetInt(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t dim = 0; dim < 4; ++dim) {
        if (k.GetDim(dim) != v.GetDim(dim)) return ge::GRAPH_FAILED;
    }
    for (int64_t dim : {q.GetDim(0), q.GetDim(1), k.GetDim(0), k.GetDim(1),
                         k.GetDim(2), table.GetDim(1), *attrs->GetInt(1)}) {
        if (dim < 0 || static_cast<uint64_t>(dim) > UINT32_MAX) return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TqPackedAttentionTilingData plan{};
    if (!tq_packed_attention::MakePlan(
            static_cast<uint32_t>(q.GetDim(0)), static_cast<uint32_t>(q.GetDim(1)),
            static_cast<uint32_t>(k.GetDim(2)), static_cast<uint32_t>(k.GetDim(0)),
            static_cast<uint32_t>(k.GetDim(1)), static_cast<uint32_t>(table.GetDim(1)),
            ubBytes, platform.GetCoreNumAiv(), static_cast<uint32_t>(*attrs->GetInt(1)),
            *attrs->GetFloat(0), plan)) return ge::GRAPH_FAILED;
    *context->GetTilingData<TqPackedAttentionTilingData>() = plan;
    context->SetBlockDim(plan.blockDim);
    // User GM workspace only. Caller still honors the actual ACLNN workspace query.
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus PackedAttentionInferShape(gert::InferShapeContext *context)
{
    const auto *q = context->GetInputShape(0);
    const auto *k = context->GetInputShape(1);
    if (q == nullptr || k == nullptr || q->GetDimNum() != 3 || k->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *q;
    *context->GetOutputShape(1) = gert::Shape({q->GetDim(0), k->GetDim(2)});
    return GRAPH_SUCCESS;
}
static graphStatus PackedAttentionInferType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DT_BF16);
    context->SetOutputDataType(1, DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class TqPackedAttention : public OpDef {
public:
    explicit TqPackedAttention(const char *name) : OpDef(name)
    {
        this->Input("query_rot").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("table").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lengths").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroids").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_rot").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("status").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").Float();
        this->Attr("api_scratch_bytes").Int();
        this->SetInferShape(ge::PackedAttentionInferShape).SetInferDataType(ge::PackedAttentionInferType);
        this->AICore().SetTiling(optiling::PackedAttentionTiling);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
    }
};
OP_ADD(TqPackedAttention);
}
