#include "register/op_def_registry.h"

namespace ge {
ge::graphStatus InferShapeKvTqReadBench(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeKvTqReadBench(gert::InferDataTypeContext *context);
} // namespace ge
namespace optiling {
ge::graphStatus TilingKvTqReadBench(gert::TilingContext *context);
} // namespace optiling

namespace ops {
class KvTqReadBench : public OpDef {
public:
    explicit KvTqReadBench(const char *name) : OpDef(name)
    {
        this->Input("packed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dense")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mode").Int();
        this->Attr("rows").Int();

        this->SetInferShape(ge::InferShapeKvTqReadBench).SetInferDataType(ge::InferDataTypeKvTqReadBench);
        this->AICore().SetTiling(optiling::TilingKvTqReadBench);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(KvTqReadBench);
} // namespace ops