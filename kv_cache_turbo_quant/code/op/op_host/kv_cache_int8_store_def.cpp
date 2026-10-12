#include "register/op_def_registry.h"

namespace ge {
ge::graphStatus InferShapeKvI8Store(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeKvI8Store(gert::InferDataTypeContext *context);
} // namespace ge
namespace optiling {
ge::graphStatus TilingKvI8Store(gert::TilingContext *context);
} // namespace optiling

namespace ops {
class KvCacheInt8Store : public OpDef {
public:
    explicit KvCacheInt8Store(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_blocks").Int();
        this->Attr("block_size").Int();

        this->SetInferShape(ge::InferShapeKvI8Store).SetInferDataType(ge::InferDataTypeKvI8Store);
        this->AICore().SetTiling(optiling::TilingKvI8Store);

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
OP_ADD(KvCacheInt8Store);
} // namespace ops