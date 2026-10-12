#include "register/op_def_registry.h"

namespace ge {
ge::graphStatus InferShapeKvTqReadBench(gert::InferShapeContext *context)
{
    gert::Shape *outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outShape = gert::Shape({1024});
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvTqReadBench(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge