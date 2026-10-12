#include "register/op_def_registry.h"

namespace ge {
ge::graphStatus InferShapeKvI8Store(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    int64_t numBlocks = 0;
    int64_t blockSize = 0;
    if (context->GetAttrs() != nullptr) {
        if (context->GetAttrs()->GetAttrPointer<int64_t>(0) != nullptr) {
            numBlocks = *context->GetAttrs()->GetAttrPointer<int64_t>(0);
        }
        if (context->GetAttrs()->GetAttrPointer<int64_t>(1) != nullptr) {
            blockSize = *context->GetAttrs()->GetAttrPointer<int64_t>(1);
        }
    }
    gert::Shape *cacheShape = context->GetOutputShape(0);
    if (cacheShape == nullptr) {
        return GRAPH_FAILED;
    }
    *cacheShape = gert::Shape({numBlocks, blockSize, xShape->GetDim(1), xShape->GetDim(2)});
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvI8Store(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge