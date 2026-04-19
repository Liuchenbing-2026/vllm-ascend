/**
 * @file compute_slot_mapping_infershape.cpp
 * @brief InferShape and InferDataType for ComputeSlotMapping
 */

#include "register/op_def_registry.h"
#include "log/ops_log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ComputeSlotMapping(gert::InferShapeContext* context)
{
    // Output slot_mapping has same shape as input req_indices: [num_tokens]
    const gert::Shape* reqIndicesShape = context->GetInputShape(0);
    if (reqIndicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    outShape->SetDimNum(1);
    outShape->SetDim(0, reqIndicesShape->GetDim(0));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ComputeSlotMapping(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ComputeSlotMapping)
    .InferShape(InferShape4ComputeSlotMapping)
    .InferDataType(InferDataType4ComputeSlotMapping);

}  // namespace ops
