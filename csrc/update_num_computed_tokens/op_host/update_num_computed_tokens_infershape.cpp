/**
 * @file update_num_computed_tokens_infershape.cpp
 * @brief InferShape and InferDataType for UpdateNumComputedTokens
 */

#include "register/op_def_registry.h"
#include "log/ops_log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4UpdateNumComputedTokens(gert::InferShapeContext* context)
{
    // Both outputs have shape [num_reqs], derived from cpu_values (input 3)
    const gert::Shape* cpuValuesShape = context->GetInputShape(3);
    if (cpuValuesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t numReqs = cpuValuesShape->GetDim(0);

    // num_computed_tokens: [num_reqs]
    gert::Shape* outShape0 = context->GetOutputShape(0);
    if (outShape0 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape0->SetDimNum(1);
    outShape0->SetDim(0, numReqs);

    // num_accepted_tokens: [num_reqs]
    gert::Shape* outShape1 = context->GetOutputShape(1);
    if (outShape1 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape1->SetDimNum(1);
    outShape1->SetDim(0, numReqs);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4UpdateNumComputedTokens(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    context->SetOutputDataType(1, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UpdateNumComputedTokens)
    .InferShape(InferShape4UpdateNumComputedTokens)
    .InferDataType(InferDataType4UpdateNumComputedTokens);

}  // namespace ops
