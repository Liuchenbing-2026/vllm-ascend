/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/op_def_registry.h"
#include "log/ops_log.h"

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                                       \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)

using namespace ge;

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // sampled_token_ids: [B, max_new]
    const gert::Shape* sampledShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampledShape);

    int64_t B = sampledShape->GetDim(0);
    int64_t maxNew = sampledShape->GetDim(1);

    // next_token_ids: [B]
    gert::Shape* outNextTokenIds = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outNextTokenIds);
    outNextTokenIds->SetDimNum(1);
    outNextTokenIds->SetDim(0, B);

    // valid_count: [B]
    gert::Shape* outValidCount = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outValidCount);
    outValidCount->SetDimNum(1);
    outValidCount->SetDim(0, B);

    // valid_sampled_token_ids: [B, max_new]
    gert::Shape* outValidSampled = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outValidSampled);
    outValidSampled->SetDimNum(2);
    outValidSampled->SetDim(0, B);
    outValidSampled->SetDim(1, maxNew);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    context->SetOutputDataType(1, DT_INT32);
    context->SetOutputDataType(2, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UpdateTokenIdsNgram)
    .InferShape(InferShape)
    .InferDataType(InferDataType);

}  // namespace ops
