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
    // token_ids: [B, max_len]
    const gert::Shape* tokenIdsShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenIdsShape);
    int64_t B = tokenIdsShape->GetDim(0);

    // Get k from attributes
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t k = *(attrs->GetAttrPointer<int64_t>(2));

    // draft_tokens: [B, k]
    gert::Shape* outDraftTokens = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDraftTokens);
    outDraftTokens->SetDimNum(2);
    outDraftTokens->SetDim(0, B);
    outDraftTokens->SetDim(1, k);

    // num_valid_draft_tokens: [B]
    gert::Shape* outNumValid = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outNumValid);
    outNumValid->SetDimNum(1);
    outNumValid->SetDim(0, B);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    context->SetOutputDataType(1, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NgramMatchExtract)
    .InferShape(InferShape)
    .InferDataType(InferDataType);

}  // namespace ops
