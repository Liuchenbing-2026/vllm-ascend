/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_dist_top_k_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
// indices_in is op input index 4 (index_q, index_k_cache, seq_len, block_table,
// indices_in). The output `indices` aliases / mirrors indices_in's shape.
static constexpr size_t INDICES_IN_INPUT_INDEX = 4;

static ge::graphStatus InferShapeMsaDistTopK(gert::InferShapeContext *context)
{
    gert::Shape *outShape = context->GetOutputShape(0);
    const gert::Shape *inputShape = context->GetInputShape(INDICES_IN_INPUT_INDEX);
    *outShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMsaDistTopK(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MsaDistTopK)
    .InferShape(InferShapeMsaDistTopK)
    .InferDataType(InferDataTypeMsaDistTopK);
} // namespace ops
