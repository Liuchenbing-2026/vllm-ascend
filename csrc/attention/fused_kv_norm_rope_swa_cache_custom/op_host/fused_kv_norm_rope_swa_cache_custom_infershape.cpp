/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

/*!
 * \file fused_kv_norm_rope_swa_cache_custom_infershape.cpp
 * \brief kv_cache_out shape/dtype = kv_cache(in) shape/dtype.
 */
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int IDX_CACHE_IN = 5;
static constexpr int OUT_CACHE = 0;

static ge::graphStatus InferShape4FusedKvNormRopeSwa(gert::InferShapeContext* context)
{
    const gert::Shape* cacheShape = context->GetInputShape(IDX_CACHE_IN);
    if (cacheShape == nullptr) {
        return GRAPH_FAILED;
    }
    gert::Shape* cacheOut = context->GetOutputShape(OUT_CACHE);
    if (cacheOut == nullptr) {
        return GRAPH_FAILED;
    }
    *cacheOut = *cacheShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FusedKvNormRopeSwa(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUT_CACHE, context->GetInputDataType(IDX_CACHE_IN));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedKvNormRopeSwaCacheCustom)
    .InferShape(InferShape4FusedKvNormRopeSwa)
    .InferDataType(InferDataType4FusedKvNormRopeSwa);

}  // namespace ops
