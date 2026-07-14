#ifndef TRANSPOSE_KV_CACHE_BY_BLOCK_PROTO_H_
#define TRANSPOSE_KV_CACHE_BY_BLOCK_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(TransposeKvCacheByBlock)
    .DYNAMIC_INPUT(KCache, ge::TensorType::ALL())
    .DYNAMIC_INPUT(VCache, ge::TensorType::ALL())
    .INPUT(blockIDs, ge::TensorType::ALL())
    .REQUIRED_ATTR(blockSize, Int)
    .REQUIRED_ATTR(headNum, Int)
    .REQUIRED_ATTR(headDim, Int)
    .REQUIRED_ATTR(splitNum, Int)
    .REQUIRED_ATTR(layerNum, Int)
    .OP_END_FACTORY_REG(TransposeKvCacheByBlock);

}

#endif
