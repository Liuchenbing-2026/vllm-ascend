#ifndef STORE_KV_BLOCK_PROTO_H_
#define STORE_KV_BLOCK_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(StoreKVBlock)
    .INPUT(keyIn, ge::TensorType::ALL())
    .INPUT(keyCacheIn, ge::TensorType::ALL())
    .INPUT(groupLen, ge::TensorType::ALL())
    .INPUT(groupKeyIdx, ge::TensorType::ALL())
    .INPUT(groupKeyCacheIdx, ge::TensorType::ALL())
    .REQUIRED_ATTR(blockSize, Int)
    .OP_END_FACTORY_REG(StoreKVBlock);

}

#endif
