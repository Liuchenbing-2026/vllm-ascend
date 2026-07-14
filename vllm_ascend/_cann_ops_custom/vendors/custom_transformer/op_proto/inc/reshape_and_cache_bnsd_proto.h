#ifndef RESHAPE_AND_CACHE_BNSD_PROTO_H_
#define RESHAPE_AND_CACHE_BNSD_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ReshapeAndCacheBnsd)
    .INPUT(keyIn, ge::TensorType::ALL())
    .INPUT(keyCacheIn, ge::TensorType::ALL())
    .INPUT(slotMapping, ge::TensorType::ALL())
    .INPUT(seqLen, ge::TensorType::ALL())
    .OUTPUT(keyCacheOut, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(ReshapeAndCacheBnsd);

}

#endif
