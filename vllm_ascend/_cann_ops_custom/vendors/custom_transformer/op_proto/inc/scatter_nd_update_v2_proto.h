#ifndef SCATTER_ND_UPDATE_V2_PROTO_H_
#define SCATTER_ND_UPDATE_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ScatterNdUpdateV2)
    .INPUT(var, ge::TensorType::ALL())
    .INPUT(indices, ge::TensorType::ALL())
    .INPUT(updates, ge::TensorType::ALL())
    .OUTPUT(var, ge::TensorType::ALL())
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdUpdateV2);

}

#endif
