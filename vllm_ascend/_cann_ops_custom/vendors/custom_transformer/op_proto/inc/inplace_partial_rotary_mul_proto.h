#ifndef INPLACE_PARTIAL_ROTARY_MUL_PROTO_H_
#define INPLACE_PARTIAL_ROTARY_MUL_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(InplacePartialRotaryMul)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(cos, ge::TensorType::ALL())
    .INPUT(sin, ge::TensorType::ALL())
    .OUTPUT(x, ge::TensorType::ALL())
    .ATTR(mode, Int, 0)
    .ATTR(partial_slice, ListInt, {0,0})
    .OP_END_FACTORY_REG(InplacePartialRotaryMul);

}

#endif
