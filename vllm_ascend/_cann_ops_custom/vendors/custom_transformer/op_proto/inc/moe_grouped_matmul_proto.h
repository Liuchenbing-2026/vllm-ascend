#ifndef MOE_GROUPED_MATMUL_PROTO_H_
#define MOE_GROUPED_MATMUL_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MoeGroupedMatmul)
    .DYNAMIC_INPUT(x, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight, ge::TensorType::ALL())
    .INPUT(group_list, ge::TensorType::ALL())
    .DYNAMIC_OUTPUT(y, ge::TensorType::ALL())
    .ATTR(transpose_weight, Bool, false)
    .OP_END_FACTORY_REG(MoeGroupedMatmul);

}

#endif
