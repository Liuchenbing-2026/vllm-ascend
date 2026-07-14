#ifndef GROUPED_MATMUL_SWIGLU_QUANT_PROTO_H_
#define GROUPED_MATMUL_SWIGLU_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GroupedMatmulSwigluQuant)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(weight_scale, ge::TensorType::ALL())
    .INPUT(x_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(weight_assistance_matrix, ge::TensorType::ALL())
    .INPUT(group_list, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(y_scale, ge::TensorType::ALL())
    .ATTR(is_enable_weight_assistance_matrix, Bool, true)
    .ATTR(dequant_mode, Int, 0)
    .ATTR(limited, Float, 0)
    .OP_END_FACTORY_REG(GroupedMatmulSwigluQuant);

}

#endif
