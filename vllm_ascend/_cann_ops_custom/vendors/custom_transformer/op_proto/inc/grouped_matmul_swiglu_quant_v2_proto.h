#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_PROTO_H_
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GroupedMatmulSwigluQuantV2)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(x_scale, ge::TensorType::ALL())
    .INPUT(group_list, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight_scale, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight_assist_matrix, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(smooth_scale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(y_scale, ge::TensorType::ALL())
    .ATTR(dequant_mode, Int, 0)
    .ATTR(dequant_dtype, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(quant_dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(group_list_type, Int, 0)
    .ATTR(tuning_config, ListInt, {0})
    .ATTR(swiglu_limit, Float, 0)
    .OP_END_FACTORY_REG(GroupedMatmulSwigluQuantV2);

}

#endif
