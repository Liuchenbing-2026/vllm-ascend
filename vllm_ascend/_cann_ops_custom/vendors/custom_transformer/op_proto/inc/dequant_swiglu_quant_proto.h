#ifndef DEQUANT_SWIGLU_QUANT_PROTO_H_
#define DEQUANT_SWIGLU_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DequantSwigluQuant)
    .INPUT(x, ge::TensorType::ALL())
    .OPTIONAL_INPUT(weight_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(activation_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_offset, ge::TensorType::ALL())
    .OPTIONAL_INPUT(group_index, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(scale, ge::TensorType::ALL())
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "static")
    .ATTR(dst_type, Int, 2)
    .ATTR(round_mode, String, "rint")
    .ATTR(activate_dim, Int, -1)
    .ATTR(swiglu_mode, Int, 0)
    .ATTR(clamp_limit, Float, 0)
    .ATTR(glu_alpha, Float, 1)
    .ATTR(glu_bias, Float, 0)
    .OP_END_FACTORY_REG(DequantSwigluQuant);

}

#endif
