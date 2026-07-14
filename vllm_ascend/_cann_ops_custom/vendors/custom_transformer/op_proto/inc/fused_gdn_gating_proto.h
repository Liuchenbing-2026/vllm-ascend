#ifndef FUSED_GDN_GATING_PROTO_H_
#define FUSED_GDN_GATING_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FusedGdnGating)
    .INPUT(a_log, ge::TensorType::ALL())
    .INPUT(a, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .INPUT(dt_bias, ge::TensorType::ALL())
    .OUTPUT(g, ge::TensorType::ALL())
    .OUTPUT(beta_output, ge::TensorType::ALL())
    .ATTR(beta, Float, 1)
    .ATTR(threshold, Float, 20)
    .OP_END_FACTORY_REG(FusedGdnGating);

}

#endif
