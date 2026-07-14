#ifndef HC_PRE_PROTO_H_
#define HC_PRE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(HcPre)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(hc_fn, ge::TensorType::ALL())
    .INPUT(hc_scale, ge::TensorType::ALL())
    .INPUT(hc_base, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(post, ge::TensorType::ALL())
    .OUTPUT(comb_frag, ge::TensorType::ALL())
    .ATTR(hc_mult, Int, 4)
    .ATTR(hc_sinkhorn_iters, Int, 20)
    .ATTR(hc_eps, Float, 1e-06)
    .ATTR(norm_eps, Float, 1e-06)
    .OP_END_FACTORY_REG(HcPre);

}

#endif
