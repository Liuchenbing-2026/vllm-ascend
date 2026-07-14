#ifndef HC_POST_PROTO_H_
#define HC_POST_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(HcPost)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(residual, ge::TensorType::ALL())
    .INPUT(post, ge::TensorType::ALL())
    .INPUT(comb, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(HcPost);

}

#endif
