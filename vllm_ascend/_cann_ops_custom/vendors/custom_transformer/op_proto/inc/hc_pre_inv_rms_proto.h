#ifndef HC_PRE_INV_RMS_PROTO_H_
#define HC_PRE_INV_RMS_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(HcPreInvRms)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 1e-06)
    .OP_END_FACTORY_REG(HcPreInvRms);

}

#endif
