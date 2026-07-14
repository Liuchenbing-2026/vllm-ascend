#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_PROTO_H_
#define MATMUL_ALLREDUCE_ADD_RMSNORM_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MatmulAllreduceAddRmsnorm)
    .INPUT(x1, ge::TensorType::ALL())
    .INPUT(x2, ge::TensorType::ALL())
    .INPUT(residual, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(add_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(group_tp, String)
    .REQUIRED_ATTR(tp_rank_size, Int)
    .REQUIRED_ATTR(tp_rank_id, Int)
    .ATTR(epsilon, Float, 1e-06)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(is_gather_add_out, Bool, false)
    .OP_END_FACTORY_REG(MatmulAllreduceAddRmsnorm);

}

#endif
