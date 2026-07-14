#ifndef COMPRESSOR_PROTO_H_
#define COMPRESSOR_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Compressor)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(wkv, ge::TensorType::ALL())
    .INPUT(wgate, ge::TensorType::ALL())
    .INPUT(state_cache, ge::TensorType::ALL())
    .INPUT(ape, ge::TensorType::ALL())
    .INPUT(norm_weight, ge::TensorType::ALL())
    .INPUT(rope_sin, ge::TensorType::ALL())
    .INPUT(rope_cos, ge::TensorType::ALL())
    .OPTIONAL_INPUT(state_block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused, ge::TensorType::ALL())
    .OPTIONAL_INPUT(start_pos, ge::TensorType::ALL())
    .OUTPUT(cmp_kv, ge::TensorType::ALL())
    .OUTPUT(state_cache, ge::TensorType::ALL())
    .REQUIRED_ATTR(rope_head_dim, Int)
    .REQUIRED_ATTR(cmp_ratio, Int)
    .ATTR(coff, Int, 1)
    .ATTR(norm_eps, Float, 1e-06)
    .ATTR(rotary_mode, Int, 1)
    .ATTR(cache_mode, Int, 1)
    .ATTR(state_cache_stride_dim0, Int, 0)
    .OP_END_FACTORY_REG(Compressor);

}

#endif
