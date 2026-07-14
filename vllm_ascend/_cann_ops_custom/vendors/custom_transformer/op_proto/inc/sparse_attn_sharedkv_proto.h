#ifndef SPARSE_ATTN_SHAREDKV_PROTO_H_
#define SPARSE_ATTN_SHAREDKV_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseAttnSharedkv)
    .INPUT(q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(sinks, ge::TensorType::ALL())
    .OPTIONAL_INPUT(metadata, ge::TensorType::ALL())
    .OUTPUT(attn_out, ge::TensorType::ALL())
    .OUTPUT(softmax_lse, ge::TensorType::ALL())
    .REQUIRED_ATTR(softmax_scale, Float)
    .REQUIRED_ATTR(cmp_ratio, Int)
    .REQUIRED_ATTR(ori_mask_mode, Int)
    .REQUIRED_ATTR(cmp_mask_mode, Int)
    .REQUIRED_ATTR(ori_kv_stride, Int)
    .REQUIRED_ATTR(cmp_kv_stride, Int)
    .ATTR(ori_win_left, Int, 127)
    .ATTR(ori_win_right, Int, 0)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "PA_ND")
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(SparseAttnSharedkv);

}

#endif
