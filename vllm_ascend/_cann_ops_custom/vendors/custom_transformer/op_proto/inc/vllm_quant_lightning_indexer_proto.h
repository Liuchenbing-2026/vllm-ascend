#ifndef VLLM_QUANT_LIGHTNING_INDEXER_PROTO_H_
#define VLLM_QUANT_LIGHTNING_INDEXER_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(VllmQuantLightningIndexer)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(weights, ge::TensorType::ALL())
    .INPUT(query_dequant_scale, ge::TensorType::ALL())
    .INPUT(key_dequant_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_key, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(metadata, ge::TensorType::ALL())
    .OUTPUT(sparse_indices, ge::TensorType::ALL())
    .OUTPUT(sparse_values, ge::TensorType::ALL())
    .REQUIRED_ATTR(query_quant_mode, Int)
    .REQUIRED_ATTR(key_quant_mode, Int)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "BSND")
    .ATTR(sparse_count, Int, 2048)
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(cmp_ratio, Int, 1)
    .ATTR(return_values, Bool, false)
    .ATTR(stride, Int, 1)
    .ATTR(scale_stride, Int, 1)
    .OP_END_FACTORY_REG(VllmQuantLightningIndexer);

}

#endif
