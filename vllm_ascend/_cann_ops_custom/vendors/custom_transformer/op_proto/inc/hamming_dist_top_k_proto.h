#ifndef HAMMING_DIST_TOP_K_PROTO_H_
#define HAMMING_DIST_TOP_K_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(HammingDistTopK)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key_compressed, ge::TensorType::ALL())
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(seq_len, ge::TensorType::ALL())
    .INPUT(chunk_size, ge::TensorType::ALL())
    .INPUT(key_block_table, ge::TensorType::ALL())
    .INPUT(indices_in, ge::TensorType::ALL())
    .OPTIONAL_INPUT(key_compressed_rope, ge::TensorType::ALL())
    .OPTIONAL_INPUT(mask, ge::TensorType::ALL())
    .OUTPUT(indices, ge::TensorType::ALL())
    .ATTR(max_seq_len, Int, 0)
    .ATTR(sink, Int, 0)
    .ATTR(recent, Int, 0)
    .ATTR(support_offload, Int, 0)
    .OP_END_FACTORY_REG(HammingDistTopK);

}

#endif
