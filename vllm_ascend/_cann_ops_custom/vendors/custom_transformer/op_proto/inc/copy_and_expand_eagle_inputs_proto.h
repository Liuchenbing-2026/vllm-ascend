#ifndef COPY_AND_EXPAND_EAGLE_INPUTS_PROTO_H_
#define COPY_AND_EXPAND_EAGLE_INPUTS_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(CopyAndExpandEagleInputs)
    .INPUT(target_token_ids, ge::TensorType::ALL())
    .INPUT(target_positions, ge::TensorType::ALL())
    .INPUT(next_token_ids, ge::TensorType::ALL())
    .INPUT(query_start_loc, ge::TensorType::ALL())
    .INPUT(query_end_loc, ge::TensorType::ALL())
    .OUTPUT(out_input_ids, ge::TensorType::ALL())
    .OUTPUT(out_positions, ge::TensorType::ALL())
    .OUTPUT(out_is_rejected_token_mask, ge::TensorType::ALL())
    .OUTPUT(out_is_masked_token_mask, ge::TensorType::ALL())
    .OUTPUT(out_new_token_indices, ge::TensorType::ALL())
    .OUTPUT(out_hidden_state_mapping, ge::TensorType::ALL())
    .REQUIRED_ATTR(padding_token_id, Int)
    .REQUIRED_ATTR(parallel_drafting_token_id, Int)
    .REQUIRED_ATTR(num_padding_slots_per_request, Int)
    .REQUIRED_ATTR(shift_input_ids, Bool)
    .REQUIRED_ATTR(total_input_tokens, Int)
    .OP_END_FACTORY_REG(CopyAndExpandEagleInputs);

}

#endif
