#ifndef NGRAM_SPEC_DECODE_PROTO_H_
#define NGRAM_SPEC_DECODE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(NgramSpecDecode)
    .INPUT(tokenIds, ge::TensorType::ALL())
    .INPUT(numTokensNoSpec, ge::TensorType::ALL())
    .INPUT(sampledTokenIds, ge::TensorType::ALL())
    .INPUT(discardRequestMask, ge::TensorType::ALL())
    .OUTPUT(nextTokenIds, ge::TensorType::ALL())
    .OUTPUT(draftTokenIds, ge::TensorType::ALL())
    .OUTPUT(numValidDraftTokens, ge::TensorType::ALL())
    .REQUIRED_ATTR(vocab_size, Int)
    .REQUIRED_ATTR(min_n, Int)
    .REQUIRED_ATTR(max_n, Int)
    .REQUIRED_ATTR(k, Int)
    .OP_END_FACTORY_REG(NgramSpecDecode);

}

#endif
