/**
 * @file update_num_computed_tokens_def.cpp
 * @brief UpdateNumComputedTokens OpDef registration
 */

#include "register/op_def_registry.h"

namespace ops {

class UpdateNumComputedTokens : public OpDef {
public:
    explicit UpdateNumComputedTokens(const char* name) : OpDef(name)
    {
        // -------------------- Inputs --------------------
        this->Input("prev_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("valid_sampled_token_count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prev_num_draft_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cpu_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Outputs --------------------
        // Pre-allocated by caller; kernel selectively updates elements
        this->Output("num_computed_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("num_accepted_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(UpdateNumComputedTokens);

}  // namespace ops
