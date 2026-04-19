/**
 * @file compute_slot_mapping_def.cpp
 * @brief ComputeSlotMapping OpDef registration
 */

#include "register/op_def_registry.h"

namespace ops {

class ComputeSlotMapping : public OpDef {
public:
    explicit ComputeSlotMapping(const char* name) : OpDef(name)
    {
        // -------------------- Inputs --------------------
        this->Input("req_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Outputs --------------------
        this->Output("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Attributes --------------------
        this->Attr("block_size").Int();
        this->Attr("block_table_stride").Int();
        this->Attr("cp_size").Int();
        this->Attr("cp_rank").Int();
        this->Attr("cp_interleave").Int();

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ComputeSlotMapping);

}  // namespace ops
