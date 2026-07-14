#ifndef COMPRESSOR_METADATA_PROTO_H_
#define COMPRESSOR_METADATA_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(CompressorMetadata)
    .INPUT(ropeCos, ge::TensorType::ALL())
    .INPUT(ropeSin, ge::TensorType::ALL())
    .INPUT(cuSeqlens, ge::TensorType::ALL())
    .INPUT(startPos, ge::TensorType::ALL())
    .INPUT(kvBlockTable, ge::TensorType::ALL())
    .OUTPUT(compressCos, ge::TensorType::ALL())
    .OUTPUT(compressSin, ge::TensorType::ALL())
    .OUTPUT(slotMapping, ge::TensorType::ALL())
    .REQUIRED_ATTR(kvBlockSize, Int)
    .REQUIRED_ATTR(slotMappingFormat, Int)
    .REQUIRED_ATTR(cmpRatio, Int)
    .REQUIRED_ATTR(actualNumReqs, Int)
    .OP_END_FACTORY_REG(CompressorMetadata);

}

#endif
