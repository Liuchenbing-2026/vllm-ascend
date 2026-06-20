#ifndef MSA_DIST_TOP_K_TILING_H
#define MSA_DIST_TOP_K_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MsaDistTopKTilingParams)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, head);
    TILING_DATA_FIELD_DEF(uint32_t, batchN);
    TILING_DATA_FIELD_DEF(uint32_t, dimension);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, maxK);
    TILING_DATA_FIELD_DEF(uint32_t, blockCount);
    TILING_DATA_FIELD_DEF(uint32_t, topkTotal);
    TILING_DATA_FIELD_DEF(uint32_t, localBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, initBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, topKInnerSize);
    TILING_DATA_FIELD_DEF(uint32_t, matmulResultSize);
    TILING_DATA_FIELD_DEF(uint32_t, outer);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MsaDistTopKTilingParamsOp, MsaDistTopKTilingParams)

BEGIN_TILING_DATA_DEF(MsaDistTopKTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(MsaDistTopKTilingParams, params);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MsaDistTopK, MsaDistTopKTilingData)
REGISTER_TILING_DATA_CLASS(MsaDistTopKTilingDataOp, MsaDistTopKTilingData)

struct MsaDistTopKMatmulInfo {
    bool transA = false;
    bool transB = false;
    bool hasBias = false;
    uint64_t mSize = 0UL;
    uint64_t kSize = 0UL;
    uint64_t nSize = 0UL;
    ge::DataType queryDtype = ge::DT_BF16;
    ge::DataType keyDtype = ge::DT_BF16;
    ge::DataType seqLenDtype = ge::DT_INT32;
    ge::DataType indicesDtype = ge::DT_INT32;
    int64_t outDtype = 0L;
    uint32_t libApiWorkSpaceSize = 0U;
    uint64_t bf16ExtreWorkSpaceSize = 0UL;
    const char *opName = nullptr;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format cFormat = ge::FORMAT_ND;
};

struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};
// using MsaDistTopKCompileInfo = gert::GemmCompileInfo;
}

#endif
