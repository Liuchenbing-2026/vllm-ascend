#ifndef MSA_DIST_TOP_K_H
#define MSA_DIST_TOP_K_H


#include "msa_dist_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
class MsaDistTopKTiling {
public:
    // from parent class
    gert::TilingContext *context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_{0};

    // from child class
    MsaDistTopKMatmulInfo inputParams_;
    uint32_t libApiWorkSpaceSize_ = 0;
    uint32_t coreNum_ = 1;
    const char *opName_ = "";
    int32_t dtypeByte_ = 2; /* 2: size of float16 */
    MsaDistTopKTilingData tilingData_;
    bool compileInfoInit_ = false;
    uint32_t seqLen_ = 1;

    MsaDistTopKTiling(gert::TilingContext *context) : context_(context) {
        InitAttrParam();
    }

    bool IsCapable();
    // 1. Obtain platform information such as CoreNum, UB/L1/L0C resource size
    ge::graphStatus GetPlatformInfo();
    // 2. Obtain INPUT/OUTPUT/ATTR information
    ge::graphStatus GetShapeAttrsInfo();
    // 3. Calculate data split TilingData
    ge::graphStatus DoOpTiling();
    // 4. Calculate TilingData for high-level API
    ge::graphStatus DoLibApiTiling();
    // 5. Calculate TilingKey
    uint64_t GetTilingKey();
    // 6. Calculate Workspace size
    ge::graphStatus GetWorkspaceSize();
    // 7. Save Tiling data
    ge::graphStatus PostTiling();

    void Reset();
    void SetMatmulTiling();
    void SetTopKTiling();
    bool SetPlatformInfoForTiling();
    const gert::Shape GetShape(const size_t index);
    // Get input attr data
    const uint32_t GetInputAttrData(const size_t index, const uint32_t defaultValue);
    // output shape
    const gert::Shape GetOutShape(const size_t index);

    // Initialize block_size / topk / local / init from attrs.
    const void InitAttrParam() {
        blockSize_ = GetInputAttrData(0, DEFAULT_BLOCK_SIZE);
        topk_ = GetInputAttrData(1, DEFAULT_TOPK);
        localBlocks_ = GetInputAttrData(2, DEFAULT_LOCAL_BLOCKS);
        initBlocks_ = GetInputAttrData(3, DEFAULT_INIT_BLOCKS);
    }

    // attr-derived parameters
    uint32_t blockSize_ = 128;
    uint32_t topk_ = 16;
    uint32_t localBlocks_ = 1;
    uint32_t initBlocks_ = 0;

    uint32_t DIMENSION = 128;
    uint64_t WORKSIZE = 16 * 1024 * 1024;
    uint32_t TOP_K_ALIGN_NUM = 32;
    uint32_t DEFAULT_BLOCK_SIZE = 128;
    uint32_t DEFAULT_TOPK = 16;
    uint32_t DEFAULT_LOCAL_BLOCKS = 1;
    uint32_t DEFAULT_INIT_BLOCKS = 0;

    // op input indices (mirror OpDef order)
    static constexpr size_t INDEX_Q_INPUT_INDEX = 0;
    static constexpr size_t INDEX_K_CACHE_INPUT_INDEX = 1;
    static constexpr size_t SEQ_LEN_INPUT_INDEX = 2;
    static constexpr size_t BLOCK_TABLE_INPUT_INDEX = 3;
    static constexpr size_t INDICES_IN_INPUT_INDEX = 4;
};
}

#endif
