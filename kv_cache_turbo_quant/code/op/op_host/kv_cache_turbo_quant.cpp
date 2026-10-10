#include "../op_kernel/kv_cache_turbo_quant_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matmul/matmul_tiling.h"

namespace {
constexpr uint32_t BYTE_BITS = 8;
// supported head_dim values: 128 (64 rows/AIV/tile) and 256 (32 rows/AIV/tile);
// the per-tile fp32 element count 16384 (64KB per workspace buf) is dimension-invariant.
constexpr uint32_t HALF_ELEMS_FIXED = 8192;
constexpr uint64_t WS_MM_SLACK = 1024 * 1024;  // reserved front region for matmul internal scratch
constexpr uint64_t WS_BUF_BYTES = 2 * HALF_ELEMS_FIXED * sizeof(float);
constexpr uint64_t WS_PAIR_STRIDE = 8 * WS_BUF_BYTES;  // U0 U1 Y0 Y1 R0 R1 P0 P1

// two Matmul objects statically share one AIC's L1/L0A/L0B/L0C at registration time:
// constrain each object so both fit (L0A/L0B are 64KB total on 910B, L0C is 128KB)
constexpr int32_t MM_L1_LIMIT = 224 * 1024;
constexpr int32_t MM_L0C_LIMIT = 64 * 1024;
constexpr int32_t MM_L0A_LIMIT = 32 * 1024;
constexpr int32_t MM_L0B_LIMIT = 32 * 1024;

ge::graphStatus BuildMmTiling(const platform_ascendc::PlatformAscendC &platform, int32_t m, int32_t n, int32_t k,
    AscendC::tiling::TCubeTiling &cubeTiling)
{
    matmul_tiling::MatmulApiTiling mmTiling(platform);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
        false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
        true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    if (mmTiling.SetShape(m, n, k) != 0 || mmTiling.SetOrgShape(m, n, k) != 0) {
        return ge::GRAPH_FAILED;
    }
    mmTiling.SetBias(false);
    mmTiling.SetBufferSpace(MM_L1_LIMIT, MM_L0C_LIMIT, MM_L0A_LIMIT, MM_L0B_LIMIT);
    return mmTiling.GetTiling(cubeTiling) == -1 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    KvCacheTurboQuantTilingData *tiling = context->GetTilingData<KvCacheTurboQuantTilingData>();

    const gert::StorageShape *kvShape = context->GetInputShape(0);
    const gert::StorageShape *rotShape = context->GetInputShape(1);
    const gert::StorageShape *qjlShape = context->GetInputShape(2);
    if (kvShape == nullptr || rotShape == nullptr || qjlShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (kvShape->GetStorageShape().GetDimNum() != 3 || rotShape->GetStorageShape().GetDimNum() != 2 ||
        qjlShape->GetStorageShape().GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = kvShape->GetStorageShape().GetDim(0);
    const int64_t numKvHeads = kvShape->GetStorageShape().GetDim(1);
    const int64_t headDim = kvShape->GetStorageShape().GetDim(2);
    const int64_t rotR = rotShape->GetStorageShape().GetDim(0);
    const int64_t rotC = rotShape->GetStorageShape().GetDim(1);
    const int64_t qjlDim = qjlShape->GetStorageShape().GetDim(0);
    const int64_t qjlC = qjlShape->GetStorageShape().GetDim(1);

    int64_t mseBits = 3;
    if (context->GetAttrs() != nullptr && context->GetAttrs()->GetInt(0) != nullptr) {
        mseBits = *context->GetAttrs()->GetInt(0);
    }
    if ((headDim != 128 && headDim != 256) || rotR != headDim || rotC != headDim || qjlC != headDim) {
        return ge::GRAPH_FAILED;
    }
    if (mseBits != 2 && mseBits != 3 && mseBits != 4) {
        return ge::GRAPH_FAILED;
    }
    // supports qjl_dim == head_dim only
    if (qjlDim != headDim) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalRows = static_cast<uint64_t>(numTokens) * static_cast<uint64_t>(numKvHeads);
    if (totalRows == 0 || totalRows > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t halfRows = HALF_ELEMS_FIXED / static_cast<uint32_t>(headDim);
    const uint32_t tileRows = 2 * halfRows;
    const uint32_t numTiles = (static_cast<uint32_t>(totalRows) + tileRows - 1) / tileRows;
    const uint32_t usedPairs = numTiles < aicNum ? numTiles : aicNum;

    if (BuildMmTiling(platform, static_cast<int32_t>(halfRows), static_cast<int32_t>(headDim), static_cast<int32_t>(headDim),
            tiling->mm1Tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (BuildMmTiling(platform, static_cast<int32_t>(halfRows), static_cast<int32_t>(qjlDim), static_cast<int32_t>(headDim),
            tiling->mm2Tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalRows = static_cast<uint32_t>(totalRows);
    tiling->headDim = static_cast<uint32_t>(headDim);
    tiling->qjlDim = static_cast<uint32_t>(qjlDim);
    tiling->mseBits = static_cast<uint32_t>(mseBits);
    tiling->idxBytesPerRow = static_cast<uint32_t>(headDim * mseBits / BYTE_BITS);
    tiling->qjlBytesPerRow = static_cast<uint32_t>(qjlDim / BYTE_BITS);
    tiling->usedPairs = usedPairs;
    tiling->numTiles = numTiles;
    // The kernel receives the raw workspace pointer; the first libApiSize bytes
    // (16MB RESERVED_WORKSPACE on dav_c220) belong to the runtime/libApi (FFTS+
    // dispatch, KFC system messages via GetSysWorkSpacePtr()). All user regions
    // must therefore be addressed through GetUserWorkspace() kernel-side, and the
    // allocation must include the reserved region up front.
    const uint64_t libApiSize = platform.GetLibApiWorkSpaceSize();
    tiling->wsPairOffset = static_cast<uint32_t>(WS_MM_SLACK);
    tiling->wsPairStride = static_cast<uint32_t>(WS_PAIR_STRIDE);

    context->SetBlockDim(usedPairs);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = libApiSize + tiling->wsPairOffset + static_cast<size_t>(usedPairs) * WS_PAIR_STRIDE;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *kvShape = context->GetInputShape(0);
    const gert::Shape *qjlShape = context->GetInputShape(2);
    if (kvShape == nullptr || qjlShape == nullptr || kvShape->GetDimNum() != 3 || qjlShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    int64_t mseBits = 3;
    if (context->GetAttrs() != nullptr && context->GetAttrs()->GetAttrPointer<int64_t>(0) != nullptr) {
        mseBits = *context->GetAttrs()->GetAttrPointer<int64_t>(0);
    }
    const int64_t numTokens = kvShape->GetDim(0);
    const int64_t numKvHeads = kvShape->GetDim(1);
    const int64_t headDim = kvShape->GetDim(2);
    const int64_t qjlDim = qjlShape->GetDim(0);

    gert::Shape *idxShape = context->GetOutputShape(0);
    gert::Shape *qjlShapeOut = context->GetOutputShape(1);
    gert::Shape *normShape = context->GetOutputShape(2);
    gert::Shape *gammaShape = context->GetOutputShape(3);
    if (idxShape == nullptr || qjlShapeOut == nullptr || normShape == nullptr || gammaShape == nullptr) {
        return GRAPH_FAILED;
    }
    *idxShape = gert::Shape({numTokens, numKvHeads, headDim * mseBits / BYTE_BITS});
    *qjlShapeOut = gert::Shape({numTokens, numKvHeads, qjlDim / BYTE_BITS});
    *normShape = gert::Shape({numTokens, numKvHeads});
    *gammaShape = gert::Shape({numTokens, numKvHeads});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    context->SetOutputDataType(1, ge::DT_UINT8);
    context->SetOutputDataType(2, ge::DT_BF16);
    context->SetOutputDataType(3, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class KvCacheTurboQuant : public OpDef {
public:
    explicit KvCacheTurboQuant(const char *name) : OpDef(name)
    {
        this->Input("kv_vectors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rotation_matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("qjl_matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("quant_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("quant_qjl")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("quant_norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("quant_gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mse_bits").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(KvCacheTurboQuant);
} // namespace ops
