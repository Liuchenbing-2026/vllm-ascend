
#include "msa_dist_top_k_tiling.h"
#include "msa_dist_top_k.h"
#include "register/op_def_registry.h"
#include <sstream>
namespace optiling {

namespace {

}

bool MsaDistTopKTiling::IsCapable()
{
    return true;
}

ge::graphStatus MsaDistTopKTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MsaDistTopKTiling::GetShapeAttrsInfo() {
    inputParams_.opName = context_->GetNodeName();
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MsaDistTopKTiling::DoOpTiling() {
    if (!this->SetPlatformInfoForTiling()) {
        return ge::GRAPH_FAILED;
    }

    // index_q: [B, G, dim]; seq_len: [B]; block_table: [B, MAXB].
    uint32_t batch = GetShape(SEQ_LEN_INPUT_INDEX).GetDim(0);          // B
    uint32_t head = GetShape(INDEX_Q_INPUT_INDEX).GetDim(1);          // G (=1 for decode)
    uint32_t batchN = batch * head;
    uint32_t blockCount = GetShape(BLOCK_TABLE_INPUT_INDEX).GetDim(1); // MAXB (logical blocks)

    // dimension is the index head dim of iq (last dim, =128 for MSA).
    uint32_t dimNum = GetShape(INDEX_Q_INPUT_INDEX).GetDimNum();
    uint32_t dimension = GetShape(INDEX_Q_INPUT_INDEX).GetDim(dimNum - 1);

    // maxSeqLen: padded max KV length, multiple of blockSize. Derived from the
    // number of logical blocks in the block_table.
    seqLen_ = ops::CeilDiv(blockCount * blockSize_, blockSize_) * blockSize_;

    uint32_t maxK = topk_;
    uint32_t topkTotal = topk_ + localBlocks_ + initBlocks_;
    uint32_t topKInnerSize = ops::CeilDiv(blockCount, TOP_K_ALIGN_NUM) * TOP_K_ALIGN_NUM;
    uint32_t matmulResultSize = batchN * seqLen_;

    uint32_t reducedBatch = batchN;
    uint32_t usedCoreNum = std::min(reducedBatch, coreNum_);

    tilingData_.params.set_usedCoreNum(usedCoreNum);
    tilingData_.params.set_batch(batch);
    tilingData_.params.set_head(head);
    tilingData_.params.set_batchN(batchN);
    tilingData_.params.set_dimension(dimension);
    tilingData_.params.set_maxSeqLen(seqLen_);
    tilingData_.params.set_maxK(maxK);
    tilingData_.params.set_blockCount(blockCount);
    tilingData_.params.set_topkTotal(topkTotal);
    tilingData_.params.set_localBlocks(localBlocks_);
    tilingData_.params.set_initBlocks(initBlocks_);
    tilingData_.params.set_topKInnerSize(topKInnerSize);
    tilingData_.params.set_matmulResultSize(matmulResultSize);

    this->SetMatmulTiling();
    this->SetTopKTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MsaDistTopKTiling::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

uint64_t MsaDistTopKTiling::GetTilingKey() { return 10; }

ge::graphStatus MsaDistTopKTiling::GetWorkspaceSize() {
    uint64_t *workspaces = context_->GetWorkspaceSizes(1);
    uint64_t sysWorkspaceSize = WORKSIZE;
    /* usrWorkspaceSize = per-cube-core gathered keys (bf16) + matmul scores (half). */
    uint64_t keyBytes = static_cast<uint64_t>(tilingData_.params.get_usedCoreNum()) *
                        tilingData_.params.get_maxSeqLen() *
                        tilingData_.params.get_dimension() * sizeof(uint16_t); /* 2: sizeof(bf16) */
    uint64_t scoreBytes = static_cast<uint64_t>(tilingData_.params.get_batchN()) *
                          tilingData_.params.get_maxSeqLen() * sizeof(uint16_t); /* 2: sizeof(half) */
    uint64_t usrWorkspaceSize = keyBytes + scoreBytes;
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MsaDistTopKTiling::PostTiling() {
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    auto blockDim = tilingData_.params.get_usedCoreNum();
    context_->SetBlockDim(blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MsaDistTopKTiling::Reset() {
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    inputParams_.mSize = 0UL;
    inputParams_.kSize = 0UL;
    inputParams_.nSize = 0UL;
    inputParams_.queryDtype = ge::DT_BF16;
    inputParams_.keyDtype = ge::DT_BF16;
    inputParams_.seqLenDtype = ge::DT_INT32;
    inputParams_.indicesDtype = ge::DT_INT32;
    inputParams_.libApiWorkSpaceSize = 0U;
    inputParams_.opName = nullptr;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
}

void MsaDistTopKTiling::SetMatmulTiling() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
    tiling.SetDim(1);
    // A = iq[M=head, K=dim] NOT transposed (row-major [M,K], matches kernel
    // AMatmulType isTrans=false); B = keys[N=maxSeqLen, K=dim] transposed; C = scores half.
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, true);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    uint32_t M = tilingData_.params.get_head();
    uint32_t N = tilingData_.params.get_maxSeqLen();
    uint32_t K = tilingData_.params.get_dimension();
    tiling.SetShape(M, N, K);
    tiling.SetSingleShape(M, N, K);
    tiling.SetOrgShape(M, N, K);
    tiling.SetBias(false);
    tiling.GetTiling(tilingData_.matmulTiling); /* if ret = -1, get tiling failed */
}

void MsaDistTopKTiling::SetTopKTiling() {
    uint32_t inner = tilingData_.params.get_topKInnerSize();
    uint32_t outer = 1;
    uint32_t k = std::min(tilingData_.params.get_blockCount(), tilingData_.params.get_maxK());
    uint32_t maxSize = 0;
    uint32_t minSize = 0;
    uint32_t dTypeSize = 2; /* 2:size of float16 */
    const bool IS_REUSESOURCE = false;
    const bool IS_INITINDEX = true;
    const bool IS_LARGEST = true;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    tilingData_.params.set_outer(outer);
    AscendC::TopKTilingFunc(ascendcPlatform, inner, outer, k, dTypeSize, IS_INITINDEX, AscendC::TopKMode::TOPK_NORMAL, IS_LARGEST, tilingData_.topkTiling);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, inner, outer, IS_REUSESOURCE, IS_INITINDEX, AscendC::TopKMode::TOPK_NORMAL, IS_LARGEST, dTypeSize, maxSize, minSize);
}

const gert::Shape MsaDistTopKTiling::GetShape(const size_t index) {
    return context_->GetInputShape(index)->GetStorageShape();
}

const gert::Shape MsaDistTopKTiling::GetOutShape(const size_t index) {
    return context_->GetOutputShape(index)->GetStorageShape();
}

const uint32_t MsaDistTopKTiling::GetInputAttrData(const size_t index, const uint32_t defaultValue) {
    if (auto attrPtr = context_->GetAttrs()) {
        const int64_t* p = attrPtr->GetInt(index);
        if (p != nullptr) {
            return static_cast<uint32_t>(*p);
        }
    }
    return defaultValue;
}

bool MsaDistTopKTiling::SetPlatformInfoForTiling() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAic();
    return true;
}

}
