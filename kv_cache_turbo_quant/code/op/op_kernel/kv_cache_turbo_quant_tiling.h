#ifndef KV_CACHE_TURBO_QUANT_TILING_H
#define KV_CACHE_TURBO_QUANT_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) KvCacheTurboQuantTilingData {
    uint32_t totalRows;       // num_tokens * num_kv_heads
    uint32_t headDim;         // 128 or 256
    uint32_t qjlDim;          // rows of qjl_matrix (must equal headDim)
    uint32_t mseBits;         // 2 / 3 / 4
    uint32_t idxBytesPerRow;  // headDim * mseBits / 8
    uint32_t qjlBytesPerRow;  // qjlDim / 8
    uint32_t usedPairs;       // launched AIC count (AIV count = 2x)
    uint32_t numTiles;        // ceil(totalRows / tileRows), tileRows = 16384/headDim
    uint32_t wsPairOffset;    // byte offset of per-pair region in workspace
    uint32_t wsPairStride;    // bytes per pair in workspace
    AscendC::tiling::TCubeTiling mm1Tiling;  // U[M,headDim] @ rot^T[headDim,headDim]
    AscendC::tiling::TCubeTiling mm2Tiling;  // Rn[M,headDim] @ qjl^T[headDim,qjlDim]
};
#pragma pack(pop)

#endif // KV_CACHE_TURBO_QUANT_TILING_H
