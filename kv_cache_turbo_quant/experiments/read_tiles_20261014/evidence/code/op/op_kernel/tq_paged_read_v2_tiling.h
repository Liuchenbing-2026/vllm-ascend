#ifndef TQ_PAGED_READ_V2_TILING_H
#define TQ_PAGED_READ_V2_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct alignas(8) TqPagedReadV2TilingData {
    uint32_t jobs;
    uint32_t heads;
    uint32_t blockTokens;
    uint32_t tableWidth;
    uint32_t tileRows;
    uint32_t reserved;
    uint64_t cacheBytes;
    uint64_t outputElements;
};
#endif
