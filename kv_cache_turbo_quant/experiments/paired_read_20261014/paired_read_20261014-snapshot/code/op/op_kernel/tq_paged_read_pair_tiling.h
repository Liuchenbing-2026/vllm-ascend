#ifndef TQ_PAGED_READ_TILING_H
#define TQ_PAGED_READ_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct alignas(8) TqPagedReadPairTilingData {
    uint32_t jobs;
    uint32_t heads;
    uint32_t blockTokens;
    uint32_t tableWidth;
    uint64_t cacheBytes;
    uint64_t outputElements;
};
#endif
