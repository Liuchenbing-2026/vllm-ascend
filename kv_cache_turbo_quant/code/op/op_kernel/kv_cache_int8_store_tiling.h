#ifndef KV_CACHE_INT8_STORE_TILING_H
#define KV_CACHE_INT8_STORE_TILING_H
#include <cstdint>

#pragma pack(push, 8)
struct alignas(8) KvCacheInt8StoreTilingData {
    uint32_t numTokens;    // rows of x
    uint32_t numKvHeads;   // H
    uint32_t headDim;      // 64 / 128 / 256 / 512
    uint32_t numBlocks;    // cache dim0 (from attr)
    uint32_t blockSize;    // cache dim1 (from attr)
    uint32_t usedCores;    // launched AIV count
};
#pragma pack(pop)

#endif // KV_CACHE_INT8_STORE_TILING_H