#ifndef KV_TQ_READ_BENCH_TILING_H
#define KV_TQ_READ_BENCH_TILING_H
#include <cstdint>

#pragma pack(push, 8)
struct alignas(8) KvTqReadBenchTilingData {
    uint32_t mode;         // 0 = packed read floor, 1 = packed read + dequant, 2 = bf16 read floor
    uint32_t rows;         // total token-head rows
    uint32_t usedCores;    // launched AIV count
    uint32_t pad;
};
#pragma pack(pop)

#endif // KV_TQ_READ_BENCH_TILING_H