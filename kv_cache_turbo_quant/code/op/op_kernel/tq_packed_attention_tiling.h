#ifndef TQ_PACKED_ATTENTION_TILING_H
#define TQ_PACKED_ATTENTION_TILING_H

#include <cstddef>
#include <cstdint>

// Device ABI v1. No CANN dependency: host and device include this exact POD.
struct alignas(8) TqPackedAttentionTilingData {
    uint32_t abiVersion;
    uint32_t batch;
    uint32_t queryHeads;
    uint32_t kvHeads;
    uint32_t headDim;
    uint32_t pageTokens;
    uint32_t tableWidth;
    uint32_t physicalPages;
    uint32_t tileTokens;
    uint32_t groupHeads;
    uint32_t workItems;
    uint32_t blockDim;
    uint32_t ubPlanBytes;
    uint32_t apiScratchBytes;
    float scale;
    uint32_t flags;
    uint64_t keyBytes;
    uint64_t valueBytes;
};

static_assert(sizeof(TqPackedAttentionTilingData) == 80, "TQ ABI size changed");
static_assert(alignof(TqPackedAttentionTilingData) == 8, "TQ ABI alignment changed");
static_assert(offsetof(TqPackedAttentionTilingData, scale) == 56, "TQ scale offset changed");
static_assert(offsetof(TqPackedAttentionTilingData, keyBytes) == 64, "TQ pointer bound offset changed");

enum class TqPackedAttentionStatus : int32_t {
    Ok = 0,
    InvalidLength = 1,
    InvalidPhysicalPage = 2,
    NonFiniteNorm = 3,
};

// Planned device entry (implementation intentionally not supplied by this header):
// tq_packed_attention(query_rot, key, value, table, lengths, centroids,
//                     attention_rot, status, workspace, tiling)
#endif
