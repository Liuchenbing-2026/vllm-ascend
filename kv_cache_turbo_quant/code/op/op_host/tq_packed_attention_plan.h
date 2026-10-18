#ifndef TQ_PACKED_ATTENTION_PLAN_H
#define TQ_PACKED_ATTENTION_PLAN_H

#include "../op_kernel/tq_packed_attention_tiling.h"
#include <array>
#include <cmath>
#include <limits>

namespace tq_packed_attention {
struct BufferBudget {
    const char *name;
    uint64_t bytes;
};

inline uint64_t Align32(uint64_t bytes) { return (bytes + 31U) / 32U * 32U; }

inline std::array<BufferBudget, 14> Buffers(uint32_t tile, uint32_t group, uint32_t sdkScratch)
{
    const uint64_t t = tile;
    const uint64_t g = group;
    // 66-byte GM row -> 96-byte aligned UB row; two queue slots per K/V.
    return {{{"packed_k_two_slots", Align32(2 * t * 96)},
             {"packed_v_two_slots", Align32(2 * t * 96)},
             {"decoded_k_fp32", Align32(t * 128 * 4)},
             {"decoded_v_bf16", Align32(t * 128 * 2)},
             {"phase_scratch", Align32(t * 128 * 4)},
             {"decode_indices_or_norm_chunk16", Align32(16 * 128 * 4)},
             {"query_bf16", Align32(g * 128 * 2)},
             {"query_fp32", Align32(g * 128 * 4)},
             {"online_numerator_fp32", Align32(g * 128 * 4)},
             {"online_max_sum_carry_fp32", Align32(g * 4 * 4)},
             {"single_query_head_scores_fp32", Align32(t * 4)},
             {"centroids_bf16_and_fp32", Align32(16 * 2) + Align32(16 * 4)},
             {"control_and_dma_metadata", 4 * 32},
             {"sdk_scratch_reservation", Align32(sdkScratch)}}};
}

inline uint64_t UbBytes(uint32_t tile, uint32_t group, uint32_t sdkScratch)
{
    uint64_t total = 0;
    for (const auto &buffer : Buffers(tile, group, sdkScratch)) total += buffer.bytes;
    return total;
}

inline bool Multiply(uint64_t a, uint64_t b, uint64_t &out)
{
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}

inline bool MakePlan(uint32_t batch, uint32_t queryHeads, uint32_t kvHeads,
                     uint32_t physicalPages, uint32_t pageTokens, uint32_t tableWidth,
                     uint64_t platformUbBytes, uint32_t platformAivCores,
                     uint32_t sdkScratch, float scale, TqPackedAttentionTilingData &out)
{
    if (!queryHeads || !kvHeads || queryHeads % kvHeads || !physicalPages || !pageTokens ||
        (batch && !tableWidth) || !platformAivCores || !platformUbBytes || !std::isfinite(scale)) {
        return false;
    }
    const uint64_t jobs = static_cast<uint64_t>(batch) * kvHeads;
    if (jobs > std::numeric_limits<uint32_t>::max()) return false;
    uint64_t cacheBytes = physicalPages;
    for (uint64_t factor : {static_cast<uint64_t>(pageTokens),
                            static_cast<uint64_t>(kvHeads), uint64_t(66)}) {
        if (!Multiply(cacheBytes, factor, cacheBytes)) return false;
    }
    const uint32_t group = queryHeads / kvHeads;
    uint32_t tile = 0;
    for (uint32_t candidate : {128U, 64U, 32U, 16U}) {
        if (UbBytes(candidate, group, sdkScratch) <= platformUbBytes &&
            UbBytes(candidate, group, sdkScratch) <= std::numeric_limits<uint32_t>::max()) {
            tile = candidate;
            break;
        }
    }
    if (!tile) return false;
    out = {};
    out.abiVersion = 1;
    out.batch = batch;
    out.queryHeads = queryHeads;
    out.kvHeads = kvHeads;
    out.headDim = 128;
    out.pageTokens = pageTokens;
    out.tableWidth = tableWidth;
    out.physicalPages = physicalPages;
    out.tileTokens = tile;
    out.groupHeads = group;
    out.workItems = static_cast<uint32_t>(jobs);
    out.blockDim = jobs ? (jobs < platformAivCores ? static_cast<uint32_t>(jobs) : platformAivCores) : 1;
    out.ubPlanBytes = static_cast<uint32_t>(UbBytes(tile, group, sdkScratch));
    out.apiScratchBytes = sdkScratch;
    out.scale = scale;
    out.keyBytes = cacheBytes;
    out.valueBytes = cacheBytes;
    return true;
}
}  // namespace tq_packed_attention
#endif
