#ifndef TQ_PACKED_ATTENTION_CORE_H
#define TQ_PACKED_ATTENTION_CORE_H

#include "tq_packed_attention_tiling.h"

// The device supplies __aicore__ inline; the CPU check compiles ONLY these
// integer addressing/budget rules, never a replacement AscendC runtime.
#ifndef TQ_PACKED_CORE_INLINE
#define TQ_PACKED_CORE_INLINE inline
#define TQ_PACKED_CORE_LOCAL_INLINE
#endif

namespace tq_packed_core {
constexpr uint32_t HeadDim = 128;
constexpr uint32_t PackedRow = 66;
constexpr uint32_t LocalRow = 96;
constexpr uint32_t BufferCount = 14;

TQ_PACKED_CORE_INLINE uint64_t Align32(uint64_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}

// Same order and sizes as the existing host Buffers() contract. The new CPU
// check compares these actual device allocation rules with that host function.
TQ_PACKED_CORE_INLINE uint64_t BufferBytes(uint32_t index, uint32_t tile,
                                         uint32_t group, uint32_t reservation)
{
    const uint64_t t = tile;
    const uint64_t g = group;
    switch (index) {
        case 0: case 1: return Align32(2 * t * LocalRow);
        case 2: case 4: return Align32(t * HeadDim * 4);
        case 3: return Align32(t * HeadDim * 2);
        case 5: return Align32(16 * HeadDim * 4);
        case 6: return Align32(g * HeadDim * 2);
        case 7: case 8: return Align32(g * HeadDim * 4);
        case 9: return Align32(g * 4 * 4);
        case 10: return Align32(t * 4);
        case 11: return Align32(16 * 2) + Align32(16 * 4);
        case 12: return 4 * 32;
        case 13: return Align32(reservation);
        default: return 0;
    }
}

TQ_PACKED_CORE_INLINE bool ValidLength(int32_t length, uint32_t width, uint32_t pageTokens)
{
    return length > 0 && static_cast<uint64_t>(length) <= static_cast<uint64_t>(width) * pageTokens;
}

TQ_PACKED_CORE_INLINE bool FiniteNorm(uint16_t bits)
{
    return (bits & 0x7f80U) != 0x7f80U;
}

// No pointer is formed from an unchecked physical index. A span never crosses
// a logical page, and cache bound checks cover the final row's exact 66 bytes.
struct Span {
    uint64_t offset;
    uint32_t rows;
};

TQ_PACKED_CORE_INLINE bool PageSpan(const TqPackedAttentionTilingData &d,
                                  uint32_t position, uint32_t length, int32_t physical,
                                  uint32_t kvHead, Span &span)
{
    if (physical < 0 || static_cast<uint32_t>(physical) >= d.physicalPages ||
        kvHead >= d.kvHeads || !d.pageTokens || !d.tileTokens || position >= length) return false;
    const uint32_t inPage = position % d.pageTokens;
    uint32_t rows = length - position;
    if (rows > d.tileTokens) rows = d.tileTokens;
    if (rows > d.pageTokens - inPage) rows = d.pageTokens - inPage;
    // Host MakePlan has already checked P*S*H*66 for uint64 overflow.
    const uint64_t row = (static_cast<uint64_t>(physical) * d.pageTokens + inPage) * d.kvHeads + kvHead;
    const uint64_t first = row * PackedRow;
    const uint64_t last = first + static_cast<uint64_t>(rows - 1) * d.kvHeads * PackedRow;
    if (last > d.keyBytes || PackedRow > d.keyBytes - last ||
        last > d.valueBytes || PackedRow > d.valueBytes - last) return false;
    span = {first, rows};
    return true;
}
}  // namespace tq_packed_core

#ifdef TQ_PACKED_CORE_LOCAL_INLINE
#undef TQ_PACKED_CORE_INLINE
#undef TQ_PACKED_CORE_LOCAL_INLINE
#endif
#endif
