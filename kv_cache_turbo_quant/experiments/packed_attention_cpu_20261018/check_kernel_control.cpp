// New device-control rules only. This does not include kernel_operator.h,
// emulate AscendC, compile the kernel, or repeat the historical CPU math suite.
#include "../../code/op/op_kernel/tq_packed_attention_core.h"
#include "../../code/op/op_host/tq_packed_attention_plan.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Require(bool condition, const char *why)
{
    if (!condition) throw std::runtime_error(why);
}

static void AllocationAgreement()
{
    for (uint32_t tile : {16U, 32U, 64U, 128U}) {
        for (uint32_t group : {1U, 2U, 8U, 31U}) {
            for (uint32_t scratch : {0U, 1U, 32768U, 32769U}) {
                const auto host = tq_packed_attention::Buffers(tile, group, scratch);
                uint64_t offset = 0;
                for (uint32_t i = 0; i < tq_packed_core::BufferCount; ++i) {
                    const uint64_t bytes = tq_packed_core::BufferBytes(i, tile, group, scratch);
                    Require(offset % 32 == 0 && bytes % 32 == 0, "device arena alignment");
                    Require(bytes == host[i].bytes, "new device allocation differs from host budget");
                    offset += bytes;
                }
                Require(offset == tq_packed_attention::UbBytes(tile, group, scratch), "arena total");
                Require(tq_packed_core::BufferBytes(5, tile, group, scratch) >= 256 + 512,
                        "separate BF16 decode source and FP32 widening destination");
                Require(tq_packed_core::BufferBytes(12, tile, group, scratch) >= 96 + 32,
                        "metadata/status/reduction/norm regions must remain disjoint");
            }
        }
    }
}

static TqPackedAttentionTilingData Data()
{
    TqPackedAttentionTilingData d{};
    d.physicalPages = 3;
    d.pageTokens = 17;
    d.kvHeads = 2;
    d.tileTokens = 16;
    d.tableWidth = 4;
    d.keyBytes = static_cast<uint64_t>(d.physicalPages) * d.pageTokens * d.kvHeads * 66;
    d.valueBytes = d.keyBytes;
    return d;
}

static void PageSchedule()
{
    const auto d = Data();
    // The final physical row is an actual active row. Index -1 is inactive.
    const std::vector<int32_t> table{1, 0, 2, -1};
    const uint32_t length = 51;
    std::vector<uint64_t> actual;
    uint32_t segments = 0;
    for (uint32_t position = 0; position < length;) {
        const auto physical = table.at(position / d.pageTokens);
        tq_packed_core::Span span{};
        Require(tq_packed_core::PageSpan(d, position, length, physical, 1, span), "valid page span");
        Require(span.rows && span.rows <= d.tileTokens, "tile size");
        Require(position / d.pageTokens == (position + span.rows - 1) / d.pageTokens,
                "a DMA span crossed a page");
        for (uint32_t r = 0; r < span.rows; ++r) {
            const uint64_t address = span.offset + static_cast<uint64_t>(r) * d.kvHeads * 66;
            actual.push_back(address);
            Require(address + 66 <= d.keyBytes, "66-byte block crossed final cache byte");
        }
        ++segments;
        position += span.rows;
    }
    Require(segments == 6 && actual.size() == length, "page/tile tail schedule");
    for (uint32_t token = 0; token < length; ++token) {
        // Independent token-by-token layout oracle, without span scheduling.
        const uint64_t physical = static_cast<uint32_t>(table.at(token / 17));
        const uint64_t row = physical * 34 + (token % 17) * 2 + 1;
        Require(actual[token] == row * 66, "permuted page address");
    }
    Require(actual.back() + 66 == d.keyBytes, "test must reach the last exact physical byte");
}

static void InvalidAndWideAddressing()
{
    auto d = Data();
    tq_packed_core::Span span{};
    Require(!tq_packed_core::PageSpan(d, 0, 1, -1, 0, span), "negative active page");
    Require(!tq_packed_core::PageSpan(d, 0, 1, 3, 0, span), "out-of-range active page");
    Require(!tq_packed_core::PageSpan(d, 0, 1, 0, 2, span), "invalid KV head");
    Require(!tq_packed_core::ValidLength(0, 4, 17), "zero nonempty sequence length");
    Require(!tq_packed_core::ValidLength(-1, 4, 17), "negative sequence length");
    Require(tq_packed_core::ValidLength(68, 4, 17) && !tq_packed_core::ValidLength(69, 4, 17),
            "table coverage bound");
    --d.valueBytes;
    Require(!tq_packed_core::PageSpan(d, 50, 51, 2, 1, span), "separate V bound must be checked");
    d = Data();
    d.physicalPages = 4000000;
    d.pageTokens = 128;
    d.keyBytes = static_cast<uint64_t>(d.physicalPages) * d.pageTokens * d.kvHeads * 66;
    d.valueBytes = d.keyBytes;
    Require(tq_packed_core::PageSpan(d, 127, 128, 3999999, 1, span), "wide span");
    Require(span.offset > UINT32_MAX && span.rows == 1 && span.offset + 66 == d.keyBytes,
            "cache addressing narrowed below uint64 or crossed final row");
}

static void NormClassifications()
{
    for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
        const uint32_t word = bits << 16U;
        float value;
        static_assert(sizeof(value) == sizeof(word), "FP32 host check requires 32-bit float");
        std::memcpy(&value, &word, sizeof(value));
        Require(tq_packed_core::FiniteNorm(static_cast<uint16_t>(bits)) == std::isfinite(value),
                "device norm classifier differs from IEEE float classification");
    }
}

int main()
{
    try {
        AllocationAgreement();
        PageSchedule();
        InvalidAndWideAddressing();
        NormClassifications();
        std::cout << "{\"new_control_checks\":4,\"passed\":true,"
                     "\"allocation_configurations\":64,\"norm_bit_patterns\":65536,"
                     "\"cann_kernel_compiled\":false,\"npu_tested\":false}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
