// Local ABI/budget probe only; does not include CANN or launch an NPU.
#include "../../code/op/op_host/tq_packed_attention_plan.h"
#include <cerrno>
#include <cstdlib>
#include <iostream>

static bool Number(const char *text, uint64_t &value)
{
    if (text[0] == '-' || text[0] == '\0') return false;
    errno = 0;
    char *end = nullptr;
    const auto parsed = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0') return false;
    value = parsed;
    return true;
}

int main(int argc, char **argv)
{
    // batch Hq Hkv physical_pages page_tokens table_width UB_bytes AIV_cores scratch_bytes
    if (argc != 10) return 2;
    uint64_t values[9]{};
    for (int i = 0; i < 9; ++i) {
        if (!Number(argv[i + 1], values[i]) || (i != 6 && values[i] > UINT32_MAX)) return 2;
    }
    TqPackedAttentionTilingData plan{};
    const bool ok = tq_packed_attention::MakePlan(
        values[0], values[1], values[2], values[3], values[4], values[5],
        values[6], values[7], values[8], 1.0f / std::sqrt(128.0f), plan);
    if (!ok) {
        std::cout << "{\"plan_valid\":false,\"device_tested\":false}\n";
        return 1;
    }
    std::cout << "{\"plan_valid\":true,\"device_tested\":false,\"abi_bytes\":" << sizeof(plan)
              << ",\"tile_tokens\":" << plan.tileTokens << ",\"work_items\":" << plan.workItems
              << ",\"block_dim\":" << plan.blockDim << ",\"ub_bytes\":" << plan.ubPlanBytes
              << ",\"user_workspace_bytes\":0,\"key_bytes\":" << plan.keyBytes
              << ",\"buffers\":[";
    bool first = true;
    for (const auto &buffer : tq_packed_attention::Buffers(
             plan.tileTokens, plan.groupHeads, plan.apiScratchBytes)) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"name\":\"" << buffer.name << "\",\"bytes\":" << buffer.bytes << '}';
    }
    std::cout << "]}\n";
    return 0;
}
