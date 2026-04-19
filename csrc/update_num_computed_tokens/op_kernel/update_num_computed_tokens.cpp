/**
 * UpdateNumComputedTokens AscendC Kernel
 *
 * In async speculative decoding, the CPU-side num_computed_tokens is
 * "optimistic" (assumes all draft tokens accepted). This kernel corrects
 * the GPU-side values using the previous step's actual acceptance counts.
 *
 * For each request i:
 *   prev_idx = prev_positions[i]
 *   if prev_idx >= 0 AND prev_num_draft_tokens[prev_idx] > 0:
 *     num_computed_tokens[i] = cpu_values[i] - prev_num_draft_tokens[prev_idx]
 *                              - 1 + valid_sampled_token_count[prev_idx]
 *     num_accepted_tokens[i] = valid_sampled_token_count[prev_idx]
 *   else:
 *     num_computed_tokens[i] = cpu_values[i]
 *     // num_accepted_tokens[i] unchanged (pre-allocated output preserved)
 *
 * Multi-core: requests distributed across cores.
 * Data per request is tiny (scalar int32), so uses GetValue/SetValue on GM.
 */

#include "kernel_operator.h"

using namespace AscendC;

class UpdateNumComputedTokensKernel {
public:
    __aicore__ inline UpdateNumComputedTokensKernel() {}

    __aicore__ inline void Init(GM_ADDR prevPositions,
                                GM_ADDR validSampledTokenCount,
                                GM_ADDR prevNumDraftTokens,
                                GM_ADDR cpuValues,
                                GM_ADDR numComputedTokens,
                                GM_ADDR numAcceptedTokens,
                                const UpdateNumComputedTokensTilingData* tilingData)
    {
        numReqs_ = tilingData->numReqs;
        uint32_t validSampledSize = tilingData->validSampledSize;
        uint32_t prevDraftSize = tilingData->prevDraftSize;

        uint32_t coreId = GetBlockIdx();
        uint32_t reqsPerCore = tilingData->reqsPerCore;
        uint32_t remainderReqs = tilingData->remainderReqs;

        if (coreId < remainderReqs) {
            myStart_ = coreId * (reqsPerCore + 1);
            myCount_ = reqsPerCore + 1;
        } else {
            myStart_ = remainderReqs * (reqsPerCore + 1)
                     + (coreId - remainderReqs) * reqsPerCore;
            myCount_ = reqsPerCore;
        }

        gmPrevPositions_.SetGlobalBuffer((__gm__ int32_t*)prevPositions, numReqs_);
        gmValidSampled_.SetGlobalBuffer((__gm__ int32_t*)validSampledTokenCount, validSampledSize);
        gmPrevDraft_.SetGlobalBuffer((__gm__ int32_t*)prevNumDraftTokens, prevDraftSize);
        gmCpuValues_.SetGlobalBuffer((__gm__ int32_t*)cpuValues, numReqs_);
        gmNumComputed_.SetGlobalBuffer((__gm__ int32_t*)numComputedTokens, numReqs_);
        gmNumAccepted_.SetGlobalBuffer((__gm__ int32_t*)numAcceptedTokens, numReqs_);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < myCount_; r++) {
            uint32_t i = myStart_ + r;

            int32_t prevPos = gmPrevPositions_.GetValue(i);
            int32_t cpuVal = gmCpuValues_.GetValue(i);

            if (prevPos >= 0) {
                int32_t draftTokens = gmPrevDraft_.GetValue(prevPos);
                if (draftTokens > 0) {
                    int32_t validCount = gmValidSampled_.GetValue(prevPos);
                    gmNumComputed_.SetValue(i, cpuVal - draftTokens - 1 + validCount);
                    gmNumAccepted_.SetValue(i, validCount);
                    continue;
                }
            }
            // New request or no draft: use CPU value directly
            gmNumComputed_.SetValue(i, cpuVal);
        }
    }

private:
    GlobalTensor<int32_t> gmPrevPositions_, gmValidSampled_, gmPrevDraft_;
    GlobalTensor<int32_t> gmCpuValues_, gmNumComputed_, gmNumAccepted_;
    uint32_t numReqs_;
    uint32_t myStart_, myCount_;
};

extern "C" __global__ __aicore__ void update_num_computed_tokens(
    GM_ADDR prevPositions,
    GM_ADDR validSampledTokenCount,
    GM_ADDR prevNumDraftTokens,
    GM_ADDR cpuValues,
    GM_ADDR numComputedTokens,
    GM_ADDR numAcceptedTokens,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        UpdateNumComputedTokensKernel op;
        op.Init(prevPositions, validSampledTokenCount, prevNumDraftTokens,
                cpuValues, numComputedTokens, numAcceptedTokens, &tilingData);
        op.Process();
    }
}
