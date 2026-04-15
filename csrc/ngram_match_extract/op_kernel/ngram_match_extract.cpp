/**
 * NgramMatchExtract AscendC Kernel
 *
 * Fuses _find_first_and_extract_all_n_parallel + leading valid count:
 *   1. For each request, try n-gram sizes from max_n down to min_n
 *   2. Find the earliest suffix match in token history
 *   3. Extract k draft tokens after the match
 *   4. Count leading contiguous valid tokens
 *
 * Multi-core: each core processes a subset of requests.
 * Token history is scanned in chunks to fit within UB.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t CHUNK_SIZE = 4096;
constexpr uint32_t MAX_N_CAP = 16;
constexpr uint32_t MAX_K_CAP = 32;
constexpr uint32_t SCAN_BUF_ELEMS = CHUNK_SIZE + MAX_N_CAP;

class NgramMatchExtractKernel {
public:
    __aicore__ inline NgramMatchExtractKernel() {}

    __aicore__ inline void Init(GM_ADDR tokenIds, GM_ADDR seqLengths,
                                GM_ADDR combinedMask,
                                GM_ADDR outDraftTokens, GM_ADDR outNumValid,
                                const NgramMatchExtractTilingData* tilingData)
    {
        usedCoreNum_ = tilingData->usedCoreNum;
        numReqs_ = tilingData->numReqs;
        reqsPerCore_ = tilingData->reqsPerCore;
        remainderReqs_ = tilingData->remainderReqs;
        maxSeqLen_ = tilingData->maxSeqLen;
        minN_ = tilingData->minN;
        maxN_ = tilingData->maxN;
        k_ = tilingData->k;

        uint32_t coreId = GetBlockIdx();
        if (coreId < remainderReqs_) {
            myStartReq_ = coreId * (reqsPerCore_ + 1);
            myNumReqs_ = reqsPerCore_ + 1;
        } else {
            myStartReq_ = remainderReqs_ * (reqsPerCore_ + 1) +
                           (coreId - remainderReqs_) * reqsPerCore_;
            myNumReqs_ = reqsPerCore_;
        }

        // Bind GM tensors
        gmTokenIds_.SetGlobalBuffer((__gm__ int32_t*)tokenIds, numReqs_ * maxSeqLen_);
        gmSeqLengths_.SetGlobalBuffer((__gm__ int32_t*)seqLengths, numReqs_);
        gmCombinedMask_.SetGlobalBuffer((__gm__ int8_t*)combinedMask, numReqs_);
        gmOutDraftTokens_.SetGlobalBuffer((__gm__ int32_t*)outDraftTokens, numReqs_ * k_);
        gmOutNumValid_.SetGlobalBuffer((__gm__ int32_t*)outNumValid, numReqs_);

        // Allocate UB buffers
        // Pre-loaded metadata
        pipe_.InitBuffer(seqLenBuf_, AlignUp(myNumReqs_ * sizeof(int32_t), ONE_BLK_SIZE));
        pipe_.InitBuffer(maskBuf_, AlignUp(myNumReqs_ * sizeof(int8_t), ONE_BLK_SIZE));

        // Suffix buffer: holds last max_n tokens of the sequence
        pipe_.InitBuffer(suffixBuf_, AlignUp(MAX_N_CAP * sizeof(int32_t), ONE_BLK_SIZE));

        // Scan buffer: chunk of tokens for n-gram matching
        pipe_.InitBuffer(scanBuf_, AlignUp(SCAN_BUF_ELEMS * sizeof(int32_t), ONE_BLK_SIZE));

        // Draft output buffer
        pipe_.InitBuffer(draftBuf_, AlignUp(MAX_K_CAP * sizeof(int32_t), ONE_BLK_SIZE));

        // Scalar output buffer
        pipe_.InitBuffer(scalarBuf_, ONE_BLK_SIZE);

        // Pre-load metadata
        if (myNumReqs_ > 0) {
            LocalTensor<int32_t> lSeqLen = seqLenBuf_.Get<int32_t>();
            DataCopyIn_int32(lSeqLen, gmSeqLengths_, (int32_t)myStartReq_, (int32_t)myNumReqs_);

            LocalTensor<int8_t> lMask = maskBuf_.Get<int8_t>();
            DataCopyIn_int8(lMask, gmCombinedMask_, (int32_t)myStartReq_, (int32_t)myNumReqs_);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rLocal = 0; rLocal < myNumReqs_; rLocal++) {
            ProcessOneRequest(myStartReq_ + rLocal, rLocal);
        }
    }

private:
    static __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
    {
        return (x + a - 1) / a * a;
    }

    __aicore__ inline void DataCopyIn_int32(LocalTensor<int32_t>& dst,
                                             GlobalTensor<int32_t>& src,
                                             int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int32_t);
        int32_t aligned = (count + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
        DataCopy(dst, src[gmOffset], aligned);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void DataCopyIn_int8(LocalTensor<int8_t>& dst,
                                            GlobalTensor<int8_t>& src,
                                            int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        constexpr int32_t ELEMS_PER_BLK = ONE_BLK_SIZE / (int32_t)sizeof(int8_t);
        int32_t aligned = (count + ELEMS_PER_BLK - 1) / ELEMS_PER_BLK * ELEMS_PER_BLK;
        DataCopy(dst, src[gmOffset], aligned);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void DataCopyOut_int32(GlobalTensor<int32_t>& dst,
                                              LocalTensor<int32_t>& src,
                                              int32_t gmOffset, int32_t count)
    {
        if (count <= 0) return;
        uint32_t totalBytes = static_cast<uint32_t>(count) * sizeof(int32_t);
        pipe_barrier(PIPE_ALL);
        DataCopyPad(dst[gmOffset], src, DataCopyExtParams(1, totalBytes, 0, 0, 0));
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void WriteDraftMinusOnes(uint32_t r)
    {
        LocalTensor<int32_t> lDraft = draftBuf_.Get<int32_t>();
        for (uint32_t j = 0; j < k_; j++) {
            lDraft.SetValue(j, (int32_t)-1);
        }
        int32_t gmDraftOffset = (int32_t)r * (int32_t)k_;
        DataCopyOut_int32(gmOutDraftTokens_, lDraft, gmDraftOffset, (int32_t)k_);

        LocalTensor<int32_t> lScalar = scalarBuf_.Get<int32_t>();
        lScalar.SetValue(0, (int32_t)0);
        DataCopyOut_int32(gmOutNumValid_, lScalar, (int32_t)r, 1);
    }

    __aicore__ inline void ProcessOneRequest(uint32_t r, uint32_t rLocal)
    {
        // Check combined_mask
        LocalTensor<int8_t> lMask = maskBuf_.Get<int8_t>();
        if (lMask.GetValue(rLocal) == 0) {
            WriteDraftMinusOnes(r);
            return;
        }

        // Get sequence length
        LocalTensor<int32_t> lSeqLen = seqLenBuf_.Get<int32_t>();
        int32_t seqLen = lSeqLen.GetValue(rLocal);
        if (seqLen < (int32_t)minN_) {
            WriteDraftMinusOnes(r);
            return;
        }

        // Load suffix: last maxN tokens of the sequence
        // Clamp suffixStart to [0, seqLen)
        int32_t effMaxN = (int32_t)maxN_;
        if (effMaxN > seqLen) effMaxN = seqLen;

        int32_t suffixStart = seqLen - effMaxN;
        int32_t gmRowStart = (int32_t)r * (int32_t)maxSeqLen_;

        LocalTensor<int32_t> lSuffix = suffixBuf_.Get<int32_t>();
        DataCopyIn_int32(lSuffix, gmTokenIds_, gmRowStart + suffixStart, effMaxN);

        // Try each n from maxN down to minN, prefer longest match
        int32_t bestMatchPos = -1;
        int32_t bestN = 0;

        for (int32_t n = effMaxN; n >= (int32_t)minN_; n--) {
            int32_t suffixOffset = effMaxN - n;  // Where this n's suffix starts in suffixBuf
            int32_t maxSearchPos = seqLen - n - 1;  // Inclusive, must leave room for >=1 draft token
            if (maxSearchPos < 0) continue;

            bool found = false;

            // Scan token history in chunks
            for (int32_t chunkStart = 0; chunkStart <= maxSearchPos && !found;
                 chunkStart += (int32_t)CHUNK_SIZE)
            {
                // Load chunk with overlap for n-gram comparison
                int32_t chunkEnd = chunkStart + (int32_t)CHUNK_SIZE + n - 1;
                if (chunkEnd > seqLen) chunkEnd = seqLen;
                int32_t loadCount = chunkEnd - chunkStart;

                LocalTensor<int32_t> lScan = scanBuf_.Get<int32_t>();
                DataCopyIn_int32(lScan, gmTokenIds_, gmRowStart + chunkStart, loadCount);

                // Scan positions in this chunk
                int32_t scanEnd = chunkStart + (int32_t)CHUNK_SIZE;
                if (scanEnd > maxSearchPos + 1) scanEnd = maxSearchPos + 1;

                for (int32_t pos = chunkStart; pos < scanEnd; pos++) {
                    int32_t localPos = pos - chunkStart;
                    bool match = true;

                    for (int32_t j = 0; j < n; j++) {
                        if (lScan.GetValue(localPos + j) != lSuffix.GetValue(suffixOffset + j)) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
                        bestMatchPos = pos;
                        bestN = n;
                        found = true;
                        break;
                    }
                }
            }

            if (found) break;  // Found longest n, no need to try shorter
        }

        // No match found
        if (bestMatchPos < 0) {
            WriteDraftMinusOnes(r);
            return;
        }

        // Extract draft tokens starting after the match
        int32_t draftStart = bestMatchPos + bestN;
        int32_t tokensAvail = seqLen - draftStart;
        int32_t loadCount = (int32_t)k_;
        if (loadCount > tokensAvail) loadCount = tokensAvail;
        if (loadCount < 0) loadCount = 0;

        LocalTensor<int32_t> lDraft = draftBuf_.Get<int32_t>();

        if (loadCount > 0) {
            DataCopyIn_int32(lDraft, gmTokenIds_, gmRowStart + draftStart, loadCount);
        }

        // Fill remaining positions with -1
        for (int32_t j = loadCount; j < (int32_t)k_; j++) {
            lDraft.SetValue(j, (int32_t)-1);
        }

        // Count leading contiguous valid tokens (non -1)
        int32_t numValid = 0;
        for (int32_t j = 0; j < (int32_t)k_; j++) {
            if (lDraft.GetValue(j) != -1) {
                numValid++;
            } else {
                break;
            }
        }

        // Write outputs
        int32_t gmDraftOffset = (int32_t)r * (int32_t)k_;
        DataCopyOut_int32(gmOutDraftTokens_, lDraft, gmDraftOffset, (int32_t)k_);

        LocalTensor<int32_t> lScalar = scalarBuf_.Get<int32_t>();
        lScalar.SetValue(0, numValid);
        DataCopyOut_int32(gmOutNumValid_, lScalar, (int32_t)r, 1);
    }

private:
    GlobalTensor<int32_t> gmTokenIds_;
    GlobalTensor<int32_t> gmSeqLengths_;
    GlobalTensor<int8_t> gmCombinedMask_;
    GlobalTensor<int32_t> gmOutDraftTokens_;
    GlobalTensor<int32_t> gmOutNumValid_;

    uint32_t usedCoreNum_, numReqs_, reqsPerCore_, remainderReqs_;
    uint32_t maxSeqLen_, minN_, maxN_, k_;
    uint32_t myStartReq_, myNumReqs_;

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> seqLenBuf_, maskBuf_;
    TBuf<QuePosition::VECCALC> suffixBuf_, scanBuf_, draftBuf_, scalarBuf_;
};

extern "C" __global__ __aicore__ void ngram_match_extract(
    GM_ADDR tokenIds, GM_ADDR seqLengths, GM_ADDR combinedMask,
    GM_ADDR outDraftTokens, GM_ADDR outNumValid,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        NgramMatchExtractKernel op;
        op.Init(tokenIds, seqLengths, combinedMask,
                outDraftTokens, outNumValid, &tilingData);
        op.Process();
    }
}
