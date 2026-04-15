/**
 * UpdateTokenIdsNgram AscendC Kernel
 *
 * Fuses the update_token_ids_ngram Python function:
 *   1. Backup last valid token per request
 *   2. Invalidate sampled tokens for discarded requests
 *   3. Count valid tokens and find the last valid
 *   4. Select next_token = last_valid or backup
 *
 * Multi-core: each core processes a subset of requests.
 */

#include "kernel_operator.h"

using namespace AscendC;

class UpdateTokenIdsNgramKernel {
public:
    __aicore__ inline UpdateTokenIdsNgramKernel() {}

    __aicore__ inline void Init(GM_ADDR sampledTokenIds, GM_ADDR tokenIdsGpu,
                                GM_ADDR numTokensNoSpec, GM_ADDR discardMask,
                                GM_ADDR outNextTokenIds, GM_ADDR outValidCount,
                                GM_ADDR outValidSampledTokenIds,
                                const UpdateTokenIdsNgramTilingData* tilingData)
    {
        usedCoreNum_ = tilingData->usedCoreNum;
        numReqs_ = tilingData->numReqs;
        reqsPerCore_ = tilingData->reqsPerCore;
        remainderReqs_ = tilingData->remainderReqs;
        maxNewTokens_ = tilingData->maxNewTokens;
        maxSeqLen_ = tilingData->maxSeqLen;
        vocabSize_ = tilingData->vocabSize;

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
        gmSampledTokenIds_.SetGlobalBuffer((__gm__ int32_t*)sampledTokenIds,
                                           numReqs_ * maxNewTokens_);
        gmTokenIdsGpu_.SetGlobalBuffer((__gm__ int32_t*)tokenIdsGpu,
                                       numReqs_ * maxSeqLen_);
        gmNumTokensNoSpec_.SetGlobalBuffer((__gm__ int32_t*)numTokensNoSpec, numReqs_);
        gmDiscardMask_.SetGlobalBuffer((__gm__ int8_t*)discardMask, numReqs_);

        gmOutNextTokenIds_.SetGlobalBuffer((__gm__ int32_t*)outNextTokenIds, numReqs_);
        gmOutValidCount_.SetGlobalBuffer((__gm__ int32_t*)outValidCount, numReqs_);
        gmOutValidSampled_.SetGlobalBuffer((__gm__ int32_t*)outValidSampledTokenIds,
                                           numReqs_ * maxNewTokens_);

        // Allocate UB buffers
        // Metadata: num_tokens_no_spec for my requests
        uint32_t numTokBufSize = AlignUp(myNumReqs_ * sizeof(int32_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(numTokBuf_, numTokBufSize);

        // Metadata: discard_mask for my requests
        uint32_t discardBufSize = AlignUp(myNumReqs_ * sizeof(int8_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(discardBuf_, discardBufSize);

        // Sampled tokens buffer (per request processing)
        uint32_t sampledBufSize = AlignUp(maxNewTokens_ * sizeof(int32_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(sampledBuf_, sampledBufSize);

        // Backup token buffer (single int32, aligned)
        pipe_.InitBuffer(backupBuf_, ONE_BLK_SIZE);

        // Output buffer for valid_sampled (reuse sampledBuf for output? No, use separate)
        pipe_.InitBuffer(outSampledBuf_, sampledBufSize);

        // Output buffer for scalar outputs (next_token, valid_count)
        pipe_.InitBuffer(scalarBuf_, ONE_BLK_SIZE);

        // Pre-load metadata
        if (myNumReqs_ > 0) {
            LocalTensor<int32_t> lNumTok = numTokBuf_.Get<int32_t>();
            DataCopyIn_int32(lNumTok, gmNumTokensNoSpec_, (int32_t)myStartReq_, (int32_t)myNumReqs_);

            LocalTensor<int8_t> lDiscard = discardBuf_.Get<int8_t>();
            DataCopyIn_int8(lDiscard, gmDiscardMask_, (int32_t)myStartReq_, (int32_t)myNumReqs_);
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

    __aicore__ inline void ProcessOneRequest(uint32_t r, uint32_t rLocal)
    {
        // 1. Read backup token: token_ids_gpu[r, max(0, numTok-1)]
        LocalTensor<int32_t> lNumTok = numTokBuf_.Get<int32_t>();
        int32_t numTok = lNumTok.GetValue(rLocal);
        int32_t backupIdx = numTok - 1;
        if (backupIdx < 0) backupIdx = 0;

        int32_t gmBackupOffset = (int32_t)r * (int32_t)maxSeqLen_ + backupIdx;
        LocalTensor<int32_t> lBackup = backupBuf_.Get<int32_t>();
        DataCopyIn_int32(lBackup, gmTokenIdsGpu_, gmBackupOffset, 1);
        int32_t backupToken = lBackup.GetValue(0);

        // 2. Read sampled tokens for this request
        int32_t sampledGmOffset = (int32_t)r * (int32_t)maxNewTokens_;
        LocalTensor<int32_t> lSampled = sampledBuf_.Get<int32_t>();
        DataCopyIn_int32(lSampled, gmSampledTokenIds_, sampledGmOffset, (int32_t)maxNewTokens_);

        // 3. Check discard mask
        LocalTensor<int8_t> lDiscard = discardBuf_.Get<int8_t>();
        int8_t discarded = lDiscard.GetValue(rLocal);

        // 4. Process: apply discard, count valid, find last valid token
        LocalTensor<int32_t> lOutSampled = outSampledBuf_.Get<int32_t>();
        int32_t validCount = 0;
        int32_t lastValidToken = backupToken;

        for (uint32_t j = 0; j < maxNewTokens_; j++) {
            int32_t tok = lSampled.GetValue(j);
            if (discarded != 0) {
                tok = -1;
            }
            lOutSampled.SetValue(j, tok);
            if (tok != -1 && tok < vocabSize_) {
                validCount++;
                lastValidToken = tok;
            }
        }

        // 5. Determine next_token_ids
        // Original logic: next_token = valid_sampled[valid_count - 1] if count > 0 else backup
        // Since valid tokens are contiguous from start, lastValidToken == valid_sampled[count-1]
        int32_t nextToken = (validCount > 0) ? lastValidToken : backupToken;

        // 6. Write outputs
        LocalTensor<int32_t> lScalar = scalarBuf_.Get<int32_t>();

        // next_token_ids[r]
        lScalar.SetValue(0, nextToken);
        DataCopyOut_int32(gmOutNextTokenIds_, lScalar, (int32_t)r, 1);

        // valid_count[r]
        lScalar.SetValue(0, validCount);
        DataCopyOut_int32(gmOutValidCount_, lScalar, (int32_t)r, 1);

        // valid_sampled_token_ids[r, :]
        DataCopyOut_int32(gmOutValidSampled_, lOutSampled, sampledGmOffset, (int32_t)maxNewTokens_);
    }

private:
    GlobalTensor<int32_t> gmSampledTokenIds_, gmTokenIdsGpu_;
    GlobalTensor<int32_t> gmNumTokensNoSpec_;
    GlobalTensor<int8_t> gmDiscardMask_;
    GlobalTensor<int32_t> gmOutNextTokenIds_, gmOutValidCount_, gmOutValidSampled_;

    uint32_t usedCoreNum_, numReqs_, reqsPerCore_, remainderReqs_;
    uint32_t maxNewTokens_, maxSeqLen_;
    int32_t vocabSize_;
    uint32_t myStartReq_, myNumReqs_;

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> numTokBuf_, discardBuf_;
    TBuf<QuePosition::VECCALC> sampledBuf_, backupBuf_, outSampledBuf_, scalarBuf_;
};

extern "C" __global__ __aicore__ void update_token_ids_ngram(
    GM_ADDR sampledTokenIds, GM_ADDR tokenIdsGpu,
    GM_ADDR numTokensNoSpec, GM_ADDR discardMask,
    GM_ADDR outNextTokenIds, GM_ADDR outValidCount,
    GM_ADDR outValidSampledTokenIds,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        UpdateTokenIdsNgramKernel op;
        op.Init(sampledTokenIds, tokenIdsGpu, numTokensNoSpec, discardMask,
                outNextTokenIds, outValidCount, outValidSampledTokenIds,
                &tilingData);
        op.Process();
    }
}
