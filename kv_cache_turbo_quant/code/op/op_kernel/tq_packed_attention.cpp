// First real AIV implementation. Source candidate only: not CANN-compiled or
// device-verified. No claim of FIA bitwise equivalence or performance gain.
#include "kernel_operator.h"
#define TQ_PACKED_CORE_INLINE __aicore__ inline
#include "tq_packed_attention_core.h"
#undef TQ_PACKED_CORE_INLINE

using namespace AscendC;

class TqPackedAttentionKernel {
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR query, GM_ADDR key, GM_ADDR value,
        GM_ADDR table, GM_ADDR lengths, GM_ADDR centroids, GM_ADDR output, GM_ADDR status,
        const TqPackedAttentionTilingData *data)
    {
        pipe_ = pipe;
        d_ = data;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(key), d_->keyBytes);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(value), d_->valueBytes);
        tableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        lengthsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        centroidsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(centroids), 16);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        statusGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
        // One TPipe-owned arena; no fixed-address UB or queues sharing events.
        pipe_->InitBuffer(arena_, d_->ubPlanBytes);
        auto bytes = arena_.Get<uint8_t>();
        uint32_t offset = 0;
        for (uint32_t i = 0; i < tq_packed_core::BufferCount; ++i) {
            const uint32_t size = static_cast<uint32_t>(tq_packed_core::BufferBytes(
                i, d_->tileTokens, d_->groupHeads, d_->apiScratchBytes));
            if (size) {
                memory_[i] = bytes[offset];
                memory_[i].SetSize(size);
            }
            offset += size;
        }
        // Only the 16 BF16 centroids are inputs. The historical 256-entry byte
        // LUT/offset payload is neither required nor silently substituted.
        DataCopy(memory_[11].ReinterpretCast<bfloat16_t>(), centroidsGm_, 16);
        Sync<HardEvent::MTE2_S>();
    }

    __aicore__ inline void Process()
    {
        // Host-validated ABI v1; B=0 has one launch block and no input/output IO.
        for (uint64_t job = GetBlockIdx(); job < d_->workItems; job += GetBlockNum()) {
            RunJob(static_cast<uint32_t>(job));
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        // Never hardcode event IDs in a TPipe project. Each fetched event is
        // set/waited immediately, including error paths; none remains pending.
        const int32_t id = static_cast<int32_t>(pipe_->FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline int32_t ReadInt(GlobalTensor<int32_t> source, uint64_t index)
    {
        auto meta = memory_[12].ReinterpretCast<int32_t>();
        Sync<HardEvent::S_MTE2>();
        DataCopyPad(meta, source[index], DataCopyExtParams{1, 4, 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{true, 0, 7, 0});
        Sync<HardEvent::MTE2_S>();
        return meta.GetValue(0);
    }

    __aicore__ inline void ReadPacked(GlobalTensor<uint8_t> source,
                                    LocalTensor<uint8_t> target,
                                    const tq_packed_core::Span &span)
    {
        const uint64_t stride = static_cast<uint64_t>(d_->kvHeads - 1) * tq_packed_core::PackedRow;
        // Exact 66B GM blocks, 96B UB pitch. Large GM strides use one-row
        // copies instead of narrowing a 64-bit stride into DataCopyExtParams.
        if (stride <= UINT32_MAX) {
            DataCopyPad(target, source[span.offset],
                DataCopyExtParams{static_cast<uint16_t>(span.rows), tq_packed_core::PackedRow,
                                  static_cast<uint32_t>(stride), 0, 0},
                DataCopyPadExtParams<uint8_t>{true, 0, 30, 0});
        } else {
            for (uint32_t r = 0; r < span.rows; ++r) {
                const uint64_t address = span.offset + static_cast<uint64_t>(r) * d_->kvHeads * 66;
                DataCopyPad(target[r * tq_packed_core::LocalRow], source[address],
                    DataCopyExtParams{1, tq_packed_core::PackedRow, 0, 0, 0},
                    DataCopyPadExtParams<uint8_t>{true, 0, 30, 0});
            }
        }
    }

    __aicore__ inline bool DecodeRow(LocalTensor<uint8_t> packed, uint32_t row,
                                     LocalTensor<bfloat16_t> rounded)
    {
        const uint32_t base = row * tq_packed_core::LocalRow;
        const uint16_t bits = static_cast<uint16_t>(packed.GetValue(base + 64)) |
                              (static_cast<uint16_t>(packed.GetValue(base + 65)) << 8U);
        if (!tq_packed_core::FiniteNorm(bits)) return false;
        // Copy BF16 centroid bits as uint16; no scalar BF16 conversion or
        // arithmetic overload is required from the target SDK.
        auto lut = memory_[11].ReinterpretCast<uint16_t>();
        auto code = memory_[5].ReinterpretCast<uint16_t>();  // bytes [0,256)
        auto values = memory_[5][256].ReinterpretCast<float>();  // [256,768)
        for (uint32_t j = 0; j < 64; ++j) {
            const uint8_t byte = packed.GetValue(base + j);
            code.SetValue(j * 2, lut.GetValue(byte & 15U));
            code.SetValue(j * 2 + 1, lut.GetValue(byte >> 4U));
        }
        // BF16 norm -> FP32 is exact bit placement, not a scalar BF16 cast.
        auto normBits = memory_[12][96].ReinterpretCast<uint32_t>();
        normBits.SetValue(0, static_cast<uint32_t>(bits) << 16U);
        const float norm = normBits.ReinterpretCast<float>().GetValue(0);
        Sync<HardEvent::S_V>();
        Cast(values, code.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, 128);
        PipeBarrier<PIPE_V>();
        Muls(values, values, norm, 128);
        PipeBarrier<PIPE_V>();
        // Original codec is BF16 centroid * BF16 norm -> BF16. This rounding
        // cannot be omitted by keeping the product in FP32 for attention.
        Cast(rounded, values, RoundMode::CAST_RINT, 128);
        Sync<HardEvent::V_S>();  // decode scratch is safe for next scalar fill
        return true;
    }

    __aicore__ inline bool DecodeTile(uint32_t rows)
    {
        auto key = memory_[2].ReinterpretCast<float>();
        auto value = memory_[3].ReinterpretCast<bfloat16_t>();
        // A 256B row in phase scratch holds the rounded K before widening.
        // Separate source/destination is mandatory for widening BF16 -> FP32.
        auto roundedKey = memory_[4].ReinterpretCast<bfloat16_t>();
        for (uint32_t r = 0; r < rows; ++r) {
            if (!DecodeRow(memory_[0], r, roundedKey)) return false;
            Sync<HardEvent::S_V>();
            Cast(key[r * 128], roundedKey, RoundMode::CAST_NONE, 128);
            Sync<HardEvent::V_S>();
            if (!DecodeRow(memory_[1], r, value[r * 128])) return false;
        }
        return true;
    }

    __aicore__ inline float Sum128(LocalTensor<float> values)
    {
        auto sums = memory_[12][64].ReinterpretCast<float>();
        PipeBarrier<PIPE_V>();
        // Basic instruction: two 64-lane FP32 reductions, explicit 32B
        // destination. No high-level ReduceSum/Softmax scratch API is used.
        WholeReduceSum(sums, values, 64, 2, 1, 1, 8);
        Sync<HardEvent::V_S>();
        return sums.GetValue(0) + sums.GetValue(1);
    }

    __aicore__ inline float SumScores(LocalTensor<float> scores, uint32_t rows)
    {
        auto sums = memory_[12][64].ReinterpretCast<float>();
        PipeBarrier<PIPE_V>();
        if (rows <= 64) {
            WholeReduceSum(sums, scores, static_cast<int32_t>(rows), 1, 1, 1, 8);
        } else {
            WholeReduceSum(sums, scores, 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            // dst offset 2 is 8B aligned; src offset 64 is 256B aligned.
            WholeReduceSum(sums[2], scores[64], static_cast<int32_t>(rows - 64), 1, 1, 1, 8);
        }
        Sync<HardEvent::V_S>();
        return rows <= 64 ? sums.GetValue(0) : sums.GetValue(0) + sums.GetValue(2);
    }

    __aicore__ inline float ExpScalar(float x)
    {
        auto scalar = memory_[12][64].ReinterpretCast<float>();
        scalar.SetValue(0, x);
        Sync<HardEvent::S_V>();
        Exp(scalar, scalar, 1);
        Sync<HardEvent::V_S>();
        return scalar.GetValue(0);
    }

    __aicore__ inline void AttendTile(uint32_t rows, bool first)
    {
        auto key = memory_[2].ReinterpretCast<float>();
        auto value = memory_[3].ReinterpretCast<bfloat16_t>();
        auto phase = memory_[4].ReinterpretCast<float>();
        auto query = memory_[7].ReinterpretCast<float>();
        auto numerator = memory_[8].ReinterpretCast<float>();
        auto state = memory_[9].ReinterpretCast<float>();
        auto scores = memory_[10].ReinterpretCast<float>();
        for (uint32_t h = 0; h < d_->groupHeads; ++h) {
            float blockMax = 0.0f;
            for (uint32_t r = 0; r < rows; ++r) {
                Sync<HardEvent::S_V>();
                Mul(phase, query[h * 128], key[r * 128], 128);
                const float score = Sum128(phase) * d_->scale;
                scores.SetValue(r, score);
                if (!r || score > blockMax) blockMax = score;
            }
            const float oldMax = first ? blockMax : state.GetValue(h * 4);
            const float newMax = oldMax > blockMax ? oldMax : blockMax;
            const float oldSum = first ? 0.0f : state.GetValue(h * 4 + 1);
            const float alpha = first ? 0.0f : ExpScalar(oldMax - newMax);
            Sync<HardEvent::S_V>();
            Adds(scores, scores, -newMax, rows);
            PipeBarrier<PIPE_V>();
            Exp(scores, scores, rows);
            const float blockSum = SumScores(scores, rows);
            const float nextSum = oldSum * alpha + blockSum;
            Sync<HardEvent::S_V>();
            Muls(numerator[h * 128], numerator[h * 128], alpha, 128);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                const float weight = scores.GetValue(r);
                Sync<HardEvent::S_V>();
                Cast(phase, value[r * 128], RoundMode::CAST_NONE, 128);
                PipeBarrier<PIPE_V>();
                Muls(phase, phase, weight, 128);
                PipeBarrier<PIPE_V>();
                Add(numerator[h * 128], numerator[h * 128], phase, 128);
                PipeBarrier<PIPE_V>();
            }
            // Scalar state lives across tiles; scores and phase do not.
            state.SetValue(h * 4, newMax);
            state.SetValue(h * 4 + 1, nextSum);
        }
        // End all vector consumers before scalar/DMA phase-scratch reuse.
        Sync<HardEvent::V_S>();
    }

    __aicore__ inline void WriteOutput(uint64_t queryOffset, bool success)
    {
        auto rounded = memory_[6].ReinterpretCast<bfloat16_t>();
        auto numerator = memory_[8].ReinterpretCast<float>();
        auto state = memory_[9].ReinterpretCast<float>();
        auto phase = memory_[4].ReinterpretCast<float>();
        Sync<HardEvent::S_V>();
        if (success) {
            for (uint32_t h = 0; h < d_->groupHeads; ++h) {
                const float inverse = 1.0f / state.GetValue(h * 4 + 1);
                Sync<HardEvent::S_V>();
                Muls(phase, numerator[h * 128], inverse, 128);
                PipeBarrier<PIPE_V>();
                Cast(rounded[h * 128], phase, RoundMode::CAST_RINT, 128);
                PipeBarrier<PIPE_V>();
            }
        } else {
            // Error output is deterministic but MUST NOT be accepted as
            // success. ABI status remains mandatory for every caller.
            Duplicate(rounded.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), d_->groupHeads * 128);
        }
        Sync<HardEvent::V_MTE3>();
        DataCopy(outputGm_[queryOffset], rounded, d_->groupHeads * 128);
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void WriteStatus(uint32_t job, TqPackedAttentionStatus status)
    {
        auto local = memory_[12][32].ReinterpretCast<int32_t>();
        local.SetValue(0, static_cast<int32_t>(status));
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(statusGm_[job], local, DataCopyExtParams{1, 4, 0, 0, 0});
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void RunJob(uint32_t job)
    {
        const uint32_t batch = job / d_->kvHeads;
        const uint32_t kvHead = job % d_->kvHeads;
        const uint64_t queryOffset = (static_cast<uint64_t>(batch) * d_->queryHeads +
                                      static_cast<uint64_t>(kvHead) * d_->groupHeads) * 128;
        TqPackedAttentionStatus status = TqPackedAttentionStatus::Ok;
        const int32_t length = ReadInt(lengthsGm_, batch);
        if (!tq_packed_core::ValidLength(length, d_->tableWidth, d_->pageTokens)) {
            status = TqPackedAttentionStatus::InvalidLength;
        } else {
            auto q16 = memory_[6].ReinterpretCast<bfloat16_t>();
            DataCopy(q16, queryGm_[queryOffset], d_->groupHeads * 128);
            Sync<HardEvent::MTE2_V>();
            Cast(memory_[7].ReinterpretCast<float>(), q16, RoundMode::CAST_NONE, d_->groupHeads * 128);
            Duplicate(memory_[8].ReinterpretCast<float>(), 0.0f, d_->groupHeads * 128);
            PipeBarrier<PIPE_V>();
            for (uint32_t position = 0; position < static_cast<uint32_t>(length);) {
                const uint64_t index = static_cast<uint64_t>(batch) * d_->tableWidth + position / d_->pageTokens;
                const int32_t physical = ReadInt(tableGm_, index);
                tq_packed_core::Span span{};
                if (!tq_packed_core::PageSpan(*d_, position, static_cast<uint32_t>(length), physical, kvHead, span)) {
                    status = TqPackedAttentionStatus::InvalidPhysicalPage;
                    break;
                }
                Sync<HardEvent::S_MTE2>();
                ReadPacked(keyGm_, memory_[0], span);
                ReadPacked(valueGm_, memory_[1], span);
                Sync<HardEvent::MTE2_S>();
                if (!DecodeTile(span.rows)) {
                    status = TqPackedAttentionStatus::NonFiniteNorm;
                    break;
                }
                AttendTile(span.rows, position == 0);
                position += span.rows;
            }
        }
        WriteOutput(queryOffset, status == TqPackedAttentionStatus::Ok);
        WriteStatus(job, status);
    }

    TPipe *pipe_;
    const TqPackedAttentionTilingData *d_;
    // A2 DataCopyPad supports GM <-> VECIN/VECOUT, not GM <-> VECCALC.
    // All vector operations here also accept VECIN. One VECIN-owned arena
    // therefore preserves the host byte budget without unsupported DMA paths.
    TBuf<TPosition::VECIN> arena_;
    LocalTensor<uint8_t> memory_[tq_packed_core::BufferCount];
    GlobalTensor<bfloat16_t> queryGm_, centroidsGm_, outputGm_;
    GlobalTensor<uint8_t> keyGm_, valueGm_;
    GlobalTensor<int32_t> tableGm_, lengthsGm_, statusGm_;
};

extern "C" __global__ __aicore__ void tq_packed_attention(
    GM_ADDR query_rot, GM_ADDR key, GM_ADDR value, GM_ADDR table, GM_ADDR lengths,
    GM_ADDR centroids, GM_ADDR attention_rot, GM_ADDR status, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqPackedAttentionTilingData);
    (void)workspace;  // No user GM workspace; retain the existing ABI.
    // Existing project kernels use the exact 80B POD copy. This scalar load is
    // tiling metadata only; tensor payload uses bulk DMA, never GM GetValue.
    TqPackedAttentionTilingData data;
    const auto *source = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    auto *destination = reinterpret_cast<uint32_t *>(&data);
    for (uint32_t i = 0; i < sizeof(data) / sizeof(uint32_t); ++i) destination[i] = source[i];
    if (!data.workItems) return;
    TPipe pipe;
    TqPackedAttentionKernel kernel;
    kernel.Init(&pipe, query_rot, key, value, table, lengths, centroids, attention_rot, status, &data);
    kernel.Process();
}
