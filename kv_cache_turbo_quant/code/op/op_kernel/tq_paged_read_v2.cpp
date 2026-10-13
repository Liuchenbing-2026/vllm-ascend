#include "kernel_operator.h"
#include "tq_paged_read_v2_tiling.h"
using namespace AscendC;

namespace {
constexpr uint32_t PACKED_ROW_BYTES = 66;
constexpr uint32_t LOCAL_ROW_BYTES = 96;
constexpr uint32_t INDEX_BYTES = 64;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t LOOKUP_WORDS = 256;
constexpr uint32_t VECTOR_MASK_U16 = 128;
constexpr uint32_t VECTOR_MASK_F32 = 64;
constexpr uint32_t F32_ROW_BLOCKS = HEAD_DIM * sizeof(float) / 32;
constexpr uint32_t NORM_REPEAT_VALUES = 8;
constexpr uint32_t BYTE_PLANES = sizeof(uint32_t);
constexpr uint32_t SHORT_TILE_ROWS = 32;
constexpr uint32_t LONG_TILE_ROWS = 128;
constexpr uint32_t Log2(uint32_t value)
{
    return value <= 1 ? 0 : 1 + Log2(value / 2);
}
}

template<uint32_t TILE_ROWS>
class TqPagedReadV2Kernel {
    static constexpr uint32_t CODE_WORDS = TILE_ROWS * INDEX_BYTES / sizeof(uint32_t);
    static constexpr uint32_t OUTPUT_WORDS = TILE_ROWS * HEAD_DIM / 2;
    static constexpr uint32_t OUTPUT_ELEMENTS = TILE_ROWS * HEAD_DIM;
    static constexpr uint32_t PLANE_BYTE_SHIFT = Log2(CODE_WORDS * sizeof(uint32_t));
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR cache, GM_ADDR table, GM_ADDR jobs,
        GM_ADDR lookup, GM_ADDR output, const TqPagedReadV2TilingData *data)
    {
        data_ = data;
        cache_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(cache), data->cacheBytes);
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        jobs_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(jobs), data->jobs * 4);
        lookup_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lookup), LOOKUP_WORDS);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output), data->outputElements);
        pipe->InitBuffer(rawQueue_, 1, TILE_ROWS * LOCAL_ROW_BYTES);
        pipe->InitBuffer(outputQueue_, 1, OUTPUT_ELEMENTS * sizeof(bfloat16_t));
        pipe->InitBuffer(lut_, LOOKUP_WORDS * sizeof(float));
        pipe->InitBuffer(codes_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(shifted_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(index_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(mask_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(codeOffsets_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(normOffsets_, TILE_ROWS * sizeof(uint32_t));
        pipe->InitBuffer(norm_, TILE_ROWS * sizeof(uint32_t));
        pipe->InitBuffer(normBrcb_, TILE_ROWS * NORM_REPEAT_VALUES * sizeof(float));
        pipe->InitBuffer(planes_, OUTPUT_WORDS * sizeof(float));
        pipe->InitBuffer(values_, OUTPUT_ELEMENTS * sizeof(float));
        DataCopy(lut_.Get<float>(), lookup_, LOOKUP_WORDS);
        auto codeOffsets = codeOffsets_.Get<uint32_t>();
        auto temporary = index_.Get<uint32_t>();
        auto shifted = shifted_.Get<uint32_t>();
        auto mask = mask_.Get<uint32_t>();
        CreateVecIndex(codeOffsets.ReinterpretCast<int32_t>(), 0, CODE_WORDS);
        Duplicate(mask, 15U, CODE_WORDS);
        PipeBarrier<PIPE_V>();
        And(temporary.ReinterpretCast<uint16_t>(), codeOffsets.ReinterpretCast<uint16_t>(),
            mask.ReinterpretCast<uint16_t>(), VECTOR_MASK_U16, CODE_WORDS / 64,
            BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        ShiftLeft(temporary, temporary, 2U, CODE_WORDS);
        ShiftRight(codeOffsets, codeOffsets, 4U, CODE_WORDS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(shifted, codeOffsets, 5U, CODE_WORDS);
        ShiftLeft(codeOffsets, codeOffsets, 6U, CODE_WORDS);
        PipeBarrier<PIPE_V>();
        Add(codeOffsets.ReinterpretCast<int32_t>(), codeOffsets.ReinterpretCast<int32_t>(),
            shifted.ReinterpretCast<int32_t>(), CODE_WORDS);
        PipeBarrier<PIPE_V>();
        Add(codeOffsets.ReinterpretCast<int32_t>(), codeOffsets.ReinterpretCast<int32_t>(),
            temporary.ReinterpretCast<int32_t>(), CODE_WORDS);
        PipeBarrier<PIPE_V>();
        auto normOffsets = normOffsets_.Get<uint32_t>();
        auto normTemporary = norm_.Get<uint32_t>();
        CreateVecIndex(normOffsets.ReinterpretCast<int32_t>(), 0, TILE_ROWS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(normTemporary, normOffsets, 5U, TILE_ROWS);
        ShiftLeft(normOffsets, normOffsets, 6U, TILE_ROWS);
        PipeBarrier<PIPE_V>();
        Add(normOffsets.ReinterpretCast<int32_t>(), normOffsets.ReinterpretCast<int32_t>(),
            normTemporary.ReinterpretCast<int32_t>(), TILE_ROWS);
        PipeBarrier<PIPE_V>();
        Adds(normOffsets.ReinterpretCast<int32_t>(), normOffsets.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(INDEX_BYTES), TILE_ROWS);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t job = GetBlockIdx(); job < data_->jobs; job += GetBlockNum()) {
            const uint32_t sequence = jobs_.GetValue(job * 4);
            const uint32_t page = jobs_.GetValue(job * 4 + 1);
            const uint32_t destination = jobs_.GetValue(job * 4 + 2);
            const uint32_t tokens = jobs_.GetValue(job * 4 + 3);
            const uint32_t physical = table_.GetValue(sequence * data_->tableWidth + page);
            const uint32_t jobRows = tokens * data_->heads;
            for (uint32_t row = 0; row < jobRows; row += TILE_ROWS) {
                const uint32_t count = jobRows - row < TILE_ROWS ? jobRows - row : TILE_ROWS;
                const uint64_t source = (static_cast<uint64_t>(physical) * data_->blockTokens * data_->heads + row)
                    * PACKED_ROW_BYTES;
                const uint64_t target = (static_cast<uint64_t>(destination) * data_->heads + row) * HEAD_DIM;
                RunTile(source, target, count);
            }
        }
    }

private:
    __aicore__ inline void RunTile(uint64_t source, uint64_t target, uint32_t rows)
    {
        auto raw = rawQueue_.template AllocTensor<uint8_t>();
        if (rows < TILE_ROWS) {
            Duplicate(raw.ReinterpretCast<uint32_t>(), 0U, TILE_ROWS * LOCAL_ROW_BYTES / 4);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        DataCopyPad(raw, cache_[source], DataCopyExtParams{static_cast<uint16_t>(rows), PACKED_ROW_BYTES, 0, 0, 0},
            DataCopyPadExtParams<uint8_t>{true, 0, LOCAL_ROW_BYTES - PACKED_ROW_BYTES, 0});
        rawQueue_.EnQue(raw);
        raw = rawQueue_.template DeQue<uint8_t>();
        auto rawWords = raw.ReinterpretCast<float>();
        auto codes = codes_.Get<float>();
        Gather(codes, rawWords, codeOffsets_.Get<uint32_t>(), 0, CODE_WORDS);
        auto norm = norm_.Get<float>();
        Gather(norm, rawWords, normOffsets_.Get<uint32_t>(), 0, TILE_ROWS);
        PipeBarrier<PIPE_V>();
        // Norm is two low bytes of the padded row's norm word. Shift to the
        // high half of an FP32 word; this recovers the stored BF16 value exactly.
        ShiftLeft(norm.ReinterpretCast<uint32_t>(), norm.ReinterpretCast<uint32_t>(), 16U, TILE_ROWS);
        PipeBarrier<PIPE_V>();
        auto normBrcb = normBrcb_.Get<float>();
        Brcb(normBrcb, norm, TILE_ROWS / NORM_REPEAT_VALUES, BrcbRepeatParams(1, 8));
        Duplicate(mask_.Get<uint32_t>(), 0xFFU, CODE_WORDS);
        PipeBarrier<PIPE_V>();
        auto shifted = shifted_.Get<uint32_t>();
        auto index = index_.Get<uint32_t>();
        auto planes = planes_.Get<float>();
        for (uint32_t byte = 0; byte < 4; ++byte) {
            ShiftRight(shifted, codes.ReinterpretCast<uint32_t>(), byte * 8, CODE_WORDS);
            PipeBarrier<PIPE_V>();
            // dav_c220 bitwise And operates on 16-bit lanes. Double the lane
            // count and use the [0x00FF,0] halves of each uint32 mask.
            And(index.ReinterpretCast<uint16_t>(), shifted.ReinterpretCast<uint16_t>(),
                mask_.Get<uint32_t>().ReinterpretCast<uint16_t>(), 128, CODE_WORDS / 64,
                BinaryRepeatParams{1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            ShiftLeft(index, index, 2U, CODE_WORDS);
            PipeBarrier<PIPE_V>();
            // One 32-bit LUT word holds both BF16 centroids for a packed byte.
            // Gather copies the word bits; no FP32 arithmetic is applied here.
            Gather(planes[byte * CODE_WORDS], lut_.Get<float>(), index, 0, CODE_WORDS);
            PipeBarrier<PIPE_V>();
        }
        // Values is scratch until the lookup words have been reordered. The
        // final BF16 output buffer holds those words temporarily; both uses
        // finish before the output is enqueued for the GM write.
        auto output = outputQueue_.template AllocTensor<bfloat16_t>();
        auto values = values_.Get<float>();
        auto joinOffsets = values.ReinterpretCast<uint32_t>();
        auto planeOffsets = joinOffsets[OUTPUT_WORDS];
        CreateVecIndex(joinOffsets.ReinterpretCast<int32_t>(), 0, OUTPUT_WORDS);
        Duplicate(mask_.Get<uint32_t>(), BYTE_PLANES - 1U, CODE_WORDS);
        PipeBarrier<PIPE_V>();
        And(planeOffsets.ReinterpretCast<uint16_t>(), joinOffsets.ReinterpretCast<uint16_t>(),
            mask_.Get<uint32_t>().ReinterpretCast<uint16_t>(), VECTOR_MASK_U16, OUTPUT_WORDS / 64,
            BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        ShiftLeft(planeOffsets, planeOffsets, PLANE_BYTE_SHIFT, OUTPUT_WORDS);
        ShiftRight(joinOffsets, joinOffsets, 2U, OUTPUT_WORDS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(joinOffsets, joinOffsets, 2U, OUTPUT_WORDS);
        PipeBarrier<PIPE_V>();
        Or(joinOffsets.ReinterpretCast<uint16_t>(), joinOffsets.ReinterpretCast<uint16_t>(),
            planeOffsets.ReinterpretCast<uint16_t>(), VECTOR_MASK_U16, OUTPUT_WORDS / 64,
            BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        auto joined = output.template ReinterpretCast<float>();
        Gather(joined, planes, joinOffsets, 0, OUTPUT_WORDS);
        PipeBarrier<PIPE_V>();
        Cast(values, output, RoundMode::CAST_NONE, OUTPUT_ELEMENTS);
        PipeBarrier<PIPE_V>();
        // One norm block contains eight identical FP32 values. A zero source
        // block stride broadcasts it across each 64-element half-head; the
        // repeat stride advances to the next row, avoiding repeated Gather
        // reads from the same UB address.
        Mul(values, values, normBrcb, VECTOR_MASK_F32, TILE_ROWS,
            BinaryRepeatParams{1, 1, 0, F32_ROW_BLOCKS, F32_ROW_BLOCKS, 1});
        Mul(values[HEAD_DIM / 2], values[HEAD_DIM / 2], normBrcb, VECTOR_MASK_F32, TILE_ROWS,
            BinaryRepeatParams{1, 1, 0, F32_ROW_BLOCKS, F32_ROW_BLOCKS, 1});
        PipeBarrier<PIPE_V>();
        Cast(output, values, RoundMode::CAST_RINT, OUTPUT_ELEMENTS);
        outputQueue_.EnQue(output);
        output = outputQueue_.template DeQue<bfloat16_t>();
        DataCopy(output_[target], output, rows * HEAD_DIM);
        outputQueue_.FreeTensor(output);
        rawQueue_.FreeTensor(raw);
    }

    const TqPagedReadV2TilingData *data_;
    GlobalTensor<uint8_t> cache_;
    GlobalTensor<int32_t> table_, jobs_;
    GlobalTensor<float> lookup_;
    GlobalTensor<bfloat16_t> output_;
    TQue<QuePosition::VECIN, 1> rawQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> lut_, codes_, shifted_, index_, mask_, codeOffsets_;
    TBuf<TPosition::VECCALC> normOffsets_, norm_, normBrcb_;
    TBuf<TPosition::VECCALC> planes_, values_;
};

extern "C" __global__ __aicore__ void tq_paged_read_v2(GM_ADDR cache, GM_ADDR table, GM_ADDR jobs,
    GM_ADDR lookup, GM_ADDR dense, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TqPagedReadV2TilingData);
    TqPagedReadV2TilingData data;
    const auto *source = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    auto *destination = reinterpret_cast<uint32_t *>(&data);
    for (uint32_t i = 0; i < sizeof(data) / sizeof(uint32_t); ++i) {
        destination[i] = source[i];
    }
    TPipe pipe;
    if (data.tileRows == LONG_TILE_ROWS) {
        TqPagedReadV2Kernel<LONG_TILE_ROWS> kernel;
        kernel.Init(&pipe, cache, table, jobs, lookup, dense, &data);
        kernel.Process();
    } else {
        TqPagedReadV2Kernel<SHORT_TILE_ROWS> kernel;
        kernel.Init(&pipe, cache, table, jobs, lookup, dense, &data);
        kernel.Process();
    }
}
