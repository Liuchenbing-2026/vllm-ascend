#include "kernel_operator.h"
#include "tq_paged_read_const_tiling.h"
using namespace AscendC;

namespace {
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t PACKED_ROW_BYTES = 66;
constexpr uint32_t LOCAL_ROW_BYTES = 96;
constexpr uint32_t INDEX_BYTES = 64;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t CODE_WORDS = TILE_ROWS * INDEX_BYTES / sizeof(uint32_t);
constexpr uint32_t OUTPUT_WORDS = TILE_ROWS * HEAD_DIM / 2;
constexpr uint32_t OUTPUT_ELEMENTS = TILE_ROWS * HEAD_DIM;
constexpr uint32_t LOOKUP_WORDS = 256;
}

class TqPagedReadConstKernel {
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR cache, GM_ADDR table, GM_ADDR jobs,
        GM_ADDR lookup, GM_ADDR output, const TqPagedReadConstTilingData *data)
    {
        data_ = data;
        cache_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(cache), data->cacheBytes);
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        jobs_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(jobs), data->jobs * 4);
        lookup_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lookup), LOOKUP_WORDS + CODE_WORDS + TILE_ROWS + OUTPUT_WORDS);
        offsetLookup_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(lookup), LOOKUP_WORDS + CODE_WORDS + TILE_ROWS + OUTPUT_WORDS);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output), data->outputElements);
        pipe->InitBuffer(rawQueue_, 2, TILE_ROWS * LOCAL_ROW_BYTES);
        pipe->InitBuffer(outputQueue_, 2, OUTPUT_ELEMENTS * sizeof(bfloat16_t));
        pipe->InitBuffer(lut_, LOOKUP_WORDS * sizeof(float));
        pipe->InitBuffer(codes_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(shifted_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(index_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(mask_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(codeOffsets_, CODE_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(normOffsets_, TILE_ROWS * sizeof(uint32_t));
        pipe->InitBuffer(norm_, TILE_ROWS * sizeof(uint32_t));
        pipe->InitBuffer(normExpandOffsets_, OUTPUT_ELEMENTS * sizeof(uint32_t));
        pipe->InitBuffer(normExpand_, OUTPUT_ELEMENTS * sizeof(float));
        pipe->InitBuffer(planes_, OUTPUT_WORDS * sizeof(float));
        pipe->InitBuffer(joinOffsets_, OUTPUT_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(joined_, OUTPUT_WORDS * sizeof(float));
        pipe->InitBuffer(values_, OUTPUT_ELEMENTS * sizeof(float));
        DataCopy(lut_.Get<float>(), lookup_, LOOKUP_WORDS);
        // Immutable tables are supplied beside the centroid LUT. Each core
        // copies them in bulk instead of issuing thousands of scalar stores.
        DataCopy(codeOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS], CODE_WORDS);
        DataCopy(normOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS + CODE_WORDS], TILE_ROWS);
        DataCopy(joinOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS + CODE_WORDS + TILE_ROWS], OUTPUT_WORDS);
        auto normOffsets = normExpandOffsets_.Get<uint32_t>();
        CreateVecIndex(normOffsets.ReinterpretCast<int32_t>(), 0, OUTPUT_ELEMENTS);
        PipeBarrier<PIPE_V>();
        ShiftRight(normOffsets, normOffsets, 7U, OUTPUT_ELEMENTS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(normOffsets, normOffsets, 2U, OUTPUT_ELEMENTS);
        Duplicate(mask_.Get<uint32_t>(), 0xFFU, CODE_WORDS);
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
        auto raw = rawQueue_.AllocTensor<uint8_t>();
        if (rows < TILE_ROWS) {
            Duplicate(raw.ReinterpretCast<uint32_t>(), 0U, TILE_ROWS * LOCAL_ROW_BYTES / 4);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        DataCopyPad(raw, cache_[source], DataCopyExtParams{static_cast<uint16_t>(rows), PACKED_ROW_BYTES, 0, 0, 0},
            DataCopyPadExtParams<uint8_t>{true, 0, LOCAL_ROW_BYTES - PACKED_ROW_BYTES, 0});
        rawQueue_.EnQue(raw);
        raw = rawQueue_.DeQue<uint8_t>();
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
        Gather(normExpand_.Get<float>(), norm, normExpandOffsets_.Get<uint32_t>(), 0, OUTPUT_ELEMENTS);
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
        auto joined = joined_.Get<float>();
        Gather(joined, planes, joinOffsets_.Get<uint32_t>(), 0, OUTPUT_WORDS);
        PipeBarrier<PIPE_V>();
        auto values = values_.Get<float>();
        Cast(values, joined.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, OUTPUT_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(values, values, normExpand_.Get<float>(), OUTPUT_ELEMENTS);
        PipeBarrier<PIPE_V>();
        auto output = outputQueue_.AllocTensor<bfloat16_t>();
        Cast(output, values, RoundMode::CAST_RINT, OUTPUT_ELEMENTS);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<bfloat16_t>();
        DataCopy(output_[target], output, rows * HEAD_DIM);
        outputQueue_.FreeTensor(output);
        rawQueue_.FreeTensor(raw);
    }

    const TqPagedReadConstTilingData *data_;
    GlobalTensor<uint8_t> cache_;
    GlobalTensor<int32_t> table_, jobs_;
    GlobalTensor<float> lookup_;
    GlobalTensor<uint32_t> offsetLookup_;
    GlobalTensor<bfloat16_t> output_;
    TQue<QuePosition::VECIN, 2> rawQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> lut_, codes_, shifted_, index_, mask_, codeOffsets_;
    TBuf<TPosition::VECCALC> normOffsets_, norm_, normExpandOffsets_, normExpand_;
    TBuf<TPosition::VECCALC> planes_, joinOffsets_, joined_, values_;
};

extern "C" __global__ __aicore__ void tq_paged_read_const(GM_ADDR cache, GM_ADDR table, GM_ADDR jobs,
    GM_ADDR lookup, GM_ADDR dense, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TqPagedReadConstTilingData);
    TqPagedReadConstTilingData data;
    const auto *source = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    auto *destination = reinterpret_cast<uint32_t *>(&data);
    for (uint32_t i = 0; i < sizeof(data) / sizeof(uint32_t); ++i) {
        destination[i] = source[i];
    }
    TPipe pipe;
    TqPagedReadConstKernel kernel;
    kernel.Init(&pipe, cache, table, jobs, lookup, dense, &data);
    kernel.Process();
}
