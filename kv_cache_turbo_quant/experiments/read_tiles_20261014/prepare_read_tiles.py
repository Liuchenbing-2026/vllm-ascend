"""Derive an independent tiled reader from the verified codec, without math changes."""

from pathlib import Path
import json
import shutil

root = Path(__file__).resolve().parent
source = root.parent.parent / "source/tq-read-tiles-20261014/kv_cache_turbo_quant"
op = source / "code/op"
previous = op / "op_kernel/tq_paged_read.cpp"
code = previous.read_text()
code = code.replace('"tq_paged_read_tiling.h"', '"tq_paged_read_v2_tiling.h"')
start = code.index("constexpr uint32_t TILE_ROWS = 32;")
end = code.index("public:", start)
code = code[:start] + """constexpr uint32_t PACKED_ROW_BYTES = 66;
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
""" + code[end:]
code = code.replace("TqPagedReadTilingData", "TqPagedReadV2TilingData")
code = code.replace("rawQueue_, 2,", "rawQueue_, 1,").replace("outputQueue_, 2,", "outputQueue_, 1,")
code = code.replace("        pipe->InitBuffer(normExpandOffsets_, OUTPUT_ELEMENTS * sizeof(uint32_t));\n", "")
code = code.replace("        pipe->InitBuffer(normExpand_, OUTPUT_ELEMENTS * sizeof(float));", "        pipe->InitBuffer(normBrcb_, TILE_ROWS * NORM_REPEAT_VALUES * sizeof(float));")
code = code.replace("        pipe->InitBuffer(joinOffsets_, OUTPUT_WORDS * sizeof(uint32_t));\n", "")
code = code.replace("        pipe->InitBuffer(joined_, OUTPUT_WORDS * sizeof(float));\n", "")
start = code.index("        for (uint32_t i = 0; i < CODE_WORDS; ++i)")
stop = code.index("        SetFlag<HardEvent::MTE2_V>(0);", start)
code = code[:start] + """        auto codeOffsets = codeOffsets_.Get<uint32_t>();
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
""" + code[stop:]
code = code.replace("        auto raw = rawQueue_.AllocTensor<uint8_t>();", "        auto raw = rawQueue_.template AllocTensor<uint8_t>();")
code = code.replace("rawQueue_.DeQue<uint8_t>()", "rawQueue_.template DeQue<uint8_t>()")
# TBuf is independent of the template parameter; dependent queue methods need
# the explicit template keyword, while scalar buffer accessors do not.
start = code.index("        Gather(normExpand_.Get<float>()")
stop = code.index("        auto shifted = shifted_.Get<uint32_t>();", start)
code = code[:start] + """        auto normBrcb = normBrcb_.Get<float>();
        Brcb(normBrcb, norm, TILE_ROWS / NORM_REPEAT_VALUES, BrcbRepeatParams(1, 8));
        Duplicate(mask_.Get<uint32_t>(), 0xFFU, CODE_WORDS);
        PipeBarrier<PIPE_V>();
""" + code[stop:]
start = code.index("        auto joined = joined_.Get<float>();")
stop = code.index("        outputQueue_.EnQue(output);", start)
code = code[:start] + """        // Values is scratch until the lookup words have been reordered. The
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
""" + code[stop:]
code = code.replace("outputQueue_.DeQue<bfloat16_t>()", "outputQueue_.template DeQue<bfloat16_t>()")
code = code.replace("TQue<QuePosition::VECIN, 2>", "TQue<QuePosition::VECIN, 1>")
code = code.replace("TQue<QuePosition::VECOUT, 2>", "TQue<QuePosition::VECOUT, 1>")
code = code.replace("normOffsets_, norm_, normExpandOffsets_, normExpand_", "normOffsets_, norm_, normBrcb_")
code = code.replace("planes_, joinOffsets_, joined_, values_", "planes_, values_")
code = code.replace('void tq_paged_read(', 'void tq_paged_read_v2(')
start = code.index("    TPipe pipe;", code.index('extern "C"'))
code = code[:start] + """    TPipe pipe;
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
"""
(op / "op_kernel/tq_paged_read_v2.cpp").write_text(code)
tiling = (op / "op_kernel/tq_paged_read_tiling.h").read_text().replace("TQ_PAGED_READ_TILING_H", "TQ_PAGED_READ_V2_TILING_H").replace("TqPagedReadTilingData", "TqPagedReadV2TilingData")
tiling = tiling.replace("    uint32_t tableWidth;", "    uint32_t tableWidth;\n    uint32_t tileRows;\n    uint32_t reserved;")
(op / "op_kernel/tq_paged_read_v2_tiling.h").write_text(tiling)
host = (op / "op_host/tq_paged_read.cpp").read_text().replace("tq_paged_read_tiling.h", "tq_paged_read_v2_tiling.h").replace("TqPagedRead", "TqPagedReadV2")
host = host.replace("constexpr int64_t LOOKUP_ELEMENTS = 512;", "constexpr int64_t LOOKUP_ELEMENTS = 512;\nconstexpr int64_t LONG_HISTORY_MIN_TOKENS = 65536;")
host = host.replace("    data->outputElements =", "    data->tileRows = tokens >= LONG_HISTORY_MIN_TOKENS ? 128 : 32;\n    data->reserved = 0;\n    data->outputElements =")
(op / "op_host/tq_paged_read_v2.cpp").write_text(host)
schema = json.loads((op / "TqPagedRead.json").read_text())
schema[0]["op"] = "TqPagedReadV2"
(op / "TqPagedReadV2.json").write_text(json.dumps(schema, indent=2) + "\n")
ext = source / "code/integration/torch_ext"
glue = (ext / "tq_native_read.cpp").read_text().replace("native_read_20261013/opp/vendors/tqread", "read_tiles_20261014/opp/vendors/tqreadv2").replace("TqPagedRead", "TqPagedReadV2").replace("tq_paged_read", "tq_paged_read_v2")
(ext / "tq_read_tiles.cpp").write_text(glue)
builder = (ext / "build_native_read.py").read_text().replace("tq_native_read", "tq_read_tiles")
(ext / "build_read_tiles.py").write_text(builder)
wrapper = source / "code/integration/kvtq_native_read.py"
wrapper.write_text(wrapper.read_text().replace("native_read_20261013/torch-extensions/tq_native_read/tq_native_read.so", "read_tiles_20261014/torch-extensions/tq_read_tiles/tq_read_tiles.so").replace("tq_paged_read", "tq_paged_read_v2"))
experiment = source / "experiments/read_tiles_20261014"
experiment.mkdir(parents=True, exist_ok=True)
shutil.copy2(Path(__file__), experiment / Path(__file__).name)
print(json.dumps({"branch": "perf/tq-read-tiles-20261014", "base": "c894ba84c9bb1325dd6b689b1cf41a9ce3f26f6c", "algorithm": "unchanged 66-byte/LUT/BF16 norm/RNE; optimized broadcast and scratch reuse"}))
