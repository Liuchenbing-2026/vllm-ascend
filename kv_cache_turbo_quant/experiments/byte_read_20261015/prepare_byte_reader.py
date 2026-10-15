"""Replace four byte-plane gathers with one ordered byte LUT gather."""
from pathlib import Path
import json
r=Path(__file__).resolve().parent
source=r.parent.parent/'source/tq-byte-read-20261015/kv_cache_turbo_quant'
for name in ['op_host/tq_paged_read_pair.cpp','op_kernel/tq_paged_read_pair.cpp','op_kernel/tq_paged_read_pair_tiling.h','TqPagedReadPair.json']:
 p=source/'code/op'/name;s=p.read_text().replace('TqPagedReadPair','TqPagedReadByte').replace('tq_paged_read_pair','tq_paged_read_byte')
 target=source/'code/op'/name.replace('TqPagedReadPair','TqPagedReadByte').replace('tq_paged_read_pair','tq_paged_read_byte');target.write_text(s)
p=source/'code/op/op_kernel/tq_paged_read_byte.cpp';s=p.read_text();s=s.replace('constexpr uint32_t OUTPUT_WORDS', 'constexpr uint32_t CODE_BYTES = TILE_ROWS * INDEX_BYTES;\nconstexpr uint32_t OUTPUT_WORDS')
for line in ['        pipe->InitBuffer(shifted_, CODE_WORDS * sizeof(uint32_t));\n','        pipe->InitBuffer(mask_, CODE_WORDS * sizeof(uint32_t));\n','        pipe->InitBuffer(planes_, OUTPUT_WORDS * sizeof(float));\n','        pipe->InitBuffer(joinOffsets_, OUTPUT_WORDS * sizeof(uint32_t));\n','        DataCopy(joinOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS + CODE_WORDS + TILE_ROWS], OUTPUT_WORDS);\n','        Duplicate(mask_.Get<uint32_t>(), 0xFFU, CODE_WORDS);\n']:
 assert line in s,line;s=s.replace(line,'')
s=s.replace('pipe->InitBuffer(index_, CODE_WORDS * sizeof(uint32_t));','pipe->InitBuffer(byteHalf_, CODE_BYTES * sizeof(half));\n        pipe->InitBuffer(index_, CODE_BYTES * sizeof(uint32_t));')
begin=s.index('        auto shifted = shifted_.Get<uint32_t>();');end=s.index('        auto values = values_.Get<float>();',begin)
s=s[:begin]+'''        // The contiguous code words preserve byte order in each 66-byte row.
        // All uint8 values are exactly representable in half and int32, so
        // these conversions change neither LUT indices nor stored centroids.
        auto halfBytes = byteHalf_.Get<half>();
        Cast(halfBytes, codes.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, CODE_BYTES);
        PipeBarrier<PIPE_V>();
        auto index = index_.Get<int32_t>();
        Cast(index, halfBytes, RoundMode::CAST_RINT, CODE_BYTES);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index.ReinterpretCast<uint32_t>(), index.ReinterpretCast<uint32_t>(), 2U, CODE_BYTES);
        PipeBarrier<PIPE_V>();
        auto joined = joined_.Get<float>();
        Gather(joined, lut_.Get<float>(), index.ReinterpretCast<uint32_t>(), 0, OUTPUT_WORDS);
        PipeBarrier<PIPE_V>();
'''+s[end:]
s=s.replace('lut_, codes_, shifted_, index_, mask_, codeOffsets_;','lut_, codes_, byteHalf_, index_, codeOffsets_;').replace('planes_, joinOffsets_, joined_, values_;','joined_, values_;');p.write_text(s)
for old,new in [('tq_paired_read.cpp','tq_byte_read.cpp'),('build_paired_read.py','build_byte_read.py')]:
 p=source/'code/integration/torch_ext'/old;s=p.read_text().replace('paired_read_20261014','byte_read_20261015').replace('tqreadpair','tqreadbyte').replace('TqPagedReadPair','TqPagedReadByte').replace('tq_paged_read_pair','tq_paged_read_byte').replace('tq_paired_read','tq_byte_read');(p.parent/new).write_text(s)
p=source/'code/integration/kvtq_native_pair.py';s=p.read_text();old='''    def __init__(self):
        if not hasattr(torch.ops.turboquant, "tq_paged_read_pair"):
            torch.ops.load_library(NATIVE_EXTENSION)
''';new='''    def __init__(self, operator_name="tq_paged_read_pair", extension_path=NATIVE_EXTENSION):
        if not hasattr(torch.ops.turboquant, operator_name):
            torch.ops.load_library(extension_path)
        self._operator = getattr(torch.ops.turboquant, operator_name)
''';assert old in s;s=s.replace(old,new).replace('return torch.ops.turboquant.tq_paged_read_pair(', 'return self._operator(');p.write_text(s)
p=source/'code/integration/kvtq_store.py';s=p.read_text();assert s.count('reader = NativePagedPairReader()')==1;s=s.replace('reader = NativePagedPairReader()', '''reader = NativePagedPairReader(
            operator_name="tq_paged_read_byte",
            extension_path="/ws/byte_read_20261015/torch-extensions/tq_byte_read/tq_byte_read.so",
        )''');p.write_text(s)
s=(r.parent/'tq-opt-20261014/build_paired_read.sh').read_text().replace('paired_read_20261014','byte_read_20261015').replace('TqPagedReadPair','TqPagedReadByte').replace('tq_paged_read_pair','tq_paged_read_byte').replace('tqreadpair','tqreadbyte').replace('build_paired_read.py','build_byte_read.py');(r/'build_byte_read.sh').write_text(s)
s=(r.parent/'tq-opt-20261014/paired_read_inference_20261014-snapshot/scripts/verify.py').read_text().replace('/ws/paired_read_inference_20261014','/ws/byte_read_20261015');s=s.replace('old = load("old", "/ws/native_combined_20261013/candidate_write/integration/kvtq_native_read.py").NativePagedReader()', 'old = NativePagedPairReader()');s=s.replace('reader = NativePagedPairReader()', 'reader = NativePagedPairReader(operator_name="tq_paged_read_byte", extension_path="/ws/byte_read_20261015/torch-extensions/tq_byte_read/tq_byte_read.so")');s=s.replace('(old(key, table, lengths, lookup, metadata), old(value, table, lengths, lookup, metadata))','old(key, value, table, lengths, lookup, metadata)').replace('(old(key, table, lengths, changed, metadata), old(value, table, lengths, changed, metadata))','old(key, value, table, lengths, changed, metadata)').replace('(old(key, table, lengths, source, metadata), old(value, table, lengths, source, metadata))','old(key, value, table, lengths, source, metadata)');compile(s,'verify.py','exec');(r/'verify_byte_read.py').write_text(s)
# Compare the unmodified byte-plane permutation against ordered byte indexing.
for seed in range(5):
 import random,struct
 rng=random.Random(seed);payload=bytes(rng.randrange(256) for _ in range(2048));words=struct.unpack('<512I',payload)
 planes=[[((word>>(byte*8))&255) for word in words] for byte in range(4)]
 joined=[planes[j%4][j//4] for j in range(2048)];assert joined==list(payload)
(r/'byte-layout-cpu.json').write_text(json.dumps({'random_layouts':5,'all_bytes_preserved':True,'old_plane_permutation_equals_contiguous_bytes':True,'scope':'integer layout only; NPU cast/kernel precision not yet verified'})+'\n')
print('prepared isolated byte reader; device validation pending')
