"""Combine K/V reading, retaining proven byte decoding and separate outputs."""

import copy
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root.parent.parent / "source/tq-paired-read-20261014/kv_cache_turbo_quant"
    constant = root.parent.parent / "source/tq-read-constants-20261014/kv_cache_turbo_quant"
    for section, name in [("op_host", "tq_paged_read_const.cpp"), ("op_kernel", "tq_paged_read_const.cpp"),
                          ("op_kernel", "tq_paged_read_const_tiling.h")]:
        text = (constant / "code/op" / section / name).read_text()
        text = text.replace("TqPagedReadConst", "TqPagedReadPair").replace("tq_paged_read_const", "tq_paged_read_pair")
        if section == "op_host":
            for old, new in [("GetInputShape(3)", "GetInputShape(4)"), ("GetInputShape(2)", "GetInputShape(3)"), ("GetInputShape(1)", "GetInputShape(2)")]:
                text = text.replace(old, new)
            text = text.replace("    const auto &c = cache->GetStorageShape();", """    const auto *value = context->GetInputShape(1);
    if (value == nullptr || value->GetStorageShape().GetDimNum() != 4) return ge::GRAPH_FAILED;
    const auto &c = cache->GetStorageShape();
    if (c.GetDimNum() != 4) return ge::GRAPH_FAILED;
    for (int i = 0; i < 4; ++i) {
        if (value->GetStorageShape().GetDim(i) != c.GetDim(i)) return ge::GRAPH_FAILED;
    }""")
            text = text.replace("j.GetDim(0) > UINT32_MAX", "j.GetDim(0) > UINT32_MAX / 2")
            text = text.replace("context->SetBlockDim(data->jobs < cores ? data->jobs : cores);", "context->SetBlockDim(data->jobs * 2 < cores ? data->jobs * 2 : cores);")
            text = text.replace("    return GRAPH_SUCCESS;\n}\nstatic ge::graphStatus InferDataType", "    *context->GetOutputShape(1) = *context->GetOutputShape(0);\n    return GRAPH_SUCCESS;\n}\nstatic ge::graphStatus InferDataType")
            text = text.replace("context->SetOutputDataType(0, DT_BF16);", "context->SetOutputDataType(0, DT_BF16);\n    context->SetOutputDataType(1, DT_BF16);")
            needle = '        this->Input("table")'
            text = text.replace(needle, '        this->Input("value_cache").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND})\n            .UnknownShapeFormat({ge::FORMAT_ND});\n' + needle)
            needle = '        this->Attr("tokens").Int();'
            text = text.replace(needle, '        this->Output("dense_value").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})\n            .UnknownShapeFormat({ge::FORMAT_ND});\n' + needle)
        elif name.endswith(".cpp"):
            text = text.replace("GM_ADDR cache, GM_ADDR table", "GM_ADDR cache, GM_ADDR value_cache, GM_ADDR table")
            text = text.replace("GM_ADDR lookup, GM_ADDR output,", "GM_ADDR lookup, GM_ADDR output, GM_ADDR value_output,")
            text = text.replace("        cache_.SetGlobalBuffer", "        cache_[0].SetGlobalBuffer")
            text = text.replace("        table_.SetGlobalBuffer", "        cache_[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(value_cache), data->cacheBytes);\n        table_.SetGlobalBuffer")
            text = text.replace("        output_.SetGlobalBuffer", "        output_[0].SetGlobalBuffer")
            text = text.replace("        pipe->InitBuffer(rawQueue_", "        output_[1].SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(value_output), data->outputElements);\n        pipe->InitBuffer(rawQueue_")
            needle = "        for (uint32_t job = GetBlockIdx(); job < data_->jobs; job += GetBlockNum()) {"
            assert text.count(needle) == 1
            text = text.replace(needle, """        for (uint32_t logicalJob = GetBlockIdx(); logicalJob < data_->jobs * 2; logicalJob += GetBlockNum()) {
            const uint32_t side = logicalJob / data_->jobs;
            const uint32_t job = logicalJob % data_->jobs;""")
            text = text.replace("RunTile(source, target, count);", "RunTile(source, target, count, side);")
            text = text.replace("uint32_t rows)", "uint32_t rows, uint32_t side)")
            text = text.replace("cache_[source]", "cache_[side][source]").replace("output_[target]", "output_[side][target]")
            text = text.replace("GlobalTensor<uint8_t> cache_;", "GlobalTensor<uint8_t> cache_[2];").replace("GlobalTensor<bfloat16_t> output_;", "GlobalTensor<bfloat16_t> output_[2];")
            text = text.replace("GM_ADDR lookup, GM_ADDR dense, GM_ADDR workspace", "GM_ADDR lookup, GM_ADDR dense, GM_ADDR dense_value, GM_ADDR workspace")
            text = text.replace("kernel.Init(&pipe, cache, table, jobs, lookup, dense, &data);", "kernel.Init(&pipe, cache, value_cache, table, jobs, lookup, dense, dense_value, &data);")
        (source / "code/op" / section / name.replace("tq_paged_read_const", "tq_paged_read_pair")).write_text(text)
    schema = json.loads((constant / "code/op/TqPagedReadConst.json").read_text())
    schema[0]["op"] = "TqPagedReadPair"
    value = copy.deepcopy(schema[0]["input_desc"][0]); value["name"] = "value_cache"; schema[0]["input_desc"].insert(1, value)
    output = copy.deepcopy(schema[0]["output_desc"][0]); output["name"] = "dense_value"; schema[0]["output_desc"].append(output)
    (source / "code/op/TqPagedReadPair.json").write_text(json.dumps(schema, indent=2) + "\n")
    text = (constant / "code/integration/torch_ext/tq_direct_constants.cpp").read_text()
    text = text.replace("#include <vector>", "#include <vector>\n#include <tuple>")
    text = text.replace("/ws/read_constants_20261014/opp/vendors/tqreadconst", "/ws/paired_read_20261014/opp/vendors/tqreadpair")
    text = text.replace("aclnnTqPagedReadConst", "aclnnTqPagedReadPair").replace("tq_paged_read_const_direct", "tq_paged_read_pair")
    text = text.replace("const aclTensor *,\n    int64_t, const aclTensor *, uint64_t", "const aclTensor *, const aclTensor *,\n    int64_t, const aclTensor *, const aclTensor *, uint64_t")
    text = text.replace("at::Tensor Read(const at::Tensor &cache, const at::Tensor &table", "std::tuple<at::Tensor, at::Tensor> Read(const at::Tensor &cache, const at::Tensor &value, const at::Tensor &table")
    text = text.replace("    TORCH_CHECK(table.dim()", "    TORCH_CHECK(value.sizes() == cache.sizes() && value.scalar_type() == cache.scalar_type() && value.device() == cache.device(), \"paired value cache mismatch\");\n    TORCH_CHECK(table.dim()")
    text = text.replace("    if (tokens == 0)", "    auto valueOutput = at::empty_like(output);\n    if (tokens == 0)").replace("return output;", "return std::make_tuple(output, valueOutput);")
    text = text.replace("tensors{Wrap(cache), Wrap(table), Wrap(jobs), Wrap(lookup), Wrap(output)}", "tensors{Wrap(cache), Wrap(value), Wrap(table), Wrap(jobs), Wrap(lookup), Wrap(output), Wrap(valueOutput)}")
    text = text.replace("api.workspace(tensors[0], tensors[1], tensors[2], tensors[3], tokens,\n        tensors[4]", "api.workspace(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tokens,\n        tensors[5], tensors[6]")
    text = text.replace("at::Tensor ReadMeta(const at::Tensor &cache, const at::Tensor &, const at::Tensor &,", "std::tuple<at::Tensor, at::Tensor> ReadMeta(const at::Tensor &cache, const at::Tensor &, const at::Tensor &, const at::Tensor &,")
    text = text.replace("    return at::empty({tokens, cache.size(2), 128}, cache.options().dtype(at::kBFloat16));", "    auto output = at::empty({tokens, cache.size(2), 128}, cache.options().dtype(at::kBFloat16));\n    return std::make_tuple(output, at::empty_like(output));")
    text = text.replace("(Tensor cache, Tensor table", "(Tensor cache, Tensor value, Tensor table").replace("int tokens) -> Tensor", "int tokens) -> (Tensor, Tensor)")
    (source / "code/integration/torch_ext/tq_paired_read.cpp").write_text(text)
    text = (constant / "code/integration/torch_ext/build_direct_constants.py").read_text().replace("tq_direct_constants", "tq_paired_read")
    (source / "code/integration/torch_ext/build_paired_read.py").write_text(text)
    text = (constant / "code/integration/kvtq_native_read.py").read_text().replace("NativePagedReader", "NativePagedPairReader")
    text = text.replace("read_direct_constants_20261014", "paired_read_20261014").replace("tq_direct_constants", "tq_paired_read").replace("tq_paged_read_const_direct", "tq_paged_read_pair")
    text = text.replace("def __call__(self, cache, table", "def __call__(self, cache, value, table")
    needle = "            return torch.empty(\n                (0, cache.shape[2], 128), dtype=torch.bfloat16, device=cache.device\n            )"
    assert needle in text
    text = text.replace(needle, "            empty = torch.empty((0, cache.shape[2], 128), dtype=torch.bfloat16, device=cache.device)\n            return empty, empty.clone()")
    text = text.replace("cache, table.contiguous(), jobs", "cache, value, table.contiguous(), jobs")
    (source / "code/integration/kvtq_native_pair.py").write_text(text)
    store = source / "code/integration/kvtq_store.py"
    text = (root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant/code/integration/kvtq_store.py").read_text()
    needle = "def _tq_decode(self, query, attn_metadata, output, num_decodes=None):"
    text = text.replace(needle, """def _dequant_dense_pair(self, table, lengths, metadata=None, segment='decode'):
    if _MSE_BITS != 4:
        return (_dequant_dense(self, self.key_cache, table, lengths, metadata, segment),
                _dequant_dense(self, self.value_cache, table, lengths, metadata, segment))
    reader = getattr(self, '_kvtq_pair_reader', None)
    if reader is None:
        from kvtq_native_pair import NativePagedPairReader
        reader = NativePagedPairReader()
        self._kvtq_pair_reader = reader
    return reader(self.key_cache.view(torch.uint8), self.value_cache.view(torch.uint8),
                  table, lengths, _get_byte_lut(self.key_cache.device), metadata, segment)


""" + needle)
    needle = "    k_dense = _dequant_dense(self, self.key_cache, block_table, seq_lens_list, attn_metadata)\n    v_dense = _dequant_dense(self, self.value_cache, block_table, seq_lens_list, attn_metadata)"
    assert needle in text
    text = text.replace(needle, "    k_dense, v_dense = _dequant_dense_pair(self, block_table, seq_lens_list, attn_metadata)")
    needle = "    k_dense = _dequant_dense(self, self.key_cache, prefill_bt, prefill_sl,\n                             attn_metadata, 'prefill')\n    v_dense = _dequant_dense(self, self.value_cache, prefill_bt, prefill_sl,\n                             attn_metadata, 'prefill')"
    assert needle in text
    text = text.replace(needle, "    k_dense, v_dense = _dequant_dense_pair(self, prefill_bt, prefill_sl, attn_metadata, 'prefill')")
    store.write_text(text)
    text = (root / "build_read_constants.sh").read_text()
    for old, new in [("read_constants_20261014", "paired_read_20261014"), ("TqPagedReadConst", "TqPagedReadPair"),
                     ("tq_paged_read_const", "tq_paged_read_pair"), ("tqreadconst", "tqreadpair"), ("build_read_constants.py", "build_paired_read.py")]:
        text = text.replace(old, new)
    (root / "build_paired_read.sh").write_text(text)


if __name__ == "__main__":
    main()
