"""Reuse immutable read-index tables while preserving every decoding operation."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root.parent.parent / "source/tq-read-constants-20261014/kv_cache_turbo_quant"
    baseline = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant"
    for section, name in [("op_host", "tq_paged_read.cpp"), ("op_kernel", "tq_paged_read.cpp"),
                          ("op_kernel", "tq_paged_read_tiling.h")]:
        text = (baseline / "code/op" / section / name).read_text()
        text = text.replace("TqPagedRead", "TqPagedReadConst").replace("tq_paged_read", "tq_paged_read_const")
        if section == "op_host":
            text = text.replace("constexpr int64_t LOOKUP_ELEMENTS = 512;", "constexpr int64_t LOOKUP_ELEMENTS = 5696;  // LUT plus fixed uint32 offset tables, as BF16 bits")
        elif name.endswith(".cpp"):
            text = text.replace("data->jobs * 4);", "data->jobs * 4);")
            text = text.replace("lookup), LOOKUP_WORDS);", "lookup), LOOKUP_WORDS + CODE_WORDS + TILE_ROWS + OUTPUT_WORDS);")
            text = text.replace("        output_.SetGlobalBuffer", "        offsetLookup_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(lookup), LOOKUP_WORDS + CODE_WORDS + TILE_ROWS + OUTPUT_WORDS);\n        output_.SetGlobalBuffer")
            text = text.replace("    GlobalTensor<float> lookup_;", "    GlobalTensor<float> lookup_;\n    GlobalTensor<uint32_t> offsetLookup_;")
            start = text.index("        for (uint32_t i = 0; i < CODE_WORDS; ++i)")
            stop = text.index("        auto normOffsets = normExpandOffsets_.Get<uint32_t>();", start)
            text = text[:start] + """        // Immutable tables are supplied beside the centroid LUT. Each core
        // copies them in bulk instead of issuing thousands of scalar stores.
        DataCopy(codeOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS], CODE_WORDS);
        DataCopy(normOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS + CODE_WORDS], TILE_ROWS);
        DataCopy(joinOffsets_.Get<uint32_t>(), offsetLookup_[LOOKUP_WORDS + CODE_WORDS + TILE_ROWS], OUTPUT_WORDS);
""" + text[stop:]
        (source / "code/op" / section / name.replace("tq_paged_read", "tq_paged_read_const")).write_text(text)
    schema = json.loads((baseline / "code/op/TqPagedRead.json").read_text())
    schema[0]["op"] = "TqPagedReadConst"
    (source / "code/op/TqPagedReadConst.json").write_text(json.dumps(schema, indent=2) + "\n")
    ext = source / "code/integration/torch_ext"
    text = (baseline / "code/integration/torch_ext/tq_queued_read.cpp").read_text()
    text = text.replace("aclnnTqPagedRead", "aclnnTqPagedReadConst").replace("tq_paged_read_queued", "tq_paged_read_const")
    text = text.replace("/ws/native_read_20261013/opp/vendors/tqread", "/ws/read_constants_20261014/opp/vendors/tqreadconst")
    text = text.replace("lookup.numel() == 512", "lookup.numel() == 5696")
    (ext / "tq_read_constants.cpp").write_text(text)
    text = (baseline / "code/integration/torch_ext/build_queued_read.py").read_text().replace("tq_queued_read", "tq_read_constants")
    (ext / "build_read_constants.py").write_text(text)
    text = (baseline / "code/integration/kvtq_native_read.py").read_text()
    text = text.replace("queued_read_20261014", "read_constants_20261014").replace("tq_queued_read", "tq_read_constants").replace("tq_paged_read_queued", "tq_paged_read_const")
    needle = "    def __call__(self, cache, table, lengths, lookup, metadata=None, segment=\"decode\"):"
    assert text.count(needle) == 1
    text = text.replace(needle, """    def _prepare_lookup(self, lookup):
        source = getattr(self, "_lookup_source", None)
        if source is lookup and self._lookup_version == lookup._version:
            return self._prepared_lookup
        code = torch.arange(512, dtype=torch.int32)
        code = (code // 16) * 96 + (code % 16) * 4
        norm = torch.arange(32, dtype=torch.int32) * 96 + 64
        join = torch.arange(2048, dtype=torch.int32)
        join = ((join % 4) * 512 + join // 4) * 4
        indices = torch.cat([code, norm, join]).view(torch.bfloat16).to(lookup.device)
        self._prepared_lookup = torch.cat([lookup.flatten(), indices])
        self._lookup_source = lookup
        self._lookup_version = lookup._version
        return self._prepared_lookup

""" + needle)
    text = text.replace("jobs, lookup, total", "jobs, self._prepare_lookup(lookup), total")
    (source / "code/integration/kvtq_native_read.py").write_text(text)
    text = (root / "build_read_tiles.sh").read_text()
    for old, new in [("read_tiles_20261014", "read_constants_20261014"), ("TqPagedReadV2", "TqPagedReadConst"),
                     ("tq_paged_read_v2", "tq_paged_read_const"), ("tqreadv2", "tqreadconst"), ("build_read_tiles.py", "build_read_constants.py")]:
        text = text.replace(old, new)
    (root / "build_read_constants.sh").write_text(text)


if __name__ == "__main__":
    main()
