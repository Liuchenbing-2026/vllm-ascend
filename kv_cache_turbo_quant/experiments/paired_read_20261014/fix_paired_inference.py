"""Support inference tensors without hiding mutations or reading absent counters."""

from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root.parent.parent / "source/tq-paired-read-20261014/kv_cache_turbo_quant/code/integration"
    reader = source / "kvtq_native_pair.py"
    text = reader.read_text()
    start, end = text.index("    def _prepare_lookup"), text.index("    def __call__")
    text = text[:start] + """    def _prepare_lookup(self, lookup):
        # Inference tensors have no version counter. Repack their LUT prefix
        # on each call so untracked in-place edits cannot return stale values.
        version = None if torch.is_inference(lookup) else lookup._version
        source = getattr(self, "_lookup_source", None)
        if version is not None and source is lookup and self._lookup_version == version:
            return self._prepared_lookup
        if getattr(self, "_lookup_offsets_device", None) != lookup.device:
            code = torch.arange(512, dtype=torch.int32, device="cpu")
            code = (code // 16) * 96 + (code % 16) * 4
            norm = torch.arange(32, dtype=torch.int32, device="cpu") * 96 + 64
            join = torch.arange(2048, dtype=torch.int32, device="cpu")
            join = ((join % 4) * 512 + join // 4) * 4
            self._lookup_offsets = torch.cat([code, norm, join]).view(torch.bfloat16).to(lookup.device)
            self._lookup_offsets_device = lookup.device
        self._prepared_lookup = torch.cat([lookup.flatten(), self._lookup_offsets])
        self._lookup_source = lookup
        self._lookup_version = version
        return self._prepared_lookup

""" + text[end:]
    reader.write_text(text)
    store = source / "kvtq_store.py"
    text = store.read_text()
    start, end = text.index("def _get_byte_lut("), text.index("def _unpack_indices(")
    segment = text[start:end]
    if "with torch.inference_mode(False):" not in segment:
        first, last = segment.index("        cent = "), segment.index("    return _cache[key]")
        body = segment[first:last]
        segment = segment[:first] + "        # Keep the shared constant LUT versioned even in vLLM inference mode.\n        with torch.inference_mode(False):\n" + "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True)) + segment[last:]
    store.write_text(text[:start] + segment + text[end:])


if __name__ == "__main__":
    main()
