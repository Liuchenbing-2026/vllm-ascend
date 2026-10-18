# SPDX-License-Identifier: Apache-2.0
"""Private, opt-in worker KV snapshots for TP1 full-attention diagnostics.

Load KVSnapshotWorkerExtension with --worker-extension-cls. The caller must
keep the selected request's pages retained and exclude concurrent requests.
This tool does not claim, release, register, or transfer any KV page.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def token_digest(tokens):
    return hashlib.sha256(b"".join(struct.pack("<I", token) for token in tokens)).hexdigest()


def capture_pages(caches, layer_names, block_ids, num_tokens, block_size, directory, provenance):
    # Import after worker/device initialization; CPU tests use the same helper.
    import torch

    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (num_tokens, block_size)):
        raise ValueError("positive integer token count and block size required")
    if not layer_names or len(set(layer_names)) != len(layer_names) or set(layer_names) != set(caches):
        raise ValueError("layer names must uniquely identify every registered cache")
    required = math.ceil(num_tokens / block_size)
    if len(block_ids) != required or len(set(block_ids)) != required:
        raise ValueError("one distinct physical block per logical block required")
    if any(isinstance(block, bool) or not isinstance(block, int) or block < 0 for block in block_ids):
        raise ValueError("invalid physical block ID")
    estimated_bytes = 0
    for name in layer_names:
        pair = caches[name]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("only separate K/V full-attention caches are supported")
        for cache in pair:
            if cache.dtype != torch.bfloat16 or cache.ndim != 4 or cache.shape[1] != block_size:
                raise ValueError("expected BF16 [blocks, block_size, heads, head_dim] cache")
            if max(block_ids) >= cache.shape[0]:
                raise ValueError("physical block ID exceeds cache storage")
            estimated_bytes += num_tokens * math.prod(cache.shape[2:]) * cache.element_size()
    max_snapshot_bytes = 2 * 1024**3
    if estimated_bytes > max_snapshot_bytes:
        raise ValueError("snapshot exceeds the diagnostic two-GiB limit")
    root = Path(directory)
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    entries = []
    for layer_index, name in enumerate(layer_names):
        for kind, cache in zip(("K", "V"), caches[name]):
            indices = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
            # index_select respects non-contiguous/interleaved K/V page strides.
            selected = cache.index_select(0, indices).to("cpu").contiguous()
            for logical_block, physical_block in enumerate(block_ids):
                valid = min(block_size, num_tokens - logical_block * block_size)
                page = selected[logical_block, :valid].contiguous()
                raw = page.view(torch.uint8).numpy().tobytes()
                filename = f"l{layer_index:03d}-{kind}-b{logical_block:05d}.bin"
                path = root / filename
                path.write_bytes(raw)
                path.chmod(0o600)
                entries.append(
                    {
                        "layer": name,
                        "kind": kind,
                        "logical_block": logical_block,
                        "physical_block": physical_block,
                        "valid_tokens": valid,
                        "shape": list(page.shape),
                        "source_stride": list(cache.stride()),
                        "dtype": "bfloat16",
                        "file": filename,
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    manifest = {
        "schema_version": 1,
        "scope": "worker-owned KV tensors; no performance claim",
        "num_tokens": num_tokens,
        "block_size": block_size,
        "layer_names": layer_names,
        "provenance": provenance,
        "entries": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o600)
    return {
        "manifest": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "entries": len(entries),
        "total_bytes": manifest["total_bytes"],
        "provenance": provenance,
    }


class KVSnapshotWorkerExtension:
    """Add a diagnostic RPC without replacing the worker/model/connector."""

    def mooncake_transport_endpoint(self):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group().connector_worker
        return {
            "engine_id": connector.engine_id,
            "host": connector.side_channel_host,
            "rpc_port": connector.te_rpc_port,
            "handshake_port": connector.handshake_port,
        }

    def capture_mooncake_kv(self, specification: str):
        import torch
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = json.loads(specification)
        connector = get_kv_transfer_group().connector_worker
        config = connector.vllm_config
        parallel = config.parallel_config
        sizes = (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
        groups = connector.kv_cache_config.kv_cache_groups
        if sizes != (1, 1, 1, 1, 1) or config.speculative_config is not None:
            raise ValueError("snapshot supports only TP/PP/DP/PCP/DCP=1 without speculation")
        if len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("snapshot requires a single full-attention cache group")
        if connector.engine_id != spec["engine_id"]:
            raise ValueError("source engine epoch does not match snapshot request")
        prefix = spec["prefix_token_ids"]
        if len(prefix) != spec["num_tokens"]:
            raise ValueError("expected prefix length does not match snapshot length")
        worker_state = {"available": False}
        state = self.model_runner.requests.get(spec["request_id"])
        if state is not None:
            actual = list(state.prompt_token_ids or []) + list(state.output_token_ids)
            if len(actual) < len(prefix) or actual[: len(prefix)] != prefix:
                raise ValueError("actual worker token prefix does not match fixture")
            worker_state = {
                "available": True,
                "prefix_exact": True,
                "prefix_sha256_u32le": token_digest(actual[: len(prefix)]),
                "num_computed_tokens": state.num_computed_tokens,
            }
        torch.npu.synchronize()
        return capture_pages(
            connector.kv_caches,
            list(groups[0].layer_names),
            spec["block_ids"],
            spec["num_tokens"],
            groups[0].kv_cache_spec.block_size,
            spec["directory"],
            {
                "engine_id": connector.engine_id,
                "request_id": spec["request_id"],
                "phase": spec["phase"],
                "expected_prefix_sha256_u32le": token_digest(prefix),
                "worker_state": worker_state,
                "ownership_precondition": "caller retains pages and excludes concurrent work",
            },
        )


def compare_snapshots(left_path, right_path, numerical=False):
    import numpy as np

    left_path, right_path = Path(left_path), Path(right_path)
    left, right = json.loads(left_path.read_text()), json.loads(right_path.read_text())
    if (left["num_tokens"], left["block_size"]) != (right["num_tokens"], right["block_size"]):
        raise ValueError("snapshot token ranges differ")
    indexed = []
    for manifest in (left, right):
        if not manifest["layer_names"] or len(set(manifest["layer_names"])) != len(manifest["layer_names"]):
            raise ValueError("invalid snapshot layer identity list")
        items = {(entry["layer"], entry["kind"], entry["logical_block"]): entry for entry in manifest["entries"]}
        if len(items) != len(manifest["entries"]):
            raise ValueError("duplicate layer/cache/logical-block entry")
        expected = {
            (layer, kind, block)
            for layer in manifest["layer_names"]
            for kind in ("K", "V")
            for block in range(math.ceil(manifest["num_tokens"] / manifest["block_size"]))
        }
        if items.keys() != expected:
            raise ValueError("incomplete snapshot layer/cache/logical-block coverage")
        indexed.append(items)
    if not indexed[0] or indexed[0].keys() != indexed[1].keys():
        raise ValueError("snapshot layer/cache/logical-block identities differ")
    results = []
    for key in sorted(indexed[0]):
        a, b = indexed[0][key], indexed[1][key]
        if (a["shape"], a["dtype"], a["valid_tokens"]) != (b["shape"], b["dtype"], b["valid_tokens"]):
            raise ValueError("snapshot valid shape/dtype differs")
        raw = []
        for path, entry in ((left_path, a), (right_path, b)):
            if entry["dtype"] != "bfloat16" or entry["bytes"] != 2 * math.prod(entry["shape"]):
                raise ValueError("snapshot must contain complete BF16 tensor bytes")
            if Path(entry["file"]).name != entry["file"]:
                raise ValueError("snapshot file must be a local basename")
            data = (path.parent / entry["file"]).read_bytes()
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("snapshot file fingerprint mismatch")
            raw.append(data)
        item = {"layer": key[0], "kind": key[1], "logical_block": key[2], "bytes_exact": raw[0] == raw[1]}
        if numerical:
            arrays = [(np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32) for data in raw]
            finite = all(np.isfinite(array).all() for array in arrays)
            item["all_finite"] = bool(finite)
            if finite:
                delta = arrays[0].astype(np.float64) - arrays[1].astype(np.float64)
                item.update(
                    max_abs=float(np.abs(delta).max()),
                    different_values=int(np.count_nonzero(delta)),
                    relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(arrays[1]), 1e-12)),
                )
        results.append(item)
    return {
        "left": str(left_path),
        "right": str(right_path),
        "blocks": len(results),
        "all_bytes_exact": all(item["bytes_exact"] for item in results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing results; choose a new output file")
    result = compare_snapshots(args.left, args.right, args.numerical)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}))


if __name__ == "__main__":
    main()
