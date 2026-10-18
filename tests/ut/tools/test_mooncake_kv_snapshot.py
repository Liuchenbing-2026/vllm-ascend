# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from tools.mooncake_kv_snapshot import capture_pages, compare_snapshots


def caches():
    # Match the production interleaved K/V stride: each view skips its sibling.
    backing = torch.arange(6 * 2 * 4 * 2 * 3).reshape(6, 2, 4, 2, 3).to(torch.bfloat16)
    return {"layer.2": (backing[:, 0], backing[:, 1]), "layer.10": (backing[:, 0] + 1000, backing[:, 1] + 1000)}


def snapshot(root, values=None, names=None, blocks=None, tokens=6):
    values = caches() if values is None else values
    capture_pages(values, names or list(values), [4, 1] if blocks is None else blocks, tokens, 4, root, {})
    return root / "manifest.json"


def test_noncontiguous_physical_blocks_and_partial_tail_match_independent_tensor_reference(tmp_path):
    values = caches()
    manifest = json.loads(snapshot(tmp_path / "snapshot", values).read_text())
    assert len(manifest["entries"]) == 8
    for entry in manifest["entries"]:
        cache = values[entry["layer"]][0 if entry["kind"] == "K" else 1]
        block = [4, 1][entry["logical_block"]]
        expected = cache[block, : entry["valid_tokens"]].contiguous().view(torch.uint8).numpy().tobytes()
        assert (tmp_path / "snapshot" / entry["file"]).read_bytes() == expected
        assert entry["valid_tokens"] == (4 if entry["logical_block"] == 0 else 2)


def test_comparison_uses_layer_names_and_cache_kinds_not_registration_order(tmp_path):
    first = snapshot(tmp_path / "a")
    second = snapshot(tmp_path / "b", names=["layer.10", "layer.2"])
    assert compare_snapshots(first, second, numerical=True)["all_bytes_exact"]


def test_difference_is_reported_without_relaxing_byte_comparison(tmp_path):
    first = snapshot(tmp_path / "a")
    values = caches()
    values["layer.2"][0][4, 0, 0, 0] += 8
    second = snapshot(tmp_path / "b", values)
    result = compare_snapshots(first, second, numerical=True)
    changed = [entry for entry in result["results"] if not entry["bytes_exact"]]
    assert not result["all_bytes_exact"] and len(changed) == 1
    assert changed[0]["kind"] == "K" and changed[0]["different_values"] == 1
    assert changed[0]["max_abs"] == 8


@pytest.mark.parametrize("blocks,tokens", [([6, 1], 6), ([1, 1], 6), ([1], 6), ([-1, 1], 6), ([True, 1], 6)])
def test_invalid_physical_mappings_are_rejected_before_writing(tmp_path, blocks, tokens):
    with pytest.raises(ValueError):
        snapshot(tmp_path / "bad", blocks=blocks, tokens=tokens)
    assert not (tmp_path / "bad").exists()


def test_dtype_and_inconsistent_layer_mapping_are_rejected(tmp_path):
    values = {name: tuple(cache.float() for cache in pair) for name, pair in caches().items()}
    with pytest.raises(ValueError, match="BF16"):
        snapshot(tmp_path / "dtype", values)
    with pytest.raises(ValueError, match="layer names"):
        snapshot(tmp_path / "layers", names=["layer.2"])


def test_corrupt_dump_is_not_treated_as_tensor_difference(tmp_path):
    first = snapshot(tmp_path / "a")
    second = snapshot(tmp_path / "b")
    entry = json.loads(second.read_text())["entries"][0]
    (second.parent / entry["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="fingerprint"):
        compare_snapshots(first, second)


def test_existing_snapshot_and_incompatible_valid_range_are_rejected(tmp_path):
    first = snapshot(tmp_path / "a")
    with pytest.raises(FileExistsError):
        snapshot(tmp_path / "a")
    second = snapshot(tmp_path / "b", tokens=5)
    with pytest.raises(ValueError, match="token ranges"):
        compare_snapshots(first, second)


def test_missing_cache_page_cannot_make_two_incomplete_dumps_pass(tmp_path):
    first = snapshot(tmp_path / "a")
    second = snapshot(tmp_path / "b")
    for path in (first, second):
        manifest = json.loads(path.read_text())
        manifest["entries"].pop()
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        compare_snapshots(first, second)
