# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
TOOL_PATH = REPO_ROOT / "examples" / "quantization" / "modelslim" / "deepseek_v4_dspark" / "finalize_w8a8_checkpoint.py"
MODULE_NAME = "finalize_deepseek_v4_dspark_w8a8"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = TOOL
SPEC.loader.exec_module(TOOL)


def write_safetensors(path: Path, tensors: OrderedDict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)


def make_checkpoint(root: Path, head_shape: list[int], head_data: bytes, canonical: bool = False) -> None:
    root.mkdir()
    shard_name = "weights-00001-of-00001.safetensors"
    tensors: OrderedDict[str, tuple[str, list[int], bytes]] = OrderedDict()
    if canonical:
        tensors["prefix.weight"] = ("F32", [1], b"P" * 4)
    tensors["head.weight"] = ("F32", head_shape, head_data)
    tensors["norm.weight"] = ("F32", [1], b"N" * 4)
    write_safetensors(root / shard_name, tensors)
    weight_map = {name: shard_name for name in tensors}
    (root / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    if canonical:
        return
    (root / "config.json").write_text(json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}), encoding="utf-8")
    optional = root / "optional"
    optional.mkdir()
    (optional / "quarot.safetensors").write_bytes(b"rotation")
    description = {
        "mtp.0.main_proj.weight": "W8A8_DYNAMIC",
        "optional": {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}},
    }
    (root / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize("link_mode", ["copy", "hardlink"])
def test_finalize_checkpoint_replaces_only_head_and_sets_basis(tmp_path: Path, link_mode: str) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    output = tmp_path / "final"
    raw_head = b"R" * 16
    canonical_head = b"C" * 16
    make_checkpoint(raw, [2, 2], raw_head)
    make_checkpoint(canonical, [2, 2], canonical_head, canonical=True)

    raw_head_location = TOOL.locate_tensor(raw, "head.weight")
    raw_norm_location = TOOL.locate_tensor(raw, "norm.weight")
    raw_norm_hash = TOOL.hash_range(raw_norm_location)
    result = TOOL.finalize_checkpoint(raw, canonical, output, link_mode=link_mode)

    output_head = TOOL.locate_tensor(output, "head.weight")
    output_norm = TOOL.locate_tensor(output, "norm.weight")
    assert TOOL.hash_range(output_head) == TOOL.hash_range(TOOL.locate_tensor(canonical, "head.weight"))
    assert TOOL.hash_range(output_norm) == raw_norm_hash
    assert TOOL.hash_range(raw_head_location) != TOOL.hash_range(output_head)
    assert (output / "tokenizer.json").is_file()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["dspark_quarot_draft_basis"] == "rotated_decoder"
    assert config["dspark_quarot_hc_head_basis"] == "canonical"
    assert result["head"]["shape"] == [2, 2]
    assert json.loads((output / ".dspark_w8a8_finalize.json").read_text())["head"]["sha256"] == result["head"]["sha256"]


def test_finalize_checkpoint_rejects_incompatible_head_without_partial_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    output = tmp_path / "final"
    make_checkpoint(raw, [2, 2], b"R" * 16)
    make_checkpoint(canonical, [1, 4], b"C" * 16, canonical=True)

    with pytest.raises(ValueError, match="incompatible canonical head"):
        TOOL.finalize_checkpoint(raw, canonical, output, link_mode="copy")
    assert not output.exists()


def test_finalize_checkpoint_rejects_rotated_head_as_canonical(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    output = tmp_path / "final"
    head = b"R" * 16
    make_checkpoint(raw, [2, 2], head)
    make_checkpoint(canonical, [2, 2], head, canonical=True)

    with pytest.raises(ValueError, match="byte-identical"):
        TOOL.finalize_checkpoint(raw, canonical, output, link_mode="copy")
    assert not output.exists()
