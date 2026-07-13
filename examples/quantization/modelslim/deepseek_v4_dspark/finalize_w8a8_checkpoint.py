#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Finalize a ModelSlim DeepSeek V4 DSpark W8A8 checkpoint for serving.

ModelSlim's global QuaRot pass rotates the decoder basis, including the draft
checkpoint's shared output head. The vLLM-Ascend integration consumes rotated
decoder states but a canonical shared head. This tool records those basis
semantics and replaces only ``head.weight`` with a compatible canonical copy.

Unchanged files can be symlinked, hard-linked, or copied. The patched shard is
always copied before modification, so the ModelSlim output is never mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEAD_TENSOR_NAME = "head.weight"
MAIN_PROJ_TENSOR_NAME = "mtp.0.main_proj.weight"
EXPECTED_HEAD_DTYPE = "F32"
COPY_BUFFER_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
INDEX_CANDIDATES = (
    "quant_model_weights.safetensors.index.json",
    "model.safetensors.index.json",
)
LINK_MODES = ("symlink", "hardlink", "copy")


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    data_offset: int
    data_length: int
    dtype: str
    shape: tuple[int, ...]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def find_checkpoint_index(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    for name in INDEX_CANDIDATES:
        path = checkpoint / name
        if path.is_file():
            index = load_json(path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"missing weight_map object in {path}")
            return path, index
    names = ", ".join(INDEX_CANDIDATES)
    raise FileNotFoundError(f"no checkpoint index found under {checkpoint}; expected one of: {names}")


def validate_shards(checkpoint: Path, index: dict[str, Any]) -> None:
    shard_names = index["weight_map"].values()
    if not all(isinstance(name, str) for name in shard_names):
        raise ValueError("checkpoint weight_map contains a non-string shard name")
    missing = sorted({name for name in shard_names if not (checkpoint / name).is_file()})
    if missing:
        raise FileNotFoundError(f"checkpoint has missing weight shards: {missing[:5]}")


def locate_tensor(checkpoint: Path, tensor_name: str) -> TensorLocation:
    _, index = find_checkpoint_index(checkpoint)
    weight_map = index["weight_map"]
    shard_name = weight_map.get(tensor_name)
    if not isinstance(shard_name, str):
        raise KeyError(f"{tensor_name!r} is absent from the checkpoint index at {checkpoint}")
    shard = checkpoint / shard_name
    if not shard.is_file():
        raise FileNotFoundError(f"missing shard for {tensor_name!r}: {shard}")

    file_size = shard.stat().st_size
    with shard.open("rb") as file:
        raw_header_length = file.read(8)
        if len(raw_header_length) != 8:
            raise ValueError(f"truncated safetensors length prefix: {shard}")
        header_length = struct.unpack("<Q", raw_header_length)[0]
        if header_length > MAX_SAFETENSORS_HEADER_BYTES or header_length > file_size - 8:
            raise ValueError(f"invalid safetensors header length {header_length} in {shard}")
        header = json.loads(file.read(header_length))

    entry = header.get(tensor_name)
    if not isinstance(entry, dict):
        raise KeyError(f"{tensor_name!r} is absent from its indexed shard: {shard}")
    offsets = entry.get("data_offsets")
    shape = entry.get("shape")
    dtype = entry.get("dtype")
    if not (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(isinstance(value, int) for value in offsets)
        and isinstance(shape, list)
        and all(isinstance(value, int) and value >= 0 for value in shape)
        and isinstance(dtype, str)
    ):
        raise ValueError(f"invalid safetensors metadata for {tensor_name!r} in {shard}")

    relative_start, relative_end = offsets
    data_offset = 8 + header_length + relative_start
    data_length = relative_end - relative_start
    if relative_start < 0 or data_length < 0 or data_offset + data_length > file_size:
        raise ValueError(f"out-of-range data offsets for {tensor_name!r} in {shard}")
    return TensorLocation(shard, data_offset, data_length, dtype, tuple(shape))


def validate_modelslim_output(checkpoint: Path) -> None:
    _, index = find_checkpoint_index(checkpoint)
    validate_shards(checkpoint, index)

    description = load_json(checkpoint / "quant_model_description.json")
    if description.get(MAIN_PROJ_TENSOR_NAME) != "W8A8_DYNAMIC":
        raise ValueError(f"{MAIN_PROJ_TENSOR_NAME} is not marked W8A8_DYNAMIC")

    try:
        rotation_name = description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except (KeyError, TypeError) as error:
        raise ValueError("the ModelSlim output does not declare a QuaRot global rotation") from error
    if not isinstance(rotation_name, str) or Path(rotation_name).is_absolute() or ".." in Path(rotation_name).parts:
        raise ValueError(f"invalid QuaRot rotation path: {rotation_name!r}")
    if not (checkpoint / rotation_name).is_file():
        raise FileNotFoundError(f"missing QuaRot rotation artifact: {checkpoint / rotation_name}")


def copy_tree_with_hardlinks(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)


def materialize_unchanged_entry(source: Path, destination: Path, link_mode: str) -> None:
    if link_mode == "symlink":
        relative_target = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_target, target_is_directory=source.is_dir())
    elif source.is_dir():
        if link_mode == "hardlink":
            copy_tree_with_hardlinks(source, destination)
        else:
            shutil.copytree(source, destination, symlinks=True)
    elif link_mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def copy_range_and_hash(source: TensorLocation, destination: TensorLocation) -> str:
    digest = hashlib.sha256()
    remaining = source.data_length
    with source.path.open("rb") as source_file, destination.path.open("r+b") as destination_file:
        source_file.seek(source.data_offset)
        destination_file.seek(destination.data_offset)
        while remaining:
            chunk = source_file.read(min(remaining, COPY_BUFFER_BYTES))
            if not chunk:
                raise OSError(f"unexpected EOF while reading {source.path}")
            destination_file.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        destination_file.flush()
        os.fsync(destination_file.fileno())
    return digest.hexdigest()


def hash_range(location: TensorLocation) -> str:
    digest = hashlib.sha256()
    remaining = location.data_length
    with location.path.open("rb") as file:
        file.seek(location.data_offset)
        while remaining:
            chunk = file.read(min(remaining, COPY_BUFFER_BYTES))
            if not chunk:
                raise OSError(f"unexpected EOF while verifying {location.path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def validate_head_compatibility(raw_head: TensorLocation, canonical_head: TensorLocation) -> None:
    raw_spec = (raw_head.dtype, raw_head.shape, raw_head.data_length)
    canonical_spec = (canonical_head.dtype, canonical_head.shape, canonical_head.data_length)
    if raw_spec != canonical_spec:
        raise ValueError(f"incompatible canonical head: raw={raw_spec}, canonical={canonical_spec}")
    if raw_head.dtype != EXPECTED_HEAD_DTYPE:
        raise ValueError(f"expected {HEAD_TENSOR_NAME} dtype {EXPECTED_HEAD_DTYPE}, got {raw_head.dtype}")


def write_config(source: Path, destination: Path) -> None:
    config = load_json(source)
    expected_markers = {
        "dspark_quarot_draft_basis": "rotated_decoder",
        "dspark_quarot_hc_head_basis": "canonical",
    }
    for key, expected in expected_markers.items():
        current = config.get(key)
        if current not in (None, expected):
            raise ValueError(f"refusing to replace unexpected {key} value: {current!r}")
    config["dspark_quarot_draft_basis"] = "rotated_decoder"
    config["dspark_quarot_hc_head_basis"] = "canonical"
    with destination.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")


def finalize_checkpoint(
    modelslim_output: Path,
    canonical_head_checkpoint: Path,
    output: Path,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    if link_mode not in LINK_MODES:
        raise ValueError(f"link_mode must be one of {LINK_MODES}, got {link_mode!r}")

    modelslim_output = modelslim_output.expanduser().resolve()
    canonical_head_checkpoint = canonical_head_checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    if not modelslim_output.is_dir():
        raise FileNotFoundError(f"ModelSlim output is not a directory: {modelslim_output}")
    if not canonical_head_checkpoint.is_dir():
        raise FileNotFoundError(f"canonical head checkpoint is not a directory: {canonical_head_checkpoint}")
    if canonical_head_checkpoint == modelslim_output:
        raise ValueError("the canonical head checkpoint must be distinct from the rotated ModelSlim output")
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite it: {output}")
    if modelslim_output == output or modelslim_output in output.parents:
        raise ValueError("output must not be the ModelSlim directory or a child of it")

    validate_modelslim_output(modelslim_output)
    raw_head = locate_tensor(modelslim_output, HEAD_TENSOR_NAME)
    canonical_head = locate_tensor(canonical_head_checkpoint, HEAD_TENSOR_NAME)
    validate_head_compatibility(raw_head, canonical_head)

    raw_head_shard_name = raw_head.path.name
    if raw_head.path.parent != modelslim_output:
        raise ValueError(f"the {HEAD_TENSOR_NAME} shard must be at the checkpoint root: {raw_head.path}")
    output.mkdir(parents=True)
    try:
        for entry in modelslim_output.iterdir():
            if entry.name in {"config.json", raw_head_shard_name}:
                continue
            materialize_unchanged_entry(entry, output / entry.name, link_mode)

        write_config(modelslim_output / "config.json", output / "config.json")
        temporary_shard = output / f".{raw_head_shard_name}.partial"
        shutil.copyfile(raw_head.path, temporary_shard)
        temporary_head = TensorLocation(
            temporary_shard,
            raw_head.data_offset,
            raw_head.data_length,
            raw_head.dtype,
            raw_head.shape,
        )
        source_digest = copy_range_and_hash(canonical_head, temporary_head)
        if hash_range(raw_head) == source_digest:
            raise ValueError("canonical head is byte-identical to the rotated raw head; refusing a no-op replacement")
        if hash_range(temporary_head) != source_digest:
            raise OSError("canonical head verification failed after copying")
        final_shard = output / raw_head_shard_name
        os.replace(temporary_shard, final_shard)

        output_head = locate_tensor(output, HEAD_TENSOR_NAME)
        if hash_range(output_head) != source_digest:
            raise OSError("canonical head verification failed after installing the patched shard")

        result: dict[str, Any] = {
            "output": str(output),
            "link_mode": link_mode,
            "draft_basis": "rotated_decoder",
            "hc_head_basis": "canonical",
            "head": {
                "tensor": HEAD_TENSOR_NAME,
                "shard": raw_head_shard_name,
                "dtype": raw_head.dtype,
                "shape": list(raw_head.shape),
                "bytes": raw_head.data_length,
                "sha256": source_digest,
            },
        }
        marker = output / ".dspark_w8a8_finalize.json"
        with marker.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return result
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelslim-output", type=Path, required=True)
    parser.add_argument("--canonical-head-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-mode", choices=LINK_MODES, default="symlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize_checkpoint(
        args.modelslim_output,
        args.canonical_head_checkpoint,
        args.output,
        args.link_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
