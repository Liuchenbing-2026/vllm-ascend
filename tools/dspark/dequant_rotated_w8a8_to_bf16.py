#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantize the rotated W8A8 DSpark draft (mtp.*) to bf16 in place.

Why not just use an un-rotated bf16 draft checkpoint? Because the draft
attention (cv_wkv) and the target hidden states are QuaRot-rotated, while a
plain bf16 draft is in the un-rotated basis -- feeding it rotated hidden states
mismatches the basis and collapses acceptance to ~1.0. The verified-good basis
is the *rotated* one, so we keep it: dequantize the already-rotated int8 mtp.*
weights to bf16 (symmetric, per-output-channel scale) and copy every other
tensor through unchanged. The Markov head, norms, and hc_head parameters are
already bf16/fp32 and are passed through verbatim.

The output checkpoint reuses (symlinks) the target's ``optional/`` rotation
matrices so the runtime QuaRot alignment still finds ``global_rotation``.

Usage:
    python dequant_rotated_w8a8_to_bf16.py \
        --src /path/DeepSeek-V4-Flash-DSpark-w8a8 \
        --dst /path/DeepSeek-V4-Flash-DSpark-rbf16-draft
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# mtp.* weights that are stored as symmetric-quantized int8 with a companion
# ``<name>.weight_scale`` tensor of shape [out_features, 1].
_QUANT_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIX = ".weight_scale"


def _iter_shards(src: Path):
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        return [src / s for s in shards]
    return sorted(src.glob("*.safetensors"))


def _dequantize_shard(shard: Path, dst: Path) -> dict[str, str]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt") as f:
        keys = set(f.keys())
        for name in keys:
            if name.endswith(_SCALE_SUFFIX):
                # Consumed alongside its weight; drop from the bf16 output.
                continue
            weight = f.get_tensor(name)
            scale_name = name[: -len(_QUANT_WEIGHT_SUFFIX)] + _SCALE_SUFFIX
            is_quant_mtp = (
                name.startswith("mtp.")
                and name.endswith(_QUANT_WEIGHT_SUFFIX)
                and scale_name in keys
                and weight.dtype in (torch.int8, torch.uint8)
            )
            if is_quant_mtp:
                scale = f.get_tensor(scale_name).to(torch.float32)
                # Symmetric quant, offset 0: w_bf16 = int8 * scale (per out ch).
                dequant = weight.to(torch.float32) * scale.reshape(-1, 1)
                tensors[name] = dequant.to(torch.bfloat16)
            else:
                tensors[name] = weight
    out_shard = dst / shard.name
    save_file(tensors, out_shard, metadata={"format": "pt"})
    return {k: shard.name for k in tensors}


def _write_config(src: Path, dst: Path) -> None:
    config = json.loads((src / "config.json").read_text())
    config["architectures"] = ["DeepSeekV4DSpark"]
    config["dspark_mtp_dequantized_to_bf16"] = True
    # Drop the top-level quantization marker for the dequantized draft weights;
    # the runtime still reads optional/quarot for the rotation matrices.
    config.pop("quantization_config", None)
    (dst / "config.json").write_text(json.dumps(config, indent=2))


def _link_optional(src: Path, dst: Path) -> None:
    optional = src / "optional"
    if not optional.exists():
        return
    link = dst / "optional"
    if link.exists() or link.is_symlink():
        return
    try:
        os.symlink(optional, link, target_is_directory=True)
    except OSError:
        shutil.copytree(optional, link)


def _copy_aux_files(src: Path, dst: Path) -> None:
    for name in os.listdir(src):
        if name.endswith(".safetensors") or name == "config.json":
            continue
        s = src / name
        if s.is_dir():
            continue
        shutil.copy2(s, dst / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    src, dst = args.src, args.dst
    dst.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    for shard in _iter_shards(src):
        print(f"[dequant] {shard.name}")
        weight_map.update(_dequantize_shard(shard, dst))

    index = {"metadata": {}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    _write_config(src, dst)
    _copy_aux_files(src, dst)
    _link_optional(src, dst)
    print(f"[dequant] wrote rbf16 draft to {dst}")


if __name__ == "__main__":
    main()
