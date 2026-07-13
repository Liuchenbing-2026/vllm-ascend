# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import json
import typing
from collections.abc import Iterable
from pathlib import Path

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend import envs
from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    _apply_dsv4_rope,
    _apply_dsv4_rope_tail,
    _grouped_wo_a_projection,
    _hc_head_torch,
    _linear_output,
    _make_deepseek_v4_expert_params_mapping,
    _wo_a_weight_for_eager_projection,
)
from vllm_ascend.ops.dspark_attention import (
    dspark_attention,
    dspark_attention_from_standard_cache,
    dspark_attention_from_standard_cache_sas,
)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
_LAYER_ID_RE = re.compile(r"model\.layers\.(\d+)\.")
_DSPARK_QUANT_STAGE_RE = re.compile(r"^model\.mtp\.(\d+)\.")
_DSPARK_DIRECT_QUANT_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_DSPARK_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.mtp\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_DSPARK_RAW_CHECKPOINT_TENSOR_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
_DSPARK_EP_FILTERABLE_EXPERT_WEIGHT_RE = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(?:w1|w2|w3)\.weight$")
_DSPARK_LOGICAL_QUANT_KEY_SUFFIXES = (
    ".indexer.quant_type",
    ".indexer.wq_b_weight",
)

_DSPARK_QUAROT_DRAFT_BASES = frozenset({"legacy", "unrotated", "rotated", "rotated_decoder"})
_DSPARK_QUAROT_HC_HEAD_BASES = frozenset({"canonical", "rotated"})
_DSPARK_QUAROT_DIRECTIONS = frozenset({"target_to_draft", "draft_to_target"})
_DSPARK_QUAROT_ORTHOGONALITY_PROBES = 16

_FP8_E4M3FN_SUBNORMAL_STEP = 2.0**-9
_FP8_E4M3FN_MIN_NORMAL = 2.0**-6
_FP8_E4M3FN_SUBNORMAL_NORMAL_MIDPOINT = (7 * _FP8_E4M3FN_SUBNORMAL_STEP + _FP8_E4M3FN_MIN_NORMAL) * 0.5


def _is_dspark_checkpoint_tensor_description_key(name: str) -> bool:
    """Return whether a stage-local entry requires a checkpoint tensor."""
    return ".confidence_head." not in name and not name.endswith(_DSPARK_LOGICAL_QUANT_KEY_SUFFIXES)


def _maybe_get_forward_context():
    try:
        return get_forward_context()
    except AssertionError:
        return None


def _draft_quant_config(vllm_config: VllmConfig):
    assert vllm_config.speculative_config is not None
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _target_quant_description(vllm_config: VllmConfig) -> dict[str, typing.Any] | None:
    quant_description = getattr(vllm_config.quant_config, "quant_description", None)
    return quant_description if isinstance(quant_description, dict) else None


def _draft_quant_description(vllm_config: VllmConfig) -> dict[str, typing.Any] | None:
    quant_config = _draft_quant_config(vllm_config)
    quant_description = getattr(quant_config, "quant_description", None)
    return quant_description if isinstance(quant_description, dict) else None


def _canonical_dspark_quant_entries(
    vllm_config: VllmConfig,
    quant_description: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Return only draft-layer entries in a stage-relative namespace.

    ModelSlim descriptions may address the draft either as ``model.mtp.N`` or
    directly as the expanded runtime layers. Both forms follow the lookup
    contract of :meth:`AscendModelSlimConfig.quant_prefix_mapper`; conflicting
    descriptions are rejected rather than selected silently.
    """
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    start_layer_idx = int(draft_config.num_hidden_layers)
    end_layer_idx = start_layer_idx + _get_dspark_num_mtp_layers(draft_config)
    stage_entries: dict[str, typing.Any] = {}

    def add_entry(canonical_name: str, quant_type: typing.Any, source_name: str) -> None:
        existing = stage_entries.get(canonical_name)
        if existing is not None and existing != quant_type:
            raise ValueError(
                "Conflicting DSpark quantization entries for "
                f"{source_name!r} and {canonical_name!r}: {quant_type!r} != {existing!r}."
            )
        stage_entries[canonical_name] = quant_type

    # Add stage-relative entries first, then compare direct runtime entries so
    # conflicts are diagnosed deterministically.
    for name, quant_type in quant_description.items():
        if _DSPARK_QUANT_STAGE_RE.match(name) is not None:
            add_entry(name, quant_type, name)
    for name, quant_type in quant_description.items():
        match = _DSPARK_DIRECT_QUANT_LAYER_RE.fullmatch(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if start_layer_idx <= layer_idx < end_layer_idx:
            canonical_name = f"model.mtp.{layer_idx - start_layer_idx}.{match.group(2)}"
            add_entry(canonical_name, quant_type, name)
    return stage_entries


def _dspark_main_proj_quant_type(
    vllm_config: VllmConfig,
    quant_description: dict[str, typing.Any],
) -> str | None:
    canonical_name = "model.mtp.0.main_proj.weight"
    checkpoint_name = "mtp.0.main_proj.weight"
    canonical_type = _canonical_dspark_quant_entries(vllm_config, quant_description).get(canonical_name)
    checkpoint_type = quant_description.get(checkpoint_name)
    if canonical_type is not None and checkpoint_type is not None and canonical_type != checkpoint_type:
        raise ValueError(
            "Conflicting DSpark main_proj quantization entries: "
            f"{canonical_name}={canonical_type!r}, {checkpoint_name}={checkpoint_type!r}."
        )
    quant_type = canonical_type if canonical_type is not None else checkpoint_type
    return typing.cast(str | None, quant_type)


def _draft_main_proj_quant_config(vllm_config: VllmConfig):
    """Select quantization for the DSpark context projection.

    The legacy DSpark path kept this projection floating point. Preserve that
    behavior when the quant description is absent or explicitly FLOAT,
    and opt into the existing W8 linear method only when the checkpoint says
    the projection is W8A8_DYNAMIC.  Dense per-channel W4 is intentionally not
    enabled because the current linear implementation requires a positive
    group size; routed-expert W4 remains supported by its dedicated MoE path.
    """
    quant_config = _draft_quant_config(vllm_config)
    quant_description = _draft_quant_description(vllm_config)
    if quant_config is None or quant_description is None:
        return None

    stage_entries = _canonical_dspark_quant_entries(vllm_config, quant_description)
    has_dynamic_draft_weights = any(
        name.endswith(".weight") and value in {"W8A8_DYNAMIC", "W4A8_DYNAMIC"} for name, value in stage_entries.items()
    )
    quant_type = _dspark_main_proj_quant_type(vllm_config, quant_description)
    if quant_type == "FLOAT":
        return None
    if quant_type is None:
        if has_dynamic_draft_weights:
            raise ValueError(
                "DSpark dynamic W8A8/W4A8 checkpoints must declare model.mtp.0.main_proj.weight "
                "as FLOAT or W8A8_DYNAMIC; the entry is missing."
            )
        return None
    if quant_type == "W8A8_DYNAMIC":
        return quant_config
    if quant_type == "W4A8_DYNAMIC":
        raise ValueError(
            "DSpark main_proj cannot use W4A8_DYNAMIC in the initial compatibility scope; "
            "keep dense projections W8A8_DYNAMIC and use W4A8_DYNAMIC only for routed experts."
        )
    raise ValueError(
        "DSpark main_proj cannot silently fall back to a floating-point implementation; "
        f"found unsupported quantization type {quant_type!r}."
    )


def _require_quant_companion(
    quant_description: dict[str, typing.Any],
    weight_name: str,
    suffixes: tuple[str, ...],
) -> str:
    base_name = weight_name.removesuffix(".weight")
    for suffix in suffixes:
        candidate = f"{base_name}.{suffix}"
        if candidate in quant_description:
            return candidate
    expected = " or ".join(f"{base_name}.{suffix}" for suffix in suffixes)
    raise ValueError(f"DSpark quantized weight {weight_name!r} is missing companion tensor {expected}.")


def _required_dspark_checkpoint_tensor_groups(
    quant_description: dict[str, typing.Any],
    *,
    start_layer_idx: int | None = None,
    num_dspark_layers: int | None = None,
) -> list[tuple[str, ...]]:
    """Build physical companion requirements for raw ModelSlim MTP weights.

    Some ModelSlim description versions list every physical tensor while
    others list only base weights. A tuple represents accepted source-name
    alternatives (notably ``weight_scale`` versus ``scale``).
    """
    required: set[tuple[str, ...]] = set()
    group_size = int(quant_description.get("group_size", 256) or 0)
    for name, quant_type in quant_description.items():
        checkpoint_name = _dspark_checkpoint_name_for_quant_key(
            name,
            start_layer_idx=start_layer_idx,
            num_dspark_layers=num_dspark_layers,
        )
        if checkpoint_name is None or not checkpoint_name.endswith(".weight"):
            continue
        if ".confidence_head." in checkpoint_name or quant_type not in {"W8A8_DYNAMIC", "W4A8_DYNAMIC"}:
            continue
        base_name = checkpoint_name.removesuffix(".weight")
        required.add((f"{base_name}.weight_scale", f"{base_name}.scale"))
        required.add((f"{base_name}.weight_offset",))
        if quant_type == "W4A8_DYNAMIC":
            required.add((f"{base_name}.scale_bias",))
            if group_size > 0:
                required.add((f"{base_name}.weight_scale_second",))
                required.add((f"{base_name}.weight_offset_second",))
    return sorted(required)


def _dspark_checkpoint_name_for_quant_key(
    name: str,
    *,
    start_layer_idx: int | None,
    num_dspark_layers: int | None,
) -> str | None:
    """Map raw, canonical, or direct quant keys to the DSpark checkpoint name."""
    raw_match = _DSPARK_RAW_CHECKPOINT_TENSOR_RE.fullmatch(name)
    if raw_match is not None:
        stage_idx = int(raw_match.group(1))
        remainder = raw_match.group(2)
    elif (stage_match := _DSPARK_QUANT_STAGE_RE.match(name)) is not None:
        stage_idx = int(stage_match.group(1))
        remainder = name[stage_match.end() :]
    else:
        direct_match = _DSPARK_DIRECT_QUANT_LAYER_RE.fullmatch(name)
        if direct_match is None or start_layer_idx is None or num_dspark_layers is None:
            return None
        layer_idx = int(direct_match.group(1))
        if not start_layer_idx <= layer_idx < start_layer_idx + num_dspark_layers:
            return None
        stage_idx = layer_idx - start_layer_idx
        remainder = direct_match.group(2)

    segments = remainder.split(".")
    root_mapping = {
        "self_attn": "attn",
        "mlp": "ffn",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
    }
    projection_mapping = {
        "gate_proj": "w1",
        "down_proj": "w2",
        "up_proj": "w3",
    }
    segments[0] = root_mapping.get(segments[0], segments[0])
    segments = [projection_mapping.get(segment, segment) for segment in segments]
    if segments == ["ffn", "gate", "e_score_correction_bias"]:
        segments[-1] = "bias"
    return f"mtp.{stage_idx}." + ".".join(segments)


def _dspark_checkpoint_weight_dtype_kinds(
    quant_description: dict[str, typing.Any],
    *,
    start_layer_idx: int,
    num_dspark_layers: int,
) -> dict[str, str]:
    """Map DSpark checkpoint base weights to fail-closed physical dtype classes."""
    expected: dict[str, str] = {}
    for name, quant_type in quant_description.items():
        checkpoint_name = _dspark_checkpoint_name_for_quant_key(
            name,
            start_layer_idx=start_layer_idx,
            num_dspark_layers=num_dspark_layers,
        )
        if checkpoint_name is None or not checkpoint_name.endswith(".weight"):
            continue
        if quant_type == "FLOAT":
            kind = "floating"
        elif quant_type in {"W8A8_DYNAMIC", "W4A8_DYNAMIC"}:
            kind = "int8"
        else:
            continue
        previous = expected.get(checkpoint_name)
        if previous is not None and previous != kind:
            raise ValueError(
                f"Conflicting DSpark physical dtype contracts for {checkpoint_name!r}: {previous!r} != {kind!r}."
            )
        expected[checkpoint_name] = kind
    return expected


def _validate_dspark_loaded_weight_dtype(name: str, tensor: torch.Tensor, expected_kind: str) -> None:
    if expected_kind == "floating" and tensor.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError(f"DSpark checkpoint tensor {name!r} is declared FLOAT but has physical dtype {tensor.dtype}.")
    if expected_kind == "int8" and tensor.dtype != torch.int8:
        raise ValueError(
            f"DSpark checkpoint tensor {name!r} is declared quantized but has physical dtype {tensor.dtype}; "
            "expected torch.int8 storage."
        )


def _validate_dspark_loaded_weight_shape(
    name: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"DSpark checkpoint tensor {name!r} has physical shape {actual_shape}; expected {expected_shape}."
        )


def _dspark_checkpoint_weight_shapes(config: typing.Any) -> dict[str, tuple[int, ...]]:
    """Return global checkpoint shapes that do not depend on TP/EP layout."""
    expected: dict[str, tuple[int, ...]] = {}
    hidden_size = getattr(config, "hidden_size", None)
    target_layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if isinstance(hidden_size, int) and hidden_size > 0 and target_layer_ids:
        expected["mtp.0.main_proj.weight"] = (
            hidden_size,
            hidden_size * len(target_layer_ids),
        )

    shape_values = {
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "o_groups": getattr(config, "o_groups", None),
        "o_lora_rank": getattr(config, "o_lora_rank", None),
    }
    if not all(isinstance(value, int) and value > 0 for value in shape_values.values()):
        return expected
    num_attention_heads = typing.cast(int, shape_values["num_attention_heads"])
    head_dim = typing.cast(int, shape_values["head_dim"])
    o_groups = typing.cast(int, shape_values["o_groups"])
    o_lora_rank = typing.cast(int, shape_values["o_lora_rank"])
    total_head_dim = num_attention_heads * head_dim
    if total_head_dim % o_groups != 0:
        raise ValueError(
            "Invalid DSpark attention shape contract: "
            f"num_attention_heads * head_dim ({total_head_dim}) is not divisible by o_groups ({o_groups})."
        )
    wo_a_shape = (o_groups * o_lora_rank, total_head_dim // o_groups)
    for stage_idx in range(_get_dspark_num_mtp_layers(config)):
        expected[f"mtp.{stage_idx}.attn.wo_a.weight"] = wo_a_shape
    return expected


def _missing_dspark_checkpoint_tensors(
    expected: set[str],
    seen: set[str],
    *,
    local_expert_ids_by_stage: dict[int, set[int]] | None,
) -> list[str]:
    """Return checkpoint tensors absent from the rank-local weight stream.

    vLLM's EP weight filter deliberately omits non-local numeric expert
    ``*.weight`` tensors before ``load_weights`` sees them, while retaining
    every scale, offset, and metadata tensor.  Do not mistake those filtered
    base weights for a malformed global checkpoint; all other physical
    tensors remain mandatory.
    """
    missing: list[str] = []
    for name in expected - seen:
        match = _DSPARK_EP_FILTERABLE_EXPERT_WEIGHT_RE.fullmatch(name)
        if match is not None and local_expert_ids_by_stage is not None:
            stage_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            local_ids = local_expert_ids_by_stage.get(stage_idx)
            if local_ids is not None and expert_idx not in local_ids:
                continue
        missing.append(name)
    return sorted(missing)


def _dspark_local_expert_ids_by_stage(model: typing.Any) -> dict[int, set[int]] | None:
    """Read the exact rank-local expert assignment from each draft layer."""
    layers = getattr(model, "layers", None)
    if layers is None:
        return None
    start_layer_idx = int(getattr(model, "mtp_start_layer_idx", 0))
    local_by_stage: dict[int, set[int]] = {}
    for layer_key, layer in layers.items():
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        manager = getattr(experts, "expert_map_manager", None)
        getter = getattr(manager, "get_local_expert_ids", None)
        if not callable(getter):
            return None
        stage_idx = int(layer_key) - start_layer_idx
        local_by_stage[stage_idx] = {int(expert_id) for expert_id in getter()}
    return local_by_stage


def _validate_dspark_checkpoint_index(
    model_path: str | None,
    expected_tensors: set[str],
    required_tensor_groups: list[tuple[str, ...]],
    *,
    expected_weight_dtype_kinds: dict[str, str] | None = None,
    expected_weight_shapes: dict[str, tuple[int, ...]] | None = None,
    require_mtp_float_weights: bool = False,
) -> bool:
    """Validate physical DSpark tensors against a local safetensors index.

    This global checkpoint check deliberately happens before vLLM exposes a
    rank-local weight iterator, which may omit non-local expert base weights.
    Tensor headers are checked without reading payloads so a stale or corrupt
    index cannot make the preflight pass silently.  ``False`` means no local
    indexed checkpoint was available and the caller must use its iterator
    fallback.
    """
    if not model_path:
        return False
    root = Path(model_path)
    if not root.is_dir():
        return False

    weight_map: dict[str, str] = {}
    for index_path in sorted(root.glob("*.safetensors.index.json")):
        with index_path.open(encoding="utf-8") as f:
            index_data = json.load(f)
        index_weight_map = index_data.get("weight_map")
        if not isinstance(index_weight_map, dict):
            continue
        for name, shard in index_weight_map.items():
            existing = weight_map.get(name)
            if existing is not None and existing != shard:
                raise ValueError(
                    f"DSpark checkpoint indexes map tensor {name!r} to conflicting shards: {existing!r} != {shard!r}."
                )
            weight_map[name] = shard
    if not weight_map:
        return False

    missing_groups = [group for group in required_tensor_groups if not any(name in weight_map for name in group)]
    if missing_groups:
        preview = [" or ".join(group) for group in missing_groups[:20]]
        raise ValueError(
            "DSpark quantized checkpoint index is missing required physical companion tensors: "
            f"{preview} (missing {len(missing_groups)} groups total)."
        )

    missing_tensors = sorted(expected_tensors - weight_map.keys())
    if missing_tensors:
        preview = missing_tensors[:20]
        suffix = "" if len(missing_tensors) <= len(preview) else " ..."
        raise ValueError(
            "DSpark ModelSlim source description references tensors absent from the checkpoint index: "
            f"{preview}{suffix} (missing {len(missing_tensors)} total)."
        )

    required_names = set(expected_tensors)
    for group in required_tensor_groups:
        required_names.add(next(name for name in group if name in weight_map))
    if require_mtp_float_weights:
        required_names.update(
            name
            for name in weight_map
            if _DSPARK_RAW_CHECKPOINT_TENSOR_RE.fullmatch(name) is not None and name.endswith(".weight")
        )
    names_by_shard: dict[str, set[str]] = {}
    for name in required_names:
        shard = weight_map[name]
        names_by_shard.setdefault(shard, set()).add(name)

    for shard, names in names_by_shard.items():
        shard_path = root / shard
        if not shard_path.is_file():
            raise ValueError(f"DSpark checkpoint index references missing shard {str(shard_path)!r}.")
        with safe_open(str(shard_path), framework="pt") as f:
            header_names = set(f.keys())
            missing_from_header = sorted(names - header_names)
            if missing_from_header:
                raise ValueError(
                    f"DSpark checkpoint shard {str(shard_path)!r} is missing indexed tensors: "
                    f"{missing_from_header[:20]}."
                )
            for name in names:
                tensor_slice = f.get_slice(name)
                shape = tuple(tensor_slice.get_shape())
                if name.endswith(".weight") and (not shape or any(dim <= 0 for dim in shape)):
                    raise ValueError(f"DSpark checkpoint tensor {name!r} has invalid physical shape {shape}.")
                expected_shape = (expected_weight_shapes or {}).get(name)
                if expected_shape is not None and shape != expected_shape:
                    raise ValueError(
                        f"DSpark checkpoint tensor {name!r} has physical shape {shape}; expected {expected_shape}."
                    )
                dtype = tensor_slice.get_dtype()
                expected_kind = (expected_weight_dtype_kinds or {}).get(name)
                if require_mtp_float_weights and name.endswith(".weight"):
                    expected_kind = "floating"
                if expected_kind == "floating" and dtype not in {"F16", "BF16", "F32", "F64"}:
                    raise ValueError(
                        f"DSpark checkpoint tensor {name!r} is declared FLOAT but has physical dtype {dtype}."
                    )
                if expected_kind == "int8" and dtype != "I8":
                    raise ValueError(
                        f"DSpark checkpoint tensor {name!r} is declared quantized but has physical dtype {dtype}; "
                        "expected I8 storage."
                    )
    return True


def _required_dspark_checkpoint_tensors(
    quant_description: dict[str, typing.Any],
    *,
    start_layer_idx: int,
    num_dspark_layers: int,
) -> set[str]:
    required: set[str] = set()
    for name in quant_description:
        if not _is_dspark_checkpoint_tensor_description_key(name):
            continue
        checkpoint_name = _dspark_checkpoint_name_for_quant_key(
            name,
            start_layer_idx=start_layer_idx,
            num_dspark_layers=num_dspark_layers,
        )
        if checkpoint_name is not None:
            required.add(checkpoint_name)
    return required


def _validate_dspark_quant_description(vllm_config: VllmConfig) -> None:
    """Fail before allocation when a DSpark ModelSlim contract is incomplete.

    This validator deliberately checks only invariants required by the paths
    enabled here.  It does not infer precision from the top-level model type:
    hybrid W4 checkpoints still describe dense layers as W8 and routed experts
    as W4 on a per-weight basis.
    """
    quant_description = _draft_quant_description(vllm_config)
    if quant_description is None:
        return

    # Validate main_proj even when the manifest uses only its raw checkpoint
    # name and therefore has no canonical stage entries.
    _draft_main_proj_quant_config(vllm_config)

    stage_entries = _canonical_dspark_quant_entries(vllm_config, quant_description)
    stage_ids = {
        int(match.group(1)) for name in stage_entries if (match := _DSPARK_QUANT_STAGE_RE.match(name)) is not None
    }
    if not stage_ids:
        return

    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    expected_stage_ids = set(range(_get_dspark_num_mtp_layers(draft_config)))

    # wo_a bypasses the quant method in the specialized attention projection.
    # Guard it for every ModelSlim format, including formats outside the strict
    # W8A8_DYNAMIC/W4A8_DYNAMIC profile validated below.
    for stage_idx in sorted(stage_ids):
        wo_a_name = f"model.mtp.{stage_idx}.self_attn.wo_a.weight"
        wo_a_type = stage_entries.get(wo_a_name)
        if wo_a_type is not None and wo_a_type != "FLOAT":
            raise ValueError(
                f"DSpark {wo_a_name} must remain FLOAT because its specialized projection reads the raw weight; "
                f"found {wo_a_type!r}."
            )

    supported_weight_types = {"FLOAT", "W8A8_DYNAMIC", "W4A8_DYNAMIC"}
    unsupported_entries = [
        (name, quant_type)
        for name, quant_type in stage_entries.items()
        if name.endswith(".weight") and quant_type not in supported_weight_types
    ]
    if unsupported_entries:
        preview = ", ".join(f"{name}={quant_type!r}" for name, quant_type in unsupported_entries[:20])
        raise ValueError(f"DSpark checkpoint contains unsupported draft weight quantization types: {preview}.")
    weight_types = {quant_type for name, quant_type in stage_entries.items() if name.endswith(".weight")}
    is_dynamic_profile = bool(weight_types & {"W8A8_DYNAMIC", "W4A8_DYNAMIC"})
    if not is_dynamic_profile:
        return

    if stage_ids != expected_stage_ids:
        raise ValueError(
            "DSpark quantization description stage mismatch: "
            f"expected {sorted(expected_stage_ids)}, found {sorted(stage_ids)}."
        )

    companion_suffixes = (
        "weight_scale",
        "scale",
        "weight_offset",
        "weight_scale_second",
        "weight_offset_second",
        "scale_bias",
    )
    has_companion_manifest = any(
        any(f"{name.removesuffix('.weight')}.{suffix}" in stage_entries for suffix in companion_suffixes)
        for name, quant_type in stage_entries.items()
        if name.endswith(".weight") and quant_type in {"W8A8_DYNAMIC", "W4A8_DYNAMIC"}
    )
    group_size_value = quant_description.get("group_size")
    group_size = int(group_size_value or 0)
    w4_weights: list[str] = []
    for name, quant_type in stage_entries.items():
        if _DSPARK_QUANT_STAGE_RE.match(name) is None or not name.endswith(".weight"):
            continue
        if quant_type == "FLOAT":
            continue

        if has_companion_manifest:
            _require_quant_companion(stage_entries, name, ("weight_scale", "scale"))
            _require_quant_companion(stage_entries, name, ("weight_offset",))
        if quant_type == "W4A8_DYNAMIC":
            if _DSPARK_EXPERT_WEIGHT_RE.fullmatch(name) is None:
                raise ValueError(f"DSpark W4A8_DYNAMIC is supported only for routed experts, but found {name!r}.")
            if group_size_value is None:
                raise ValueError("DSpark W4A8_DYNAMIC checkpoints must declare group_size explicitly.")
            if has_companion_manifest:
                _require_quant_companion(stage_entries, name, ("scale_bias",))
                if group_size > 0:
                    _require_quant_companion(stage_entries, name, ("weight_scale_second",))
                    _require_quant_companion(stage_entries, name, ("weight_offset_second",))
            w4_weights.append(name)

    if w4_weights and quant_description.get("version") != "1.0.0":
        raise ValueError(
            "DSpark W4A8_DYNAMIC checkpoints must use ModelSlim version '1.0.0'; "
            f"found {quant_description.get('version')!r}."
        )

    num_experts = int(draft_config.n_routed_experts)
    projections = ("gate_proj", "up_proj", "down_proj")
    for stage_idx in sorted(expected_stage_ids):
        wo_a_name = f"model.mtp.{stage_idx}.self_attn.wo_a.weight"
        wo_a_type = stage_entries.get(wo_a_name)
        if wo_a_type != "FLOAT":
            raise ValueError(
                f"DSpark {wo_a_name} must remain FLOAT because its specialized projection reads the raw weight; "
                f"found {wo_a_type!r}."
            )

        stage_expert_type: str | None = None
        for expert_idx in range(num_experts):
            expert_types: list[str] = []
            for projection in projections:
                name = f"model.mtp.{stage_idx}.mlp.experts.{expert_idx}.{projection}.weight"
                quant_type = stage_entries.get(name)
                if quant_type is None:
                    raise ValueError(
                        f"DSpark stage {stage_idx} expert {expert_idx} is missing {projection}.weight metadata."
                    )
                expert_types.append(typing.cast(str, quant_type))
            if len(set(expert_types)) != 1:
                raise ValueError(
                    f"DSpark stage {stage_idx} expert {expert_idx} has mixed projection precision: {expert_types}."
                )
            expert_type = expert_types[0]
            if stage_expert_type is None:
                stage_expert_type = expert_type
            elif expert_type != stage_expert_type:
                raise ValueError(
                    f"DSpark stage {stage_idx} mixes routed-expert precision: "
                    f"expected {stage_expert_type}, expert {expert_idx} uses {expert_type}."
                )

    # main_proj was validated before the stage checks so raw checkpoint-style
    # manifests cannot bypass the dense-projection policy.


def _get_dspark_quarot_draft_basis(vllm_config: VllmConfig) -> str:
    """Resolve the explicit coordinate contract for QuaRot draft weights.

    Row-vector convention: ``h_rotated = h_unrotated @ Q``.

    ``legacy`` preserves the existing contract: shared embedding/head states
    cross the Q boundary at runtime, while ``main_proj`` is expected to be a
    checkpoint bridge from rotated target states to the unrotated draft.
    ``unrotated`` additionally applies Q^T to every target-state block before
    an untouched, unrotated ``main_proj``.
    ``rotated`` means ModelSlim transformed the complete draft (including
    ``main_proj``) into the target's rotated basis, so no runtime boundary
    transforms are applied. ``rotated_decoder`` describes the split contract
    used by ModelSlim DSpark checkpoints whose shared embedding/head and
    decoder residual stream are rotated while ``main_proj`` and ``main_norm``
    remain canonical.  Context blocks cross into the canonical projection and
    its normalized output crosses back into the rotated decoder at runtime.
    The independently declared ``dspark_quarot_hc_head_basis`` controls the
    final multi-hidden HC-head boundary.
    """
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    config_basis = getattr(draft_config, "dspark_quarot_draft_basis", None)
    # QuaRot describes the target checkpoint and remains relevant when the MTP
    # tensors themselves were dequantized to BF16. Keep this metadata source
    # aligned with _load_dspark_quarot_rotation().
    quant_description = _target_quant_description(vllm_config) or {}
    quarot_metadata = quant_description.get("optional", {}).get("quarot", {})
    metadata_basis = quarot_metadata.get("dspark_draft_basis")
    if config_basis is not None and metadata_basis is not None and config_basis != metadata_basis:
        raise ValueError(
            "Conflicting dspark_quarot_draft_basis values in model config and ModelSlim metadata: "
            f"{config_basis!r} != {metadata_basis!r}."
        )

    explicit_basis = config_basis if config_basis is not None else metadata_basis
    draft_basis = explicit_basis or "legacy"
    if draft_basis not in _DSPARK_QUAROT_DRAFT_BASES:
        raise ValueError(
            f"Invalid dspark_quarot_draft_basis={draft_basis!r}; expected one of {sorted(_DSPARK_QUAROT_DRAFT_BASES)}."
        )

    rotation_path = quarot_metadata.get("rotation_map", {}).get("global_rotation")
    if explicit_basis is not None and draft_basis != "legacy" and rotation_path is None:
        raise ValueError(f"dspark_quarot_draft_basis={draft_basis!r} requires QuaRot global_rotation metadata.")
    if rotation_path is not None and explicit_basis is None:
        raise ValueError(
            "A DSpark checkpoint with QuaRot requires an explicit "
            "dspark_quarot_draft_basis ('legacy', 'unrotated', 'rotated', or 'rotated_decoder') "
            "to avoid a silent basis mismatch."
        )
    return typing.cast(str, draft_basis)


def _get_dspark_quarot_hc_head_basis(
    vllm_config: VllmConfig,
    *,
    draft_basis: str,
) -> str:
    """Resolve the HC-head coordinates independently from the decoder.

    A split QuaRot checkpoint can rotate the decoder residual stream while
    leaving the nested ``mtp.*.hc_head_fn`` tensor canonical.  Require that
    ambiguous ``rotated_decoder`` checkpoints state this contract explicitly
    so a future checkpoint with a correctly rotated HC head is not silently
    transformed twice.
    """
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    config_basis = getattr(draft_config, "dspark_quarot_hc_head_basis", None)
    quant_description = _target_quant_description(vllm_config) or {}
    metadata_basis = quant_description.get("optional", {}).get("quarot", {}).get("dspark_hc_head_basis")
    if config_basis is not None and metadata_basis is not None and config_basis != metadata_basis:
        raise ValueError(
            "Conflicting dspark_quarot_hc_head_basis values in model config and ModelSlim metadata: "
            f"{config_basis!r} != {metadata_basis!r}."
        )

    explicit_basis = config_basis if config_basis is not None else metadata_basis
    if explicit_basis is None:
        if draft_basis == "rotated_decoder":
            raise ValueError(
                "dspark_quarot_draft_basis='rotated_decoder' requires an explicit "
                "dspark_quarot_hc_head_basis ('canonical' or 'rotated') because "
                "ModelSlim checkpoints may leave the nested HC head unrotated."
            )
        return "rotated" if draft_basis == "rotated" else "canonical"

    if explicit_basis not in _DSPARK_QUAROT_HC_HEAD_BASES:
        raise ValueError(
            f"Invalid dspark_quarot_hc_head_basis={explicit_basis!r}; "
            f"expected one of {sorted(_DSPARK_QUAROT_HC_HEAD_BASES)}."
        )
    if draft_basis in {"legacy", "unrotated"} and explicit_basis != "canonical":
        raise ValueError(
            f"dspark_quarot_hc_head_basis={explicit_basis!r} is incompatible with "
            f"canonical decoder basis {draft_basis!r}."
        )
    return typing.cast(str, explicit_basis)


def _should_apply_dspark_fp8_qdq(config: PretrainedConfig) -> bool:
    return not (
        getattr(config, "dspark_mtp_dequantized_to_bf16", False)
        or getattr(config, "dspark_full_dequantized_to_bf16", False)
    )


def _load_dspark_quarot_rotation(
    vllm_config: VllmConfig,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    quant_config = vllm_config.quant_config
    quant_description = getattr(quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return None
    try:
        rotation_relative_path = quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except KeyError:
        return None

    rotation_path = Path(vllm_config.model_config.model) / rotation_relative_path
    tensors = load_file(rotation_path, device="cpu")
    if "global_rotation" not in tensors:
        raise ValueError(f"DSpark QuaRot file {rotation_path} does not contain 'global_rotation'.")
    rotation = tensors["global_rotation"]
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    draft_config = getattr(draft_model_config, "hf_config", None)
    hidden_size = getattr(draft_config, "hidden_size", None)
    expected_shape = (hidden_size, hidden_size) if hidden_size is not None else None
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"DSpark QuaRot global_rotation must be square, found shape={tuple(rotation.shape)}.")
    if expected_shape is not None and tuple(rotation.shape) != expected_shape:
        raise ValueError(
            "DSpark QuaRot global_rotation shape must match hidden_size: "
            f"expected {expected_shape}, found {tuple(rotation.shape)}."
        )
    if not rotation.is_floating_point():
        raise ValueError(f"DSpark QuaRot global_rotation must use a floating dtype, found {rotation.dtype}.")
    if not torch.isfinite(rotation).all():
        raise ValueError("DSpark QuaRot global_rotation contains non-finite values.")
    rotation_float = rotation.to(torch.float32)
    low_precision = rotation.dtype in {torch.float16, torch.bfloat16}
    norm_tolerance = 2e-2 if low_precision else 5e-3
    row_norms = rotation_float.square().sum(dim=1)
    column_norms = rotation_float.square().sum(dim=0)
    if not torch.allclose(
        row_norms,
        torch.ones_like(row_norms),
        rtol=0,
        atol=norm_tolerance,
    ) or not torch.allclose(
        column_norms,
        torch.ones_like(column_norms),
        rtol=0,
        atol=norm_tolerance,
    ):
        raise ValueError("DSpark QuaRot global_rotation is not orthonormal: row/column norms differ from one.")

    # A full n-by-n Gram matrix is too expensive in every TP worker. Fixed-seed
    # full-dimensional probes cover every row and column and detect corruption
    # with high probability while keeping validation O(k*n^2), k << n.
    probe_count = min(int(rotation.shape[0]), _DSPARK_QUAROT_ORTHOGONALITY_PROBES)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xD5A4C)
    # Workers install the NPU as PyTorch's default device before model
    # construction.  Validation stays on the CPU (where safetensors loaded the
    # rotation), so the probe tensor must not inherit that process-wide default.
    probes = torch.empty((rotation.shape[0], probe_count), dtype=torch.float32, device="cpu")
    probes.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
    reconstructed = rotation_float.t().matmul(rotation_float.matmul(probes))
    probe_error = float((reconstructed - probes).abs().max().item())
    probe_tolerance = 5e-2 if low_precision else 5e-3
    if probe_error > probe_tolerance:
        raise ValueError(
            "DSpark QuaRot global_rotation is not orthonormal: "
            f"Q.T @ Q probe max_abs_error={probe_error:.6g} exceeds {probe_tolerance:.6g}."
        )
    rotation = rotation.to(device=device, dtype=torch.float32)
    logger.info_once("Loaded DSpark QuaRot rotation from %s", rotation_path)
    return rotation


def _apply_dspark_quarot_rotation(
    hidden_states: torch.Tensor,
    rotation: torch.Tensor | None,
    transpose: bool,
) -> torch.Tensor:
    if rotation is None:
        return hidden_states
    rotation = rotation.t() if transpose else rotation
    if rotation.device != hidden_states.device:
        raise RuntimeError("DSpark QuaRot rotation must be loaded on the same device as hidden states")
    return torch.matmul(
        hidden_states.to(torch.float32),
        rotation,
    ).to(hidden_states.dtype)


def _derive_dspark_rotated_hc_head_fn(
    canonical_hc_head_fn: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Fold a canonical HC-head input boundary into a derived weight.

    The HC input is a concatenation of ``hc_mult`` residual branches.  If
    each branch is represented as ``x_rot = x @ Q``, then folding ``Q`` into
    every input block of ``hc_head_fn`` preserves its mixing logits while
    avoiding two full hidden-state rotations on every draft step.  This
    helper is deliberately functional: callers keep the checkpoint Parameter
    canonical and install the returned tensor as a derived buffer.
    """
    if canonical_hc_head_fn.ndim != 2:
        raise ValueError(
            f"DSpark canonical HC-head weight must be rank 2, found shape={tuple(canonical_hc_head_fn.shape)}."
        )
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(
            f"DSpark QuaRot rotation used for HC-head folding must be square, found shape={tuple(rotation.shape)}."
        )
    hc_mult, input_width = canonical_hc_head_fn.shape
    hidden_size = rotation.shape[0]
    expected_width = hc_mult * hidden_size
    if input_width != expected_width:
        raise ValueError(
            "DSpark canonical HC-head width must equal hc_mult * hidden_size: "
            f"expected {expected_width}, found {input_width}."
        )
    if canonical_hc_head_fn.device != rotation.device:
        raise RuntimeError(
            "DSpark canonical HC-head weight and QuaRot rotation must be on "
            f"the same device, found {canonical_hc_head_fn.device} and {rotation.device}."
        )
    if canonical_hc_head_fn.dtype != rotation.dtype:
        raise TypeError(
            "DSpark canonical HC-head weight and QuaRot rotation must have the "
            f"same dtype, found {canonical_hc_head_fn.dtype} and {rotation.dtype}."
        )
    if not canonical_hc_head_fn.is_floating_point() or not rotation.is_floating_point():
        raise TypeError("DSpark HC-head QuaRot folding requires floating-point weight and rotation tensors.")

    canonical_blocks = canonical_hc_head_fn.reshape(hc_mult, hc_mult, hidden_size)
    return torch.matmul(canonical_blocks, rotation).reshape_as(canonical_hc_head_fn).contiguous()


def _transition_dspark_quarot_basis(
    hidden_states: torch.Tensor,
    rotation: torch.Tensor | None,
    *,
    draft_basis: str,
    direction: str,
) -> torch.Tensor:
    if draft_basis not in _DSPARK_QUAROT_DRAFT_BASES:
        raise ValueError(f"Invalid DSpark QuaRot draft basis: {draft_basis!r}.")
    if direction not in _DSPARK_QUAROT_DIRECTIONS:
        raise ValueError(f"Invalid DSpark QuaRot transition direction: {direction!r}.")
    if rotation is None or draft_basis in {"rotated", "rotated_decoder"}:
        return hidden_states
    return _apply_dspark_quarot_rotation(
        hidden_states,
        rotation,
        transpose=direction == "target_to_draft",
    )


def _prepare_dspark_main_proj_input(
    context_states: torch.Tensor,
    rotation: torch.Tensor | None,
    *,
    draft_basis: str,
    hidden_size: int,
) -> torch.Tensor:
    if rotation is None or draft_basis not in {"unrotated", "rotated_decoder"}:
        return context_states
    if context_states.shape[-1] % hidden_size != 0:
        raise ValueError(
            "DSpark context width must contain complete target hidden-state blocks: "
            f"width={context_states.shape[-1]}, hidden_size={hidden_size}."
        )
    original_shape = context_states.shape
    target_blocks = context_states.reshape(*original_shape[:-1], -1, hidden_size)
    target_blocks = _apply_dspark_quarot_rotation(target_blocks, rotation, transpose=True)
    return target_blocks.reshape(original_shape)


def _prepare_dspark_main_proj_output(
    projected_states: torch.Tensor,
    rotation: torch.Tensor | None,
    *,
    draft_basis: str,
) -> torch.Tensor:
    """Move a canonical ``main_norm`` result into a rotated decoder.

    ``rotated_decoder`` checkpoints deliberately keep ``main_proj`` and
    ``main_norm`` canonical even though the decoder consumes rotated hidden
    states.  The transform belongs after normalization: moving it before
    ``main_norm`` would apply canonical per-channel gamma in the wrong basis.
    """
    if draft_basis not in _DSPARK_QUAROT_DRAFT_BASES:
        raise ValueError(f"Invalid DSpark QuaRot draft basis: {draft_basis!r}.")
    if rotation is None or draft_basis != "rotated_decoder":
        return projected_states
    return _apply_dspark_quarot_rotation(projected_states, rotation, transpose=False)


def _compute_dspark_hc_head(
    hidden_states: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    rotation: torch.Tensor | None,
    *,
    draft_basis: str,
    hc_head_basis: str,
) -> torch.Tensor:
    """Evaluate a canonical HC head at the rotated-decoder boundary.

    ModelSlim's split DSpark QuaRot contract rotates each decoder residual
    branch but leaves the nested ``mtp.*.hc_head_fn`` tensor canonical.  The
    HC mixing coefficients must therefore be computed from canonical branch
    coordinates.  Its weighted sum is then rotated back so the downstream
    norm and shared target head continue to consume rotated coordinates.
    """
    if draft_basis not in _DSPARK_QUAROT_DRAFT_BASES:
        raise ValueError(f"Invalid DSpark QuaRot draft basis: {draft_basis!r}.")
    if hc_head_basis not in _DSPARK_QUAROT_HC_HEAD_BASES:
        raise ValueError(f"Invalid DSpark QuaRot HC-head basis: {hc_head_basis!r}.")

    decoder_is_rotated = draft_basis in {"rotated", "rotated_decoder"}
    needs_boundary = rotation is not None and decoder_is_rotated and hc_head_basis == "canonical"
    hc_input = (
        _apply_dspark_quarot_rotation(hidden_states, rotation, transpose=True) if needs_boundary else hidden_states
    )
    head_hidden = _hc_head_torch(
        hc_input,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        norm_eps,
        hc_eps,
    )
    if not needs_boundary:
        return head_hidden
    return _apply_dspark_quarot_rotation(head_hidden, rotation, transpose=False)


def _dspark_mhc_pre_torch(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = residual.dtype
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    residual_hc = residual_flat.reshape(-1, hc_mult * hidden_size).float()
    mixes = F.linear(residual_hc, hc_fn.float())
    mixes = mixes * torch.rsqrt(residual_hc.square().mean(dim=-1, keepdim=True) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_start = hc_mult
    post_end = 2 * hc_mult
    post_logits = mixes[:, post_start:post_end] * hc_scale[1] + hc_base[post_start:post_end]
    post_mix = torch.sigmoid(post_logits) * hc_post_alpha

    comb_logits = mixes[:, post_end:].reshape(-1, hc_mult, hc_mult) * hc_scale[2] + hc_base[post_end:].reshape(
        1, hc_mult, hc_mult
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(max(int(sinkhorn_repeat) - 1, 0)):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.float(), dim=1).to(dtype)
    return (
        layer_input.reshape(*outer_shape, hidden_size),
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
    )


def _dspark_mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        res_mix.float(),
        residual.float(),
    )
    post_term = post_mix.float() * x.unsqueeze(-2).float()
    return (mixed_residual + post_term).to(residual.dtype)


def _dspark_mhc_pre_custom_op_supported(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
) -> bool:
    if residual.ndim != 3:
        return False
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mix_hc = (2 + hc_mult) * hc_mult
    return (
        residual.device.type == "npu"
        and residual.dtype == torch.bfloat16
        and hc_mult == 4
        and hidden_size in (4096, 7168)
        and hc_fn.device.type == "npu"
        and hc_scale.device.type == "npu"
        and hc_base.device.type == "npu"
        and hc_fn.dtype == torch.float32
        and hc_scale.dtype == torch.float32
        and hc_base.dtype == torch.float32
        and hc_fn.shape == (mix_hc, hc_mult * hidden_size)
        and hc_scale.shape == (3,)
        and hc_base.shape == (mix_hc,)
        and hc_pre_eps == hc_sinkhorn_eps
        and float(hc_post_alpha) == 2.0
    )


def _dspark_mhc_post_custom_op_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
) -> bool:
    if post_mix.ndim == 2:
        post_hc = post_mix.shape[-1]
        post_shape_ok = True
    elif post_mix.ndim == 3:
        post_hc = post_mix.shape[-2]
        post_shape_ok = post_mix.shape[-1] == 1
    else:
        return False
    return (
        x.device.type == "npu"
        and residual.device.type == "npu"
        and post_mix.device.type == "npu"
        and res_mix.device.type == "npu"
        and x.ndim == 2
        and residual.ndim == 3
        and post_shape_ok
        and res_mix.ndim == 3
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and post_mix.dtype == torch.float32
        and res_mix.dtype == torch.float32
        and x.shape[0] == residual.shape[0] == post_mix.shape[0] == res_mix.shape[0]
        and residual.shape[-2] == post_hc == res_mix.shape[-2] == res_mix.shape[-1] == 4
        and x.shape[-1] == residual.shape[-1]
        and x.shape[-1] in (4096, 7168)
    )


def _dspark_mhc_pre(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _dspark_mhc_pre_custom_op_supported(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
    ):
        layer_input, post_mix, res_mix = torch.ops._C_ascend.npu_hc_pre_v2(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            residual.shape[-2],
            sinkhorn_repeat,
            rms_eps,
            hc_pre_eps,
        )
        if post_mix.ndim == residual.ndim - 1:
            post_mix = post_mix.unsqueeze(-1)
        return layer_input, post_mix, res_mix
    return _dspark_mhc_pre_torch(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )


def _dspark_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
) -> torch.Tensor:
    if _dspark_mhc_post_custom_op_supported(x, residual, post_mix, res_mix):
        if post_mix.ndim == residual.ndim and post_mix.shape[-1] == 1:
            post_mix = post_mix.squeeze(-1)
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post_mix.unsqueeze(0),
            res_mix.unsqueeze(0),
        ).squeeze(0)
    return _dspark_mhc_post_torch(x, residual, post_mix, res_mix)


def _dspark_mhc_fused_post_pre_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_cur = _dspark_mhc_post_torch(x, residual, post_mix, res_mix)
    layer_input, post_mix_cur, res_mix_cur = _dspark_mhc_pre_torch(
        residual_cur,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    return residual_cur, post_mix_cur, res_mix_cur, layer_input


def _dspark_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_alpha: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _dspark_mhc_post_custom_op_supported(x, residual, post_mix, res_mix) and _dspark_mhc_pre_custom_op_supported(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
    ):
        residual_cur = _dspark_mhc_post(x, residual, post_mix, res_mix)
        layer_input, post_mix_cur, res_mix_cur = _dspark_mhc_pre(
            residual_cur,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_alpha,
            sinkhorn_repeat,
        )
        return residual_cur, post_mix_cur, res_mix_cur, layer_input
    return _dspark_mhc_fused_post_pre_torch(
        x,
        residual,
        post_mix,
        res_mix,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )


def _dspark_cache_capacity(vllm_config: VllmConfig, block_size: int, window_size: int | None = None) -> int:
    if window_size is not None:
        return max(block_size, int(window_size) + block_size)
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    return max(block_size, max_model_len + block_size)


def _dspark_max_request_slots(vllm_config: VllmConfig) -> int:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return max(1, int(getattr(scheduler_config, "max_num_seqs", 1) or 1))


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


def _dspark_standard_dsa_enabled() -> bool:
    return not envs.VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE


def _dspark_standard_dsa_sas_enabled() -> bool:
    return not envs.VLLM_ASCEND_DSPARK_USE_PTA_REF


def _dspark_private_context_cache_required(context_slot_mapping: torch.Tensor | None) -> bool:
    if not _dspark_standard_dsa_enabled():
        return True
    return context_slot_mapping is None


def _sync_npu_device_for_standard_pta(tensor: torch.Tensor) -> None:
    if tensor.device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _select_layer_value(
    value: typing.Any,
    layer_idx: int,
    layer_key: str,
    layer_prefix: str,
):
    if isinstance(value, dict):
        if layer_prefix in value:
            return value[layer_prefix]
        if layer_key in value:
            return value[layer_key]
        if layer_idx in value:
            return value[layer_idx]
        return None
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


def _get_layer_prefix(layer: nn.Module, layer_key: str) -> str:
    return getattr(
        getattr(getattr(getattr(layer, "self_attn", None), "dsa_attn", None), "swa_cache_layer", None),
        "prefix",
        layer_key,
    )


def _fp8_e4m3fn_quantized_abs(abs_scaled: torch.Tensor) -> torch.Tensor:
    subnormal = torch.floor(abs_scaled / _FP8_E4M3FN_SUBNORMAL_STEP + 0.5).clamp(0, 7) * _FP8_E4M3FN_SUBNORMAL_STEP

    normal_exp = torch.floor(torch.log2(abs_scaled.clamp_min(_FP8_E4M3FN_MIN_NORMAL))).clamp(-6, 8)
    normal_base = torch.exp2(normal_exp)
    mantissa = torch.floor((abs_scaled / normal_base - 1.0) * 8.0 + 0.5)
    carry = mantissa >= 8
    normal_exp = torch.where(carry, normal_exp + 1.0, normal_exp).clamp(-6, 8)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    mantissa = torch.where(
        normal_exp >= 8,
        mantissa.clamp(0, 6),
        mantissa.clamp(0, 7),
    )
    normal = (1.0 + mantissa / 8.0) * torch.exp2(normal_exp)

    return torch.where(
        abs_scaled < _FP8_E4M3FN_SUBNORMAL_NORMAL_MIDPOINT,
        subnormal,
        normal,
    )


def _fp8_e4m3fn_qdq(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.numel() == 0:
        return x

    orig_shape = x.shape
    last_dim = orig_shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(
            "DSpark FP8 QDQ requires the last dimension to be divisible by "
            f"the block size, but got last_dim={last_dim}, block_size={block_size}"
        )
    x_view = x.float().reshape(-1, last_dim)
    blocks = x_view.reshape(-1, last_dim // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(
        torch.full((), 2.0, dtype=torch.float32, device=x.device),
        torch.ceil(torch.log2(amax / 448.0)),
    )
    scaled = (blocks / scale).clamp(-448.0, 448.0)

    quantized_abs = _fp8_e4m3fn_quantized_abs(scaled.abs())
    qdq = torch.where(scaled < 0, -quantized_abs, quantized_abs) * scale
    return qdq.reshape(orig_shape).to(x.dtype)


def _fp8_qdq_nope_dims(
    kv: torch.Tensor,
    nope_head_dim: int,
    block_size: int = 64,
) -> torch.Tensor:
    if nope_head_dim <= 0:
        return kv
    kv_nope = _fp8_e4m3fn_qdq(kv[..., :nope_head_dim], block_size)
    return torch.cat([kv_nope, kv[..., nope_head_dim:]], dim=-1)


def _maybe_fp8_qdq_nope_dims(
    kv: torch.Tensor,
    nope_head_dim: int,
    apply_fp8_qdq: bool,
    block_size: int = 64,
) -> torch.Tensor:
    if not apply_fp8_qdq:
        return kv
    return _fp8_qdq_nope_dims(kv, nope_head_dim, block_size)


def _maybe_fp8_e4m3fn_qdq(
    x: torch.Tensor,
    apply_fp8_qdq: bool,
    block_size: int,
) -> torch.Tensor:
    if not apply_fp8_qdq:
        return x
    return _fp8_e4m3fn_qdq(x, block_size)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """DSpark sliding-window attention with an internal eager context cache."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        config = kwargs["config"]
        super().__init__(*args, **kwargs)
        self.compress_ratio = 1
        self.dsa_attn.compress_ratio = 1
        self.block_size = int(config.dspark_block_size)
        self._dspark_apply_fp8_qdq = _should_apply_dspark_fp8_qdq(config)
        cache_capacity = _dspark_cache_capacity(
            vllm_config,
            self.block_size,
            self.window_size if self.window_size is not None else None,
        )
        max_request_slots = _dspark_max_request_slots(vllm_config)
        cache_shape = (max_request_slots, cache_capacity, self.n_local_heads, self.head_dim)
        self.register_buffer(
            "_dspark_k_cache",
            torch.empty(
                cache_shape,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_v_cache",
            torch.empty(
                cache_shape,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_valid",
            torch.zeros((max_request_slots, cache_capacity), dtype=torch.bool, device=current_platform.device_type),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_positions",
            torch.full(
                (max_request_slots, cache_capacity),
                -1,
                dtype=torch.int32,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self._dspark_cache_capacity = cache_capacity
        self._dspark_max_request_slots = max_request_slots

    def _ensure_dspark_cache(self, length: int, like: torch.Tensor) -> None:
        del like
        if length > self._dspark_cache_capacity:
            raise ValueError(
                "DSpark attention cache position exceeds preallocated capacity: "
                f"length={length}, capacity={self._dspark_cache_capacity}"
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        if request_slots is None or request_slots.numel() == 0:
            return
        slots = torch.unique(request_slots.to(torch.long))
        if slots.numel() == 0:
            return
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        self._dspark_cache_valid[slots] = False
        self._dspark_cache_positions[slots] = -1

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._expand_private_kv(self._project_shared_kv(hidden_states, positions))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()

    def _expand_private_kv(self, shared_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = shared_kv.squeeze(1)
        kv = _maybe_fp8_qdq_nope_dims(kv, self.nope_head_dim, self._dspark_apply_fp8_qdq)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k = torch.cat(
            [
                k_nope.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                k_pe.unsqueeze(1).expand(-1, self.n_local_heads, -1),
            ],
            dim=-1,
        ).contiguous()
        v = kv.unsqueeze(1).expand(-1, self.n_local_heads, -1).contiguous()
        return k, v

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if not _dspark_standard_dsa_enabled():
            return
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        from vllm_ascend.device.device_op import DeviceOperator

        slot_mapping = slot_mapping.to(device=shared_kv.device, dtype=torch.int32)
        forward_context = _maybe_get_forward_context()
        num_actual_tokens = getattr(forward_context, "num_actual_tokens", None)
        if num_actual_tokens is not None and num_actual_tokens < slot_mapping.shape[0]:
            shared_kv = shared_kv[:num_actual_tokens]
            slot_mapping = slot_mapping[:num_actual_tokens]
        capture_mode = (
            getattr(forward_context, "cudagraph_runtime_mode", CUDAGraphMode.NONE) == CUDAGraphMode.FULL
            or torch.compiler.is_compiling()
        )
        if capture_mode:
            if slot_mapping.ndim == 1:
                slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_cache_layer.block_size)
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)
            return

        valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)
        if not bool(torch.any(valid).item()):
            return
        if not bool(torch.all(valid).item()):
            shared_kv = shared_kv[valid]
            slot_mapping = slot_mapping[valid]
        if slot_mapping.ndim == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_cache_layer.block_size)
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)
        # The PTA reference reads the raw SWA cache immediately after scatter,
        # outside the normal DSA attention op stream choreography.
        _sync_npu_device_for_standard_pta(shared_kv)

    _store_standard_swa_context_kv = _store_standard_swa_kv

    def _standard_query_slot_mapping_from_block_table(
        self,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not _dspark_standard_dsa_enabled() or block_table is None:
            return slot_mapping

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        cache_block_size = int(swa_cache_layer.block_size)
        out = torch.full_like(positions, -1, dtype=torch.int32)
        valid = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        if slot_mapping is not None:
            slot_mapping = slot_mapping.to(device=positions.device)
            valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)

        pos_long = positions.to(torch.long)
        if token_to_req_indices is not None:
            if token_to_req_indices.numel() < positions.numel():
                raise ValueError(
                    "DSpark token_to_req_indices must cover query tokens: "
                    f"token_to_req_indices={token_to_req_indices.numel()}, positions={positions.numel()}"
                )
            token_to_req = token_to_req_indices[: positions.numel()].to(
                device=positions.device,
                dtype=torch.long,
            )
            valid_req = (token_to_req >= 0) & (token_to_req < block_table.shape[0])
            req_indices = token_to_req.clamp(0, block_table.shape[0] - 1)
            block_nums = pos_long // cache_block_size
            block_offsets = pos_long % cache_block_size
            block_nums = block_nums.clamp(0, block_table.shape[1] - 1)
            flat_block_table = block_table.to(device=positions.device, dtype=torch.long).reshape(-1)
            flat_indices = req_indices * block_table.shape[1] + block_nums
            block_ids = flat_block_table.index_select(0, flat_indices)
            out = (block_ids * cache_block_size + block_offsets).to(torch.int32)
            valid &= valid_req
        else:
            for block_offset in range(0, positions.numel(), self.block_size):
                block_end = min(block_offset + self.block_size, positions.numel())
                req_idx = block_offset // self.block_size
                if req_idx >= block_table.shape[0]:
                    continue
                block_pos = pos_long[block_offset:block_end]
                block_nums = block_pos // cache_block_size
                block_offsets = block_pos % cache_block_size
                block_ids = (
                    block_table[req_idx].to(device=positions.device, dtype=torch.long).index_select(0, block_nums)
                )
                out[block_offset:block_end] = (block_ids * cache_block_size + block_offsets).to(torch.int32)
        out.masked_fill_(~valid, -1)
        return out

    def _run_standard_dspark_attention(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        draft_kv: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not _dspark_standard_dsa_enabled():
            return None
        forward_context = _maybe_get_forward_context()
        capture_mode = (
            getattr(forward_context, "cudagraph_runtime_mode", CUDAGraphMode.NONE) == CUDAGraphMode.FULL
            or torch.compiler.is_compiling()
        )
        if block_table is None and capture_mode:
            block_table = self._standard_dspark_block_table(forward_context)
        if block_table is None:
            if capture_mode:
                raise RuntimeError("DSpark standard-cache attention requires block_table during graph capture")
            if not capture_mode:
                logger.warning_once(
                    "DSpark standard SWA cache PTA path has no block_table; falling back to private cache"
                )
            return None

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            if capture_mode:
                raise RuntimeError("DSpark standard-cache attention requires SWA kv_cache during graph capture")
            if not capture_mode:
                logger.warning_once("DSpark standard SWA cache PTA path has no kv_cache; falling back to private cache")
            return None

        dspark_swa_indices = dspark_swa_lens = sas_metadata = None
        if capture_mode:
            dspark_swa_indices, dspark_swa_lens, sas_metadata = DeepseekV4DSparkAttention._standard_dspark_swa_metadata(
                self,
                forward_context,
            )
            # Under graph capture all three (indices, lens, sas_metadata) must be
            # prebuilt and injected via forward_context.draft_attn_metadatas. A
            # missing sas_metadata would flip skip_scheduling_guard off below and
            # pull the host-sync scheduling guard (.item()/.to('cpu')/nonzero) into
            # the captured region, silently hanging or corrupting replay. Fail loud.
            if dspark_swa_indices is None or dspark_swa_lens is None or sas_metadata is None:
                raise RuntimeError(
                    "DSpark standard-cache attention requires prebuilt SWA indices, "
                    "lens, and SAS metadata during graph capture "
                    f"(indices={dspark_swa_indices is not None}, "
                    f"lens={dspark_swa_lens is not None}, "
                    f"sas_metadata={sas_metadata is not None})"
                )
        if _dspark_standard_dsa_sas_enabled():
            sas_out = dspark_attention_from_standard_cache_sas(
                q,
                swa_kv_cache,
                block_table,
                positions,
                slot_mapping,
                self.attn_sink[: self.n_local_heads],
                self.block_size,
                int(self.window_size),
                int(swa_cache_layer.block_size),
                float(self.scale),
                query_start_loc=dspark_query_start_loc,
                seq_lens=dspark_seq_lens,
                token_to_req_indices=dspark_token_to_req_indices,
                dspark_swa_indices=dspark_swa_indices,
                dspark_swa_lens=dspark_swa_lens,
                sas_metadata=sas_metadata,
                num_query_tokens=getattr(forward_context, "num_actual_tokens", None),
                skip_scheduling_guard=capture_mode and sas_metadata is not None,
                raise_on_error=capture_mode,
            )
            if sas_out is not None:
                return sas_out
            if capture_mode:
                raise RuntimeError("DSpark standard-cache SAS attention did not produce output during graph capture")

        return dspark_attention_from_standard_cache(
            q,
            swa_kv_cache,
            block_table,
            positions,
            slot_mapping,
            draft_kv,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            int(swa_cache_layer.block_size),
            float(self.scale),
            request_slots=request_slots,
            cache_positions=getattr(self, "_dspark_cache_positions", None),
            cache_valid=getattr(self, "_dspark_cache_valid", None),
            dspark_swa_indices=dspark_swa_indices,
            dspark_swa_lens=dspark_swa_lens,
            query_start_loc=dspark_query_start_loc,
            seq_lens=dspark_seq_lens,
            token_to_req_indices=dspark_token_to_req_indices,
        )

    def _standard_dspark_layer_metadata(self, forward_context: typing.Any) -> typing.Any | None:
        draft_attn_metadatas = getattr(forward_context, "draft_attn_metadatas", None)
        if not draft_attn_metadatas:
            return None

        swa_cache_layer = getattr(self.dsa_attn, "swa_cache_layer", None)
        layer_names = (
            getattr(swa_cache_layer, "prefix", None),
            getattr(self.dsa_attn, "prefix", None),
        )
        for metadata_map in draft_attn_metadatas:
            if not isinstance(metadata_map, dict):
                continue
            metadata = None
            for layer_name in layer_names:
                if layer_name is not None and layer_name in metadata_map:
                    metadata = metadata_map[layer_name]
                    break
            if metadata is None:
                continue
            return metadata
        return None

    def _standard_dspark_block_table(self, forward_context: typing.Any) -> torch.Tensor | None:
        metadata = self._standard_dspark_layer_metadata(forward_context)
        if metadata is None:
            return None
        for sub_metadata in (
            metadata,
            getattr(metadata, "decode", None),
            getattr(metadata, "prefill", None),
        ):
            if sub_metadata is None:
                continue
            block_table = getattr(sub_metadata, "block_table", None)
            if block_table is None:
                block_table = getattr(sub_metadata, "block_tables", None)
            if isinstance(block_table, torch.Tensor):
                return block_table
        return None

    def _standard_dspark_swa_metadata(
        self,
        forward_context: typing.Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        metadata = self._standard_dspark_layer_metadata(forward_context)
        if metadata is None:
            return None, None, None
        for sub_metadata in (
            getattr(metadata, "decode", None),
            getattr(metadata, "prefill", None),
            metadata,
        ):
            if sub_metadata is None:
                continue
            dspark_swa_indices = getattr(sub_metadata, "dspark_swa_indices", None)
            dspark_swa_lens = getattr(sub_metadata, "dspark_swa_lens", None)
            if dspark_swa_indices is not None and dspark_swa_lens is not None:
                return dspark_swa_indices, dspark_swa_lens, getattr(sub_metadata, "sas_metadata", None)
        return None, None, None

    def _run_dspark_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty_like(q)
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        if request_slots.numel() != positions.numel():
            raise ValueError(
                "DSpark request_slots length must match query positions: "
                f"request_slots={request_slots.numel()}, positions={positions.numel()}"
            )

        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        return dspark_attention(
            q,
            self._dspark_k_cache,
            self._dspark_v_cache,
            self._dspark_cache_positions,
            self._dspark_cache_valid,
            k,
            v,
            request_slots,
            positions,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            float(self.scale),
            shared_kv=True,
        )

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor | None = None,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        if positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)
        self._store_standard_swa_kv(shared_kv, context_slot_mapping)
        if not _dspark_private_context_cache_required(context_slot_mapping):
            return

        k, v = self._expand_private_kv(shared_kv)
        forward_context = _maybe_get_forward_context()
        capture_mode = (
            getattr(forward_context, "cudagraph_runtime_mode", CUDAGraphMode.NONE) == CUDAGraphMode.FULL
            or torch.compiler.is_compiling()
        )
        if not capture_mode:
            max_pos = int(positions.max().item())
            self._ensure_dspark_cache(min(max_pos + 1, self._dspark_cache_capacity), k)
        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        slots_long = request_slots.to(torch.long)
        if slots_long.numel() != positions.numel():
            raise ValueError(
                "DSpark request_slots length must match context positions: "
                f"request_slots={slots_long.numel()}, positions={positions.numel()}"
            )
        if not capture_mode and int(slots_long.max().item()) >= self._dspark_max_request_slots:
            raise ValueError(
                "DSpark request slot exceeds preallocated cache slots: "
                f"slot={int(slots_long.max().item())}, capacity={self._dspark_max_request_slots}"
            )
        pos_long = positions.to(torch.long)
        cache_indices = pos_long % self._dspark_cache_capacity
        self._dspark_k_cache[slots_long, cache_indices] = k
        self._dspark_v_cache[slots_long, cache_indices] = v
        self._dspark_cache_positions[slots_long, cache_indices] = positions.to(torch.int32)
        self._dspark_cache_valid[slots_long, cache_indices] = True

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        qr = self.q_norm(_linear_output(self.wq_a, hidden_states))
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))

        q = _linear_output(self.wq_b, qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        shared_kv = torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()
        q = torch.cat([q_nope, q_pe], dim=-1)
        standard_slot_mapping = self._standard_query_slot_mapping_from_block_table(
            positions,
            slot_mapping,
            block_table,
            dspark_token_to_req_indices,
        )
        self._store_standard_swa_kv(shared_kv, standard_slot_mapping)
        standard_attn_out = self._run_standard_dspark_attention(
            q,
            positions,
            standard_slot_mapping,
            block_table,
            shared_kv,
            request_slots,
            dspark_query_start_loc,
            dspark_seq_lens,
            dspark_token_to_req_indices,
        )
        private_attn_out = None
        if standard_attn_out is None:
            private_kv = _maybe_fp8_qdq_nope_dims(
                shared_kv.squeeze(1),
                self.nope_head_dim,
                self._dspark_apply_fp8_qdq,
            )
            private_k_nope, private_k_pe = private_kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
            private_k = torch.cat(
                [
                    private_k_nope.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                    private_k_pe.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                ],
                dim=-1,
            ).contiguous()
            private_v = private_kv.unsqueeze(1).expand(-1, self.n_local_heads, -1).contiguous()
            private_attn_out = self._run_dspark_attention(q, private_k, private_v, positions, request_slots)
        attn_out = standard_attn_out if standard_attn_out is not None else private_attn_out
        assert attn_out is not None

        attn_out = _apply_dsv4_rope_tail(
            self.rotary_emb,
            positions,
            attn_out,
            inverse=True,
        )
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        attn_out = attn_out.reshape(-1, self.n_local_groups, group_dim)
        attn_out = _maybe_fp8_e4m3fn_qdq(attn_out, self._dspark_apply_fp8_qdq, 128)
        wo_a = _wo_a_weight_for_eager_projection(
            self.wo_a.weight,
            self.n_local_groups,
            self.o_lora_rank,
            group_dim,
        )
        z = _grouped_wo_a_projection(attn_out, wo_a).flatten(1)
        return _linear_output(self.wo_b, z)


class DeepseekV4DSparkDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            topk_indices_buffer=None,
            is_draft_layer=True,
            attn_cls=DeepseekV4DSparkAttention,
            quant_config_override=_draft_quant_config(vllm_config),
            use_quant_config_override=True,
        )
        self.hc_post_alpha = 2.0

    def _mhc_pre(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _dspark_mhc_pre(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            self.norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def _mhc_post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return _dspark_mhc_post(hidden_states, residual, post_mix, res_mix)

    def _mhc_fused_post_pre(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _dspark_mhc_fused_post_pre(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            hc_fn,
            hc_scale,
            hc_base,
            self.norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del llama_4_scaling
        if residual is None:
            residual = hidden_states
            hidden_states, post_mix, res_mix = self._mhc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )
        else:
            assert post_mix is not None and res_mix is not None
            residual, post_mix, res_mix, hidden_states = self._mhc_fused_post_pre(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )
        hidden_states = self.input_layernorm(hidden_states)
        attn_kwargs = {
            "request_slots": request_slots,
            "slot_mapping": slot_mapping,
            "block_table": block_table,
        }
        if dspark_query_start_loc is not None or dspark_seq_lens is not None or dspark_token_to_req_indices is not None:
            attn_kwargs.update(
                dspark_query_start_loc=dspark_query_start_loc,
                dspark_seq_lens=dspark_seq_lens,
                dspark_token_to_req_indices=dspark_token_to_req_indices,
            )
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            None,
            **attn_kwargs,
        )

        residual, post_mix, res_mix, hidden_states = self._mhc_fused_post_pre(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids)
        return hidden_states, residual, post_mix, res_mix


class DSparkMarkovHead(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.markov_w2, markov_embed)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        return self.bias(embed), embed


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DSparkDecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx + idx}"),
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=_draft_main_proj_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(
            config,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        device_config = getattr(vllm_config, "device_config", None)
        rotation_device = getattr(device_config, "device", current_platform.device_type)
        self._dspark_quarot_draft_basis = _get_dspark_quarot_draft_basis(vllm_config)
        self._dspark_quarot_hc_head_basis = _get_dspark_quarot_hc_head_basis(
            vllm_config,
            draft_basis=self._dspark_quarot_draft_basis,
        )
        rotation = _load_dspark_quarot_rotation(vllm_config, device=rotation_device)
        self.register_buffer("_dspark_quarot_rotation", rotation, persistent=False)
        self.register_buffer("_dspark_quarot_rotated_hc_head_fn", None, persistent=False)
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

    def install_quarot_hc_head_fold(self) -> None:
        """Install the effective rotated HC head after checkpoint loading.

        Reloads always derive from ``self.hc_head_fn`` (the canonical source
        Parameter), never from the existing buffer, so this operation is
        idempotent and cannot double-fold the weight. Compatible reloads copy
        into the existing buffer to preserve captured ACLGraph addresses. A
        changed tensor contract replaces the buffer and therefore requires
        graph capture to happen again.
        """
        should_fold = (
            self._dspark_quarot_rotation is not None
            and self._dspark_quarot_draft_basis in {"rotated", "rotated_decoder"}
            and self._dspark_quarot_hc_head_basis == "canonical"
        )
        if not should_fold:
            self._dspark_quarot_rotated_hc_head_fn = None
            return
        with torch.no_grad():
            derived = _derive_dspark_rotated_hc_head_fn(
                self.hc_head_fn.detach(),
                self._dspark_quarot_rotation.detach(),
            )
            existing = self._dspark_quarot_rotated_hc_head_fn
            if (
                existing is not None
                and existing.shape == derived.shape
                and existing.device == derived.device
                and existing.dtype == derived.dtype
            ):
                # ACLGraph captures the buffer address. Preserve it across
                # compatible checkpoint reloads so replay observes the newly
                # folded HC head instead of a stale or recycled allocation.
                existing.copy_(derived)
            else:
                # The first install happens before graph capture. A changed
                # tensor contract also requires callers to recapture graphs.
                self._dspark_quarot_rotated_hc_head_fn = derived

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        return _transition_dspark_quarot_basis(
            inputs_embeds,
            self._dspark_quarot_rotation,
            draft_basis=self._dspark_quarot_draft_basis,
            direction="target_to_draft",
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor
        | list[torch.Tensor | None]
        | tuple[torch.Tensor | None, ...]
        | dict[str, torch.Tensor | None]
        | dict[int, torch.Tensor | None]
        | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        rotation = getattr(self, "_dspark_quarot_rotation", None)
        draft_basis = getattr(self, "_dspark_quarot_draft_basis", "legacy")
        main_proj_input = _prepare_dspark_main_proj_input(
            context_states,
            rotation,
            draft_basis=draft_basis,
            hidden_size=getattr(self, "hidden_size", context_states.shape[-1]),
        )
        main_x = self.main_norm(_linear_output(self.main_proj, main_proj_input))
        main_x = _prepare_dspark_main_proj_output(
            main_x,
            rotation,
            draft_basis=draft_basis,
        )
        for layer_idx, (layer_key, layer) in enumerate(self.layers.items()):
            layer_prefix = _get_layer_prefix(layer, layer_key)
            layer_context_slot_mapping = _select_layer_value(
                context_slot_mapping,
                layer_idx,
                layer_key,
                layer_prefix,
            )
            layer.self_attn.precompute_context_kv(
                main_x,
                context_positions,
                request_slots=context_request_slots,
                context_slot_mapping=layer_context_slot_mapping,
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        for layer in self.layers.values():
            layer.self_attn.reset_request_slots(request_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | dict[int, torch.Tensor]
        | None = None,
        block_table: torch.Tensor
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | dict[int, torch.Tensor]
        | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for layer_idx, (layer_key, layer) in enumerate(self.layers.items()):
            layer_prefix = _get_layer_prefix(layer, layer_key)
            layer_kwargs = {
                "positions": positions,
                "hidden_states": hidden_states,
                "residual": residual,
                "post_mix": post_mix,
                "res_mix": res_mix,
                "input_ids": input_ids,
                "request_slots": request_slots,
                "slot_mapping": _select_layer_value(slot_mapping, layer_idx, layer_key, layer_prefix),
                "block_table": _select_layer_value(block_table, layer_idx, layer_key, layer_prefix),
            }
            if (
                dspark_query_start_loc is not None
                or dspark_seq_lens is not None
                or dspark_token_to_req_indices is not None
            ):
                layer_kwargs.update(
                    dspark_query_start_loc=dspark_query_start_loc,
                    dspark_seq_lens=dspark_seq_lens,
                    dspark_token_to_req_indices=dspark_token_to_req_indices,
                )
            layer_output = layer(**layer_kwargs)
            if isinstance(layer_output, tuple) and len(layer_output) == 4:
                hidden_states, residual, post_mix, res_mix = layer_output
            else:
                hidden_states = layer_output
        return self.compute_head_hidden(hidden_states, residual, post_mix, res_mix)

    def compute_head_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None and post_mix is not None and res_mix is not None:
            hidden_states = _dspark_mhc_post(hidden_states, residual, post_mix, res_mix)
        if hidden_states.dim() == 2:
            return hidden_states
        effective_hc_head_fn = self._dspark_quarot_rotated_hc_head_fn
        effective_hc_head_basis = self._dspark_quarot_hc_head_basis
        if effective_hc_head_fn is None:
            effective_hc_head_fn = self.hc_head_fn
        else:
            effective_hc_head_basis = "rotated"
        return _compute_dspark_hc_head(
            hidden_states,
            effective_hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
            self._dspark_quarot_rotation,
            draft_basis=self._dspark_quarot_draft_basis,
            hc_head_basis=effective_hc_head_basis,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        head_hidden = self.compute_head_hidden(hidden_states)
        head_hidden = self.norm(head_hidden)
        # QuaRot fuses the final RMSNorm scale into the shared target head.
        # Removing gamma is independent of whether the draft residual stream
        # itself is canonical or already rotated.
        if self._dspark_quarot_rotation is not None:
            head_hidden = head_hidden / self.norm.weight.to(device=head_hidden.device, dtype=head_hidden.dtype)
        head_hidden = _transition_dspark_quarot_basis(
            head_hidden,
            self._dspark_quarot_rotation,
            draft_basis=self._dspark_quarot_draft_basis,
            direction="draft_to_target",
        )
        return logits_processor(lm_head, head_hidden)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _make_deepseek_v4_expert_params_mapping(
            self,
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, DeepseekV2MixtureOfExperts):
    # DSpark draft embed/head are aliases of the target model, matching
    # upstream vLLM's DSparkDeepseekV4ForCausalLM contract.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = draft_model_config.hf_config
        self.quant_config = _draft_quant_config(vllm_config)
        self._dspark_checkpoint_path = str(getattr(draft_model_config, "model", "") or "")
        parallel_config = vllm_config.parallel_config
        self._dspark_expert_parallel_enabled = bool(getattr(parallel_config, "enable_expert_parallel", False))
        self._dspark_ep_weight_filter_enabled = bool(
            self._dspark_expert_parallel_enabled
            and getattr(parallel_config, "enable_ep_weight_filter", False)
            and not getattr(parallel_config, "enable_eplb", False)
        )
        _validate_dspark_quant_description(vllm_config)
        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.set_moe_parameters_from_layers(self.config, self.model.layers.values())

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        block_table: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        dspark_query_start_loc: torch.Tensor | None = None,
        dspark_seq_lens: torch.Tensor | None = None,
        dspark_token_to_req_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors, spec_step_idx
        assert input_ids is not None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            request_slots=request_slots,
            slot_mapping=slot_mapping,
            block_table=block_table,
            dspark_query_start_loc=dspark_query_start_loc,
            dspark_seq_lens=dspark_seq_lens,
            dspark_token_to_req_indices=dspark_token_to_req_indices,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor
        | list[torch.Tensor | None]
        | tuple[torch.Tensor | None, ...]
        | dict[str, torch.Tensor | None]
        | dict[int, torch.Tensor | None]
        | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
            context_request_slots,
        )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        self.model.reset_request_slots(request_slots)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        missing_mtp_params: set[str] = set()
        start_layer_idx = self.config.num_hidden_layers
        num_dspark_layers = self.model.num_dspark_layers
        last_layer_idx = start_layer_idx + num_dspark_layers - 1
        quant_description = getattr(getattr(self, "quant_config", None), "quant_description", None)
        source_quant_description = getattr(
            getattr(self, "quant_config", None),
            "source_quant_description",
            quant_description,
        )
        expected_checkpoint_tensors = (
            _required_dspark_checkpoint_tensors(
                source_quant_description,
                start_layer_idx=start_layer_idx,
                num_dspark_layers=num_dspark_layers,
            )
            if isinstance(source_quant_description, dict)
            else set()
        )
        required_checkpoint_tensor_groups = (
            _required_dspark_checkpoint_tensor_groups(
                source_quant_description,
                start_layer_idx=start_layer_idx,
                num_dspark_layers=num_dspark_layers,
            )
            if isinstance(source_quant_description, dict)
            else []
        )
        expected_weight_dtype_kinds = (
            _dspark_checkpoint_weight_dtype_kinds(
                source_quant_description,
                start_layer_idx=start_layer_idx,
                num_dspark_layers=num_dspark_layers,
            )
            if isinstance(source_quant_description, dict)
            else {}
        )
        expected_weight_shapes = _dspark_checkpoint_weight_shapes(self.config)
        mtp_dequantized_to_bf16 = bool(getattr(self.config, "dspark_mtp_dequantized_to_bf16", False))
        checkpoint_index_validated = _validate_dspark_checkpoint_index(
            getattr(self, "_dspark_checkpoint_path", None),
            expected_checkpoint_tensors,
            required_checkpoint_tensor_groups,
            expected_weight_dtype_kinds=expected_weight_dtype_kinds,
            expected_weight_shapes=expected_weight_shapes,
            require_mtp_float_weights=mtp_dequantized_to_bf16,
        )
        if checkpoint_index_validated:
            logger.info_once("DSpark checkpoint index and safetensors headers passed physical tensor preflight.")
        seen_checkpoint_tensors: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        expert_parallel_enabled = bool(getattr(self, "_dspark_expert_parallel_enabled", False))
        rank_local_expert_ids_by_stage = (
            _dspark_local_expert_ids_by_stage(self.model) if expert_parallel_enabled else None
        )
        filtered_local_expert_ids_by_stage = (
            rank_local_expert_ids_by_stage if getattr(self, "_dspark_ep_weight_filter_enabled", False) else None
        )
        unknown_expert_locality_params: set[str] = set()
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )
        for name, loaded_weight in weights:
            if _DSPARK_RAW_CHECKPOINT_TENSOR_RE.match(name) is not None:
                seen_checkpoint_tensors.add(name)
                expected_kind = expected_weight_dtype_kinds.get(name)
                if mtp_dequantized_to_bf16 and name.endswith(".weight"):
                    expected_kind = "floating"
                if expected_kind is not None:
                    _validate_dspark_loaded_weight_dtype(name, loaded_weight, expected_kind)
                expected_shape = expected_weight_shapes.get(name)
                if expected_shape is not None:
                    _validate_dspark_loaded_weight_shape(name, loaded_weight, expected_shape)
            if name == "embed.weight":
                embed_name = "model.embed_tokens.weight"
                param = params_dict[embed_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(embed_name)
                continue
            if name == "head.weight":
                head_name = "lm_head.weight"
                param = params_dict[head_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(head_name)
                continue
            mapped_name = self._remap_dspark_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                canonical_name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
                if canonical_name in params_dict:
                    name = canonical_name
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
                name = name.removesuffix(".scale") + suffix
                if name not in params_dict and ".experts." not in name:
                    missing_mtp_params.add(name)
                    continue
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if ".experts." in name or f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    missing_mtp_params.add(mapped)
                    break
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(mapped)
                break
            else:
                if ".experts." in name:
                    matched_expert_mapping = False
                    loaded_expert_mapping = False
                    failed_local_mappings: set[str] = set()
                    failed_unknown_mappings: set[str] = set()
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                        if weight_name not in name:
                            continue
                        matched_expert_mapping = True
                        mapped = name.replace(weight_name, param_name)
                        match = _LAYER_ID_RE.search(mapped)
                        stage_idx = int(match.group(1)) - start_layer_idx if match is not None else None
                        local_ids = (
                            rank_local_expert_ids_by_stage.get(stage_idx)
                            if rank_local_expert_ids_by_stage is not None and stage_idx is not None
                            else None
                        )
                        if not expert_parallel_enabled:
                            is_local_expert: bool | None = True
                        elif local_ids is None:
                            is_local_expert = None
                        else:
                            is_local_expert = int(expert_id) in local_ids
                        if mapped not in params_dict:
                            if is_local_expert is True:
                                failed_local_mappings.add(mapped)
                            elif is_local_expert is None:
                                failed_unknown_mappings.add(mapped)
                            continue
                        param = params_dict[mapped]
                        weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(mapped)
                            loaded_expert_mapping = True
                            break
                        if is_local_expert is True:
                            failed_local_mappings.add(mapped)
                        elif is_local_expert is None:
                            failed_unknown_mappings.add(mapped)
                    if not matched_expert_mapping:
                        missing_mtp_params.add(name)
                    elif not loaded_expert_mapping:
                        missing_mtp_params.update(failed_local_mappings)
                        unknown_expert_locality_params.update(failed_unknown_mappings)
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        missing_mtp_params.add(name)
                        continue
                    narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name not in params_dict:
                    missing_mtp_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        missing_checkpoint_tensors = _missing_dspark_checkpoint_tensors(
            expected_checkpoint_tensors,
            seen_checkpoint_tensors,
            local_expert_ids_by_stage=filtered_local_expert_ids_by_stage,
        )
        missing_checkpoint_tensor_groups = [
            alternatives
            for alternatives in required_checkpoint_tensor_groups
            if not any(name in seen_checkpoint_tensors for name in alternatives)
        ]
        if missing_checkpoint_tensor_groups:
            preview = [" or ".join(group) for group in missing_checkpoint_tensor_groups[:20]]
            raise ValueError(
                "DSpark quantized checkpoint weight stream is missing required physical companion tensors: "
                f"{preview} (missing {len(missing_checkpoint_tensor_groups)} groups total)."
            )
        if missing_checkpoint_tensors:
            preview = missing_checkpoint_tensors[:20]
            suffix = "" if len(missing_checkpoint_tensors) <= len(preview) else " ..."
            raise ValueError(
                "DSpark ModelSlim source description references checkpoint tensors that were not supplied: "
                f"{preview}{suffix} (missing {len(missing_checkpoint_tensors)} total)."
            )

        if unknown_expert_locality_params:
            raise ValueError(
                "DSpark EP expert tensor mappings all returned False, but this rank's local expert assignment "
                f"could not be determined: {sorted(unknown_expert_locality_params)}."
            )
        if missing_mtp_params:
            raise ValueError(
                "DSpark speculative decoding checkpoint weights did not match model parameters: "
                f"{sorted(missing_mtp_params)}"
            )

        loaded_layer_ids: set[int] = set()
        for param_name in loaded_params:
            match = _LAYER_ID_RE.search(param_name)
            if match:
                loaded_layer_ids.add(int(match.group(1)))
        for layer_idx in range(start_layer_idx, start_layer_idx + self.model.num_dspark_layers):
            if layer_idx not in loaded_layer_ids:
                raise ValueError(f"DSpark speculative decoding layer {layer_idx} weights missing from checkpoint.")
        required_params = {
            f"model.layers.{start_layer_idx}.main_proj.weight",
            f"model.layers.{start_layer_idx}.main_norm.weight",
            f"model.layers.{last_layer_idx}.norm.weight",
            "model.hc_head_fn",
            "model.hc_head_base",
            "model.hc_head_scale",
            f"model.layers.{last_layer_idx}.markov_head.markov_w1.weight",
            f"model.layers.{last_layer_idx}.markov_head.markov_w2.weight",
        }
        main_proj_prefix = f"model.layers.{start_layer_idx}.main_proj."
        required_params.update(name for name in params_dict if name.startswith(main_proj_prefix))
        quant_param_markers = ("weight_scale", "weight_offset", "scale_bias")
        for param_name in params_dict:
            match = _LAYER_ID_RE.search(param_name)
            belongs_to_draft_layer = (
                match is not None
                and start_layer_idx <= int(match.group(1)) <= last_layer_idx
                and ".compressor." not in param_name
                and ".indexer." not in param_name
            )
            if belongs_to_draft_layer and any(marker in param_name for marker in quant_param_markers):
                required_params.add(param_name)
        missing_required = sorted(required_params - loaded_params)
        if missing_required:
            raise ValueError(
                f"DSpark speculative decoding required weights missing from checkpoint load: {missing_required}"
            )
        self.model.finalize_mega_moe_weights()
        self.model.install_quarot_hc_head_fold()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        match = re.match(r"mtp\.(\d+)\.(.*)", name)
        if match is None:
            return None
        stage_idx = int(match.group(1))
        layer_idx = self.config.num_hidden_layers + stage_idx
        rest = match.group(2)
        if rest.startswith("confidence_head."):
            return None
        name = f"model.layers.{layer_idx}.{rest}"
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        name = name.replace(".mlp.gate.bias", ".mlp.gate.e_score_correction_bias")
        return name


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
