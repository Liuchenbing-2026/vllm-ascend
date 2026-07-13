# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import PretrainedConfig

import vllm_ascend.models.deepseek_v4_dspark as dspark_model_module
from vllm_ascend.models.deepseek_v4 import _hc_head_torch, _is_dspark_target_layer
from vllm_ascend.models.deepseek_v4_dspark import (
    DeepseekV4DSparkAttention,
    DeepseekV4DSparkDecoderLayer,
    DeepseekV4DSparkModel,
    DeepSeekV4DSparkMTP,
    _apply_dspark_quarot_rotation,
    _compute_dspark_hc_head,
    _derive_dspark_rotated_hc_head_fn,
    _draft_main_proj_quant_config,
    _draft_quant_config,
    _dspark_checkpoint_name_for_quant_key,
    _dspark_checkpoint_weight_shapes,
    _get_dspark_num_mtp_layers,
    _get_dspark_quarot_draft_basis,
    _get_dspark_quarot_hc_head_basis,
    _load_dspark_quarot_rotation,
    _maybe_fp8_e4m3fn_qdq,
    _maybe_fp8_qdq_nope_dims,
    _missing_dspark_checkpoint_tensors,
    _prepare_dspark_main_proj_input,
    _prepare_dspark_main_proj_output,
    _required_dspark_checkpoint_tensor_groups,
    _required_dspark_checkpoint_tensors,
    _should_apply_dspark_fp8_qdq,
    _transition_dspark_quarot_basis,
    _validate_dspark_checkpoint_index,
    _validate_dspark_loaded_weight_dtype,
    _validate_dspark_loaded_weight_shape,
    _validate_dspark_quant_description,
)
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def test_dspark_deepseek_v4_hf_config_override():
    repo_root = Path(__file__).parents[3]
    patch_module = runpy.run_path(str(repo_root / "vllm_ascend/patch/platform/patch_speculative_config.py"))

    hf_config = PretrainedConfig(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
    )

    patched = patch_module["hf_config_override"](hf_config)

    assert patched.model_type == "deepseek_mtp"
    assert patched.architectures == ["DeepSeekV4DSparkMTPModel"]
    assert patched.n_predict == 5
    assert patched.ptd_token_id == 128799


def test_dspark_quarot_rotation_loads_optional_modelslim_metadata(tmp_path):
    rotation_path = tmp_path / "quarot" / "rotation.safetensors"
    rotation_path.parent.mkdir()
    expected = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    save_file({"global_rotation": expected}, rotation_path)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=SimpleNamespace(
            quant_description={
                "optional": {
                    "quarot": {
                        "rotation_map": {
                            "global_rotation": "quarot/rotation.safetensors",
                        }
                    }
                }
            }
        ),
    )

    rotation = _load_dspark_quarot_rotation(vllm_config)

    torch.testing.assert_close(rotation, expected)


@pytest.mark.parametrize(
    "rotation",
    [
        torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float32),
        torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        / (2.0**0.5),
        torch.eye(2, dtype=torch.float32) * 1.1,
    ],
)
def test_dspark_quarot_rotation_rejects_nonorthogonal_matrices(tmp_path, rotation):
    rotation_path = tmp_path / "quarot" / "rotation.safetensors"
    rotation_path.parent.mkdir()
    save_file({"global_rotation": rotation}, rotation_path)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=SimpleNamespace(
            quant_description={
                "optional": {
                    "quarot": {
                        "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
                    }
                }
            }
        ),
    )

    with pytest.raises(ValueError, match="not orthonormal"):
        _load_dspark_quarot_rotation(vllm_config)


def test_dspark_quarot_full_dimension_probe_is_not_weakened_by_hidden_size(tmp_path):
    dimension = 1024
    theta = 0.5 * torch.asin(torch.tensor(0.1))
    a = torch.cos(theta)
    b = torch.sin(theta)
    rotation = torch.eye(dimension, dtype=torch.float32)
    rotation[:2, :2] = torch.stack((torch.stack((a, b)), torch.stack((b, a))))
    rotation_path = tmp_path / "quarot" / "rotation.safetensors"
    rotation_path.parent.mkdir()
    save_file({"global_rotation": rotation}, rotation_path)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=SimpleNamespace(
            quant_description={
                "optional": {
                    "quarot": {
                        "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
                    }
                }
            }
        ),
    )

    with pytest.raises(ValueError, match=r"Q.T @ Q probe max_abs_error"):
        _load_dspark_quarot_rotation(vllm_config)


def test_dspark_quarot_rotation_rejects_integer_dtype(tmp_path):
    rotation_path = tmp_path / "quarot" / "rotation.safetensors"
    rotation_path.parent.mkdir()
    save_file({"global_rotation": torch.eye(2, dtype=torch.int64)}, rotation_path)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=SimpleNamespace(
            quant_description={
                "optional": {
                    "quarot": {
                        "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
                    }
                }
            }
        ),
    )

    with pytest.raises(ValueError, match="floating dtype"):
        _load_dspark_quarot_rotation(vllm_config)


def test_dspark_quarot_rotation_rejects_hidden_size_mismatch(tmp_path):
    rotation_path = tmp_path / "quarot" / "rotation.safetensors"
    rotation_path.parent.mkdir()
    save_file({"global_rotation": torch.eye(2)}, rotation_path)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(hidden_size=3),
            )
        ),
        quant_config=SimpleNamespace(
            quant_description={
                "optional": {
                    "quarot": {
                        "rotation_map": {
                            "global_rotation": "quarot/rotation.safetensors",
                        }
                    }
                }
            }
        ),
    )

    with pytest.raises(ValueError, match=r"expected \(3, 3\), found \(2, 2\)"):
        _load_dspark_quarot_rotation(vllm_config)


def test_dspark_quarot_rotation_apply_respects_transpose_flag():
    hidden_states = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    rotation = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    torch.testing.assert_close(
        _apply_dspark_quarot_rotation(hidden_states, rotation, transpose=False),
        torch.tensor([[70.0, 100.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        _apply_dspark_quarot_rotation(hidden_states, rotation, transpose=True),
        torch.tensor([[50.0, 110.0]], dtype=torch.float32),
    )


def test_dspark_quarot_rotation_apply_requires_same_device():
    hidden_states = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    rotation = torch.empty((2, 2), dtype=torch.float32, device="meta")

    with pytest.raises(RuntimeError, match="must be loaded on the same device"):
        _apply_dspark_quarot_rotation(hidden_states, rotation, transpose=False)


def _add_quantized_weight(description: dict[str, Any], prefix: str, quant_type: str) -> None:
    description[f"{prefix}.weight"] = quant_type
    if quant_type in {"W8A8_DYNAMIC", "W4A8_DYNAMIC"}:
        description[f"{prefix}.weight_scale"] = quant_type
        description[f"{prefix}.weight_offset"] = quant_type
    if quant_type == "W4A8_DYNAMIC":
        description[f"{prefix}.scale_bias"] = quant_type


def _make_dspark_quant_description(
    *,
    num_stages: int = 2,
    num_experts: int = 2,
    expert_quant_type: str = "W8A8_DYNAMIC",
    main_proj_quant_type: str = "W8A8_DYNAMIC",
) -> dict[str, Any]:
    description: dict[str, Any] = {
        "version": "1.0.0",
        "group_size": 0,
    }
    for stage_idx in range(num_stages):
        stage = f"model.mtp.{stage_idx}"
        _add_quantized_weight(description, f"{stage}.self_attn.wq_a", "W8A8_DYNAMIC")
        _add_quantized_weight(description, f"{stage}.self_attn.wo_a", "FLOAT")
        for expert_idx in range(num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                _add_quantized_weight(
                    description,
                    f"{stage}.mlp.experts.{expert_idx}.{projection}",
                    expert_quant_type,
                )
    _add_quantized_weight(description, "model.mtp.0.main_proj", main_proj_quant_type)
    return description


def _make_dspark_quant_vllm_config(
    description: dict[str, Any],
    *,
    num_stages: int = 2,
    num_experts: int = 2,
    basis: str | None = None,
    hc_head_basis: str | None = None,
    mtp_dequantized_to_bf16: bool = False,
) -> SimpleNamespace:
    draft_hf_config = SimpleNamespace(
        n_mtp_layers=num_stages,
        n_routed_experts=num_experts,
        num_hidden_layers=43,
        dspark_mtp_dequantized_to_bf16=mtp_dequantized_to_bf16,
    )
    if basis is not None:
        draft_hf_config.dspark_quarot_draft_basis = basis
    if hc_head_basis is not None:
        draft_hf_config.dspark_quarot_hc_head_basis = hc_head_basis
    return SimpleNamespace(
        quant_config=AscendModelSlimConfig(description),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        ),
    )


@pytest.mark.parametrize("expert_quant_type", ["W8A8_DYNAMIC", "W4A8_DYNAMIC"])
def test_dspark_quant_description_accepts_w8_and_hybrid_w4(expert_quant_type):
    description = _make_dspark_quant_description(expert_quant_type=expert_quant_type)
    vllm_config = _make_dspark_quant_vllm_config(description)

    _validate_dspark_quant_description(vllm_config)


@pytest.mark.parametrize("expert_quant_type", ["W8A8_DYNAMIC", "W4A8_DYNAMIC"])
def test_dspark_quant_description_accepts_weight_only_schema(expert_quant_type):
    description = _make_dspark_quant_description(expert_quant_type=expert_quant_type)
    description = {
        name: quant_type
        for name, quant_type in description.items()
        if name in {"version", "group_size"} or name.endswith(".weight")
    }
    description["model.mtp.0.self_attn.fa_k.scale"] = "FAQuant"
    vllm_config = _make_dspark_quant_vllm_config(description)

    _validate_dspark_quant_description(vllm_config)


def _convert_dspark_description_to_direct_layers(
    description: dict[str, Any],
    *,
    start_layer_idx: int = 43,
) -> dict[str, Any]:
    direct: dict[str, Any] = {}
    for name, quant_type in description.items():
        if not name.startswith("model.mtp."):
            direct[name] = quant_type
            continue
        _, _, stage_idx, remainder = name.split(".", 3)
        direct[f"model.layers.{start_layer_idx + int(stage_idx)}.{remainder}"] = quant_type
    return direct


def test_dspark_direct_layer_quant_description_uses_same_safety_contract():
    description = _convert_dspark_description_to_direct_layers(_make_dspark_quant_description())
    vllm_config = _make_dspark_quant_vllm_config(description)

    _validate_dspark_quant_description(vllm_config)
    assert _draft_main_proj_quant_config(vllm_config) is vllm_config.quant_config

    description["model.layers.44.self_attn.wo_a.weight"] = "W8A8_DYNAMIC"
    with pytest.raises(ValueError, match="wo_a.*must remain FLOAT"):
        _validate_dspark_quant_description(_make_dspark_quant_vllm_config(description))


def test_dspark_quant_description_rejects_non_float_wo_a():
    description = _make_dspark_quant_description()
    _add_quantized_weight(description, "model.mtp.1.self_attn.wo_a", "W8A8_DYNAMIC")
    vllm_config = _make_dspark_quant_vllm_config(description)

    with pytest.raises(ValueError, match="wo_a.*must remain FLOAT"):
        _validate_dspark_quant_description(vllm_config)


def test_dspark_quant_description_rejects_missing_companion_tensor():
    description = _make_dspark_quant_description(expert_quant_type="W4A8_DYNAMIC")
    del description["model.mtp.1.mlp.experts.1.down_proj.scale_bias"]
    vllm_config = _make_dspark_quant_vllm_config(description)

    with pytest.raises(ValueError, match="scale_bias"):
        _validate_dspark_quant_description(vllm_config)


def test_dspark_quant_description_rejects_incomplete_expert_stage():
    description = _make_dspark_quant_description()
    for suffix in ("weight", "weight_scale", "weight_offset"):
        del description[f"model.mtp.1.mlp.experts.1.down_proj.{suffix}"]
    vllm_config = _make_dspark_quant_vllm_config(description)

    with pytest.raises(ValueError, match="stage 1.*expert 1.*down_proj"):
        _validate_dspark_quant_description(vllm_config)


def test_dspark_main_proj_quantization_is_explicit_and_fail_closed():
    w8_config = _make_dspark_quant_vllm_config(_make_dspark_quant_description())
    assert _draft_main_proj_quant_config(w8_config) is w8_config.quant_config

    float_description = _make_dspark_quant_description(main_proj_quant_type="FLOAT")
    float_config = _make_dspark_quant_vllm_config(float_description)
    assert _draft_main_proj_quant_config(float_config) is None

    absent_description = _make_dspark_quant_description()
    for suffix in ("weight", "weight_scale", "weight_offset"):
        del absent_description[f"model.mtp.0.main_proj.{suffix}"]
    absent_config = _make_dspark_quant_vllm_config(absent_description)
    with pytest.raises(ValueError, match="main_proj.*entry is missing"):
        _draft_main_proj_quant_config(absent_config)

    unrelated_description = _make_dspark_quant_description(main_proj_quant_type="W8A8_MXFP8")
    unrelated_config = _make_dspark_quant_vllm_config(unrelated_description)
    with pytest.raises(ValueError, match="main_proj.*unsupported quantization type"):
        _validate_dspark_quant_description(unrelated_config)


@pytest.mark.parametrize("bad_quant_type", ["W8A8_DYANMIC", 8])
def test_dspark_quant_description_rejects_unknown_stage_weight_type(bad_quant_type):
    description = _make_dspark_quant_description()
    description["model.mtp.1.mlp.experts.0.gate_proj.weight"] = bad_quant_type

    with pytest.raises(ValueError, match="unsupported draft weight quantization types"):
        _validate_dspark_quant_description(_make_dspark_quant_vllm_config(description))


def test_dspark_quant_description_accepts_all_float_stage_contract():
    description = _make_dspark_quant_description(main_proj_quant_type="FLOAT")
    description = {
        name: ("FLOAT" if name.endswith(".weight") else quant_type)
        for name, quant_type in description.items()
        if name in {"version", "group_size"} or name.endswith(".weight")
    }

    _validate_dspark_quant_description(_make_dspark_quant_vllm_config(description))


def test_dspark_main_proj_rejects_dense_w4_per_channel_contract():
    description = _make_dspark_quant_description(main_proj_quant_type="W4A8_DYNAMIC")
    vllm_config = _make_dspark_quant_vllm_config(description)

    with pytest.raises(ValueError, match="main_proj.*W4A8_DYNAMIC"):
        _draft_main_proj_quant_config(vllm_config)


def test_dspark_w4_quant_description_requires_explicit_group_size():
    description = _make_dspark_quant_description(expert_quant_type="W4A8_DYNAMIC")
    del description["group_size"]
    vllm_config = _make_dspark_quant_vllm_config(description)

    with pytest.raises(ValueError, match="declare group_size explicitly"):
        _validate_dspark_quant_description(vllm_config)


def test_dspark_weight_only_schema_derives_physical_checkpoint_companions():
    description = {
        "version": "1.0.0",
        "group_size": 0,
        "mtp.0.attn.wq_a.weight": "W8A8_DYNAMIC",
        "mtp.0.ffn.experts.0.w1.weight": "W4A8_DYNAMIC",
    }

    required = _required_dspark_checkpoint_tensor_groups(description)

    assert ("mtp.0.attn.wq_a.weight_scale", "mtp.0.attn.wq_a.scale") in required
    assert ("mtp.0.attn.wq_a.weight_offset",) in required
    assert ("mtp.0.ffn.experts.0.w1.scale_bias",) in required


def test_dspark_checkpoint_name_collapses_runtime_aliases_to_physical_schema():
    expected = "mtp.0.ffn.experts.7.w1.weight_scale"
    aliases = (
        "mtp.0.ffn.experts.7.w1.weight_scale",
        "mtp.0.ffn.experts.7.gate_proj.weight_scale",
        "mtp.0.mlp.experts.7.w1.weight_scale",
        "mtp.0.mlp.experts.7.gate_proj.weight_scale",
        "model.mtp.0.mlp.experts.7.gate_proj.weight_scale",
        "model.layers.43.mlp.experts.7.gate_proj.weight_scale",
    )

    for name in aliases:
        assert (
            _dspark_checkpoint_name_for_quant_key(
                name,
                start_layer_idx=43,
                num_dspark_layers=3,
            )
            == expected
        )

    description = {name.replace(".weight_scale", ".weight"): "W8A8_DYNAMIC" for name in aliases}
    required = _required_dspark_checkpoint_tensor_groups(
        description,
        start_layer_idx=43,
        num_dspark_layers=3,
    )
    assert required == [
        ("mtp.0.ffn.experts.7.w1.weight_offset",),
        ("mtp.0.ffn.experts.7.w1.weight_scale", "mtp.0.ffn.experts.7.w1.scale"),
    ]


def test_dspark_rank_local_missing_check_only_exempts_ep_filtered_base_weights():
    expert_weight = "mtp.0.ffn.experts.7.w1.weight"
    expert_scale = "mtp.0.ffn.experts.7.w1.weight_scale"
    dense_weight = "mtp.0.attn.wq_a.weight"
    expected = {expert_weight, expert_scale, dense_weight}
    seen = {expert_scale}

    assert _missing_dspark_checkpoint_tensors(
        expected,
        seen,
        local_expert_ids_by_stage={0: {0}},
    ) == [dense_weight]
    assert _missing_dspark_checkpoint_tensors(
        expected,
        seen,
        local_expert_ids_by_stage={0: {7}},
    ) == [dense_weight, expert_weight]
    assert _missing_dspark_checkpoint_tensors(
        expected,
        seen,
        local_expert_ids_by_stage=None,
    ) == [dense_weight, expert_weight]
    assert _missing_dspark_checkpoint_tensors(
        expected,
        {dense_weight},
        local_expert_ids_by_stage={0: {0}},
    ) == [expert_scale]


def test_modelslim_source_manifest_is_not_polluted_by_runtime_aliases():
    source = {
        "hc_head_fn": "FLOAT",
        "mtp.0.ffn.experts.7.w1.weight": "W8A8_DYNAMIC",
        "mtp.0.ffn.experts.7.w1.weight_scale": "W8A8_DYNAMIC",
        "mtp.0.ffn.experts.7.w1.weight_offset": "W8A8_DYNAMIC",
    }

    quant_config = AscendModelSlimConfig(source.copy())

    assert quant_config.source_quant_description == source
    assert "mtp.0.ffn.experts.7.gate_proj.weight" in quant_config.quant_description
    assert "mtp.0.ffn.experts.7.gate_proj.weight" not in quant_config.source_quant_description
    assert _required_dspark_checkpoint_tensor_groups(
        quant_config.source_quant_description,
        start_layer_idx=43,
        num_dspark_layers=3,
    ) == [
        ("mtp.0.ffn.experts.7.w1.weight_offset",),
        ("mtp.0.ffn.experts.7.w1.weight_scale", "mtp.0.ffn.experts.7.w1.scale"),
    ]


def _write_dspark_test_index(tmp_path: Path, tensors: dict[str, torch.Tensor], weight_map: dict[str, str]):
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    if tensors:
        save_file(tensors, tmp_path / shard_name)
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    return shard_name


def test_dspark_checkpoint_index_validates_index_shards_and_headers(tmp_path):
    tensors = {
        "mtp.0.main_proj.weight": torch.ones((1, 1), dtype=torch.int8),
        "mtp.0.main_proj.weight_scale": torch.ones((1, 1)),
        "mtp.0.main_proj.weight_offset": torch.zeros((1, 1)),
    }
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    _write_dspark_test_index(tmp_path, tensors, {name: shard_name for name in tensors})

    assert _validate_dspark_checkpoint_index(
        str(tmp_path),
        set(tensors),
        [
            ("mtp.0.main_proj.weight_scale", "mtp.0.main_proj.scale"),
            ("mtp.0.main_proj.weight_offset",),
        ],
    )


@pytest.mark.parametrize(
    ("tensor", "expected_kind", "message"),
    [
        (torch.ones((2, 2), dtype=torch.bfloat16), "int8", "declared quantized"),
        (torch.ones((2, 2), dtype=torch.int8), "floating", "declared FLOAT"),
    ],
)
def test_dspark_checkpoint_index_rejects_manifest_physical_dtype_mismatch(
    tmp_path,
    tensor,
    expected_kind,
    message,
):
    name = "mtp.0.main_proj.weight"
    shard_name = _write_dspark_test_index(
        tmp_path,
        {name: tensor},
        {name: "quant_model_weights-00001-of-00001.safetensors"},
    )
    assert shard_name == "quant_model_weights-00001-of-00001.safetensors"

    with pytest.raises(ValueError, match=message):
        _validate_dspark_checkpoint_index(
            str(tmp_path),
            {name},
            [],
            expected_weight_dtype_kinds={name: expected_kind},
        )


def test_dspark_checkpoint_index_rejects_int8_when_declared_dequantized(tmp_path):
    name = "mtp.0.main_proj.weight"
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    _write_dspark_test_index(
        tmp_path,
        {name: torch.ones((2, 2), dtype=torch.int8)},
        {name: shard_name},
    )

    with pytest.raises(ValueError, match="declared FLOAT"):
        _validate_dspark_checkpoint_index(
            str(tmp_path),
            set(),
            [],
            require_mtp_float_weights=True,
        )


def test_dspark_checkpoint_index_rejects_main_proj_shape_mismatch(tmp_path):
    name = "mtp.0.main_proj.weight"
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    _write_dspark_test_index(
        tmp_path,
        {name: torch.ones((2, 3), dtype=torch.int8)},
        {name: shard_name},
    )

    with pytest.raises(ValueError, match=r"physical shape \(2, 3\); expected \(2, 4\)"):
        _validate_dspark_checkpoint_index(
            str(tmp_path),
            {name},
            [],
            expected_weight_shapes={name: (2, 4)},
        )


def test_dspark_stream_weight_dtype_and_shape_gates():
    _validate_dspark_loaded_weight_dtype(
        "mtp.0.main_proj.weight",
        torch.ones((2, 4), dtype=torch.int8),
        "int8",
    )
    _validate_dspark_loaded_weight_dtype(
        "mtp.0.attn.wo_a.weight",
        torch.ones((2, 4), dtype=torch.float32),
        "floating",
    )
    _validate_dspark_loaded_weight_shape(
        "mtp.0.main_proj.weight",
        torch.ones((2, 4), dtype=torch.int8),
        (2, 4),
    )

    with pytest.raises(ValueError, match="expected torch.int8"):
        _validate_dspark_loaded_weight_dtype(
            "mtp.0.main_proj.weight",
            torch.ones((2, 4), dtype=torch.uint8),
            "int8",
        )
    with pytest.raises(ValueError, match=r"physical shape \(2, 3\); expected \(2, 4\)"):
        _validate_dspark_loaded_weight_shape(
            "mtp.0.main_proj.weight",
            torch.ones((2, 3), dtype=torch.int8),
            (2, 4),
        )


def test_dspark_checkpoint_global_shape_contracts():
    config = SimpleNamespace(
        hidden_size=2,
        dspark_target_layer_ids=[40, 41],
        num_attention_heads=4,
        head_dim=8,
        o_groups=2,
        o_lora_rank=3,
        n_mtp_layers=2,
    )

    assert _dspark_checkpoint_weight_shapes(config) == {
        "mtp.0.main_proj.weight": (2, 4),
        "mtp.0.attn.wo_a.weight": (6, 16),
        "mtp.1.attn.wo_a.weight": (6, 16),
    }


def test_dspark_checkpoint_index_rejects_missing_companion(tmp_path):
    tensors = {
        "mtp.0.main_proj.weight": torch.ones((1, 1), dtype=torch.int8),
        "mtp.0.main_proj.weight_scale": torch.ones((1, 1)),
    }
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    _write_dspark_test_index(tmp_path, tensors, {name: shard_name for name in tensors})

    with pytest.raises(ValueError, match="checkpoint index is missing.*companion"):
        _validate_dspark_checkpoint_index(
            str(tmp_path),
            set(tensors),
            [("mtp.0.main_proj.weight_offset",)],
        )


def test_dspark_checkpoint_index_rejects_stale_header_or_missing_shard(tmp_path):
    weight_name = "mtp.0.main_proj.weight"
    scale_name = "mtp.0.main_proj.weight_scale"
    shard_name = "quant_model_weights-00001-of-00001.safetensors"
    _write_dspark_test_index(
        tmp_path,
        {weight_name: torch.ones((1, 1), dtype=torch.int8)},
        {weight_name: shard_name, scale_name: shard_name},
    )

    with pytest.raises(ValueError, match="missing indexed tensors"):
        _validate_dspark_checkpoint_index(str(tmp_path), {weight_name, scale_name}, [])

    (tmp_path / shard_name).unlink()
    with pytest.raises(ValueError, match="missing shard"):
        _validate_dspark_checkpoint_index(str(tmp_path), {weight_name}, [])

    direct_description = {
        "model.layers.43.self_attn.wq_a.weight": "W8A8_DYNAMIC",
        "model.layers.43.self_attn.attn_sink": "FLOAT",
        "model.layers.43.mlp.gate.e_score_correction_bias": "FLOAT",
        "model.layers.43.self_attn.indexer.quant_type": "INT8_DYNAMIC",
        "model.layers.43.self_attn.indexer.wq_b_weight": "INT8_DYNAMIC",
        "model.mtp.1.self_attn.indexer.quant_type": "INT8_DYNAMIC",
        "model.mtp.1.self_attn.indexer.wq_b_weight": "INT8_DYNAMIC",
        "mtp.2.attn.indexer.quant_type": "INT8_DYNAMIC",
        "mtp.2.attn.indexer.wq_b_weight": "INT8_DYNAMIC",
        "model.layers.43.self_attn.indexer.wq_b.weight": "W8A8_DYNAMIC",
        "model.mtp.1.self_attn.indexer.q_rot": "FLOAT",
        "mtp.2.attn.indexer.k_rot": "FLOAT",
        "model.layers.43.self_attn.fa_k.offset": "FLOAT",
        "model.layers.43.self_attn.foo.quant_type": "FLOAT",
    }
    assert _required_dspark_checkpoint_tensors(
        direct_description,
        start_layer_idx=43,
        num_dspark_layers=3,
    ) == {
        "mtp.0.attn.wq_a.weight",
        "mtp.0.attn.attn_sink",
        "mtp.0.ffn.gate.bias",
        "mtp.0.attn.indexer.wq_b.weight",
        "mtp.1.attn.indexer.q_rot",
        "mtp.2.attn.indexer.k_rot",
        "mtp.0.attn.fa_k.offset",
        "mtp.0.attn.foo.quant_type",
    }


def test_dspark_quarot_basis_transitions_are_directional():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    hidden_states = torch.tensor([[1.0, 2.0]])

    torch.testing.assert_close(
        _transition_dspark_quarot_basis(
            hidden_states,
            rotation,
            draft_basis="legacy",
            direction="target_to_draft",
        ),
        torch.tensor([[-2.0, 1.0]]),
    )
    torch.testing.assert_close(
        _transition_dspark_quarot_basis(
            hidden_states,
            rotation,
            draft_basis="legacy",
            direction="draft_to_target",
        ),
        torch.tensor([[2.0, -1.0]]),
    )
    assert (
        _transition_dspark_quarot_basis(
            hidden_states,
            rotation,
            draft_basis="rotated",
            direction="target_to_draft",
        )
        is hidden_states
    )
    assert (
        _transition_dspark_quarot_basis(
            hidden_states,
            rotation,
            draft_basis="rotated_decoder",
            direction="draft_to_target",
        )
        is hidden_states
    )


def test_dspark_quarot_basis_is_applied_at_embedding_and_head_boundaries():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    target_embedding = torch.tensor([[1.0, 2.0]])

    legacy_model = SimpleNamespace(
        embed_tokens=lambda _input_ids: target_embedding,
        _dspark_quarot_rotation=rotation,
        _dspark_quarot_draft_basis="legacy",
    )
    rotated_model = SimpleNamespace(
        embed_tokens=lambda _input_ids: target_embedding,
        _dspark_quarot_rotation=rotation,
        _dspark_quarot_draft_basis="rotated",
    )
    torch.testing.assert_close(
        DeepseekV4DSparkModel.embed_input_ids(legacy_model, torch.tensor([1])),
        torch.tensor([[-2.0, 1.0]]),
    )
    assert DeepseekV4DSparkModel.embed_input_ids(rotated_model, torch.tensor([1])) is target_embedding
    rotated_model._dspark_quarot_draft_basis = "rotated_decoder"
    assert DeepseekV4DSparkModel.embed_input_ids(rotated_model, torch.tensor([1])) is target_embedding

    class ScaledNorm:
        weight = torch.tensor([2.0, 4.0])

        def __call__(self, hidden_states):
            return hidden_states * self.weight

    canonical_head_hidden = torch.tensor([[1.0, 2.0]])
    head_model = SimpleNamespace(
        compute_head_hidden=lambda _hidden_states: canonical_head_hidden,
        norm=ScaledNorm(),
        _dspark_quarot_rotation=rotation,
        _dspark_quarot_draft_basis="legacy",
    )
    head_input = DeepseekV4DSparkModel.compute_logits(
        head_model,
        hidden_states=torch.empty(0),
        lm_head=object(),
        logits_processor=lambda _lm_head, hidden_states: hidden_states,
    )
    torch.testing.assert_close(head_input, torch.tensor([[2.0, -1.0]]))
    head_model._dspark_quarot_draft_basis = "rotated"
    rotated_head_input = DeepseekV4DSparkModel.compute_logits(
        head_model,
        hidden_states=torch.empty(0),
        lm_head=object(),
        logits_processor=lambda _lm_head, hidden_states: hidden_states,
    )
    torch.testing.assert_close(rotated_head_input, canonical_head_hidden)
    head_model._dspark_quarot_draft_basis = "rotated_decoder"
    rotated_decoder_head_input = DeepseekV4DSparkModel.compute_logits(
        head_model,
        hidden_states=torch.empty(0),
        lm_head=object(),
        logits_processor=lambda _lm_head, hidden_states: hidden_states,
    )
    torch.testing.assert_close(rotated_decoder_head_input, canonical_head_hidden)
    head_model._dspark_quarot_rotation = None
    no_quarot_head_input = DeepseekV4DSparkModel.compute_logits(
        head_model,
        hidden_states=torch.empty(0),
        lm_head=object(),
        logits_processor=lambda _lm_head, hidden_states: hidden_states,
    )
    torch.testing.assert_close(no_quarot_head_input, torch.tensor([[2.0, 8.0]]))


def test_dspark_unrotated_main_proj_input_unrotates_each_target_block():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    context_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    converted = _prepare_dspark_main_proj_input(
        context_states,
        rotation,
        draft_basis="unrotated",
        hidden_size=2,
    )

    torch.testing.assert_close(converted, torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))
    torch.testing.assert_close(
        _prepare_dspark_main_proj_input(
            context_states,
            rotation,
            draft_basis="rotated_decoder",
            hidden_size=2,
        ),
        converted,
    )
    assert (
        _prepare_dspark_main_proj_input(
            context_states,
            rotation,
            draft_basis="legacy",
            hidden_size=2,
        )
        is context_states
    )
    assert (
        _prepare_dspark_main_proj_input(
            context_states,
            rotation,
            draft_basis="rotated",
            hidden_size=2,
        )
        is context_states
    )


def test_dspark_rotated_decoder_main_proj_output_rotates_after_canonical_norm():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    canonical_normalized = torch.tensor([[1.0, 2.0]])

    torch.testing.assert_close(
        _prepare_dspark_main_proj_output(
            canonical_normalized,
            rotation,
            draft_basis="rotated_decoder",
        ),
        torch.tensor([[2.0, -1.0]]),
    )
    for basis in ("legacy", "unrotated", "rotated"):
        assert (
            _prepare_dspark_main_proj_output(
                canonical_normalized,
                rotation,
                draft_basis=basis,
            )
            is canonical_normalized
        )


def test_dspark_rotated_decoder_bridges_canonical_hc_head():
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    canonical_hidden = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    rotated_hidden = torch.matmul(canonical_hidden, rotation)
    hc_head_fn = torch.tensor(
        [
            [0.5, -0.25, 0.75, 0.125],
            [-0.5, 0.25, 0.125, 0.75],
        ]
    )
    hc_head_scale = torch.tensor([0.8, 1.2])
    hc_head_base = torch.tensor([-0.1, 0.2])
    norm_eps = 1e-6
    hc_eps = 1e-4

    canonical_output = _hc_head_torch(
        canonical_hidden,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        norm_eps,
        hc_eps,
    )
    actual = _compute_dspark_hc_head(
        rotated_hidden,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        norm_eps,
        hc_eps,
        rotation,
        draft_basis="rotated_decoder",
        hc_head_basis="canonical",
    )

    torch.testing.assert_close(actual, torch.matmul(canonical_output, rotation))


def test_dspark_rotated_hc_head_load_time_fold_is_mathematically_equivalent():
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    canonical_hidden = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    rotated_hidden = torch.matmul(canonical_hidden, rotation)
    canonical_hc_head_fn = torch.tensor(
        [
            [0.5, -0.25, 0.75, 0.125],
            [-0.5, 0.25, 0.125, 0.75],
        ]
    )
    source_before = canonical_hc_head_fn.clone()
    hc_head_scale = torch.tensor([0.8, 1.2])
    hc_head_base = torch.tensor([-0.1, 0.2])
    norm_eps = 1e-6
    hc_eps = 1e-4

    rotated_hc_head_fn = _derive_dspark_rotated_hc_head_fn(canonical_hc_head_fn, rotation)
    folded_output = _compute_dspark_hc_head(
        rotated_hidden,
        rotated_hc_head_fn,
        hc_head_scale,
        hc_head_base,
        norm_eps,
        hc_eps,
        rotation,
        draft_basis="rotated_decoder",
        hc_head_basis="rotated",
    )
    canonical_output = _hc_head_torch(
        canonical_hidden,
        canonical_hc_head_fn,
        hc_head_scale,
        hc_head_base,
        norm_eps,
        hc_eps,
    )

    torch.testing.assert_close(folded_output, torch.matmul(canonical_output, rotation))
    torch.testing.assert_close(canonical_hc_head_fn, source_before)
    assert rotated_hc_head_fn.data_ptr() != canonical_hc_head_fn.data_ptr()


def test_dspark_rotated_hc_head_fold_fails_fast_on_shape_device_and_dtype():
    weight = torch.ones((2, 4), dtype=torch.float32)
    rotation = torch.eye(2, dtype=torch.float32)

    with pytest.raises(ValueError, match="rank 2"):
        _derive_dspark_rotated_hc_head_fn(weight.unsqueeze(0), rotation)
    with pytest.raises(ValueError, match="must be square"):
        _derive_dspark_rotated_hc_head_fn(weight, torch.ones((2, 3)))
    with pytest.raises(ValueError, match=r"expected 6, found 4"):
        _derive_dspark_rotated_hc_head_fn(weight, torch.eye(3))
    with pytest.raises(TypeError, match="same dtype"):
        _derive_dspark_rotated_hc_head_fn(weight, rotation.to(torch.float64))
    with pytest.raises(TypeError, match="floating-point"):
        _derive_dspark_rotated_hc_head_fn(weight.to(torch.int32), rotation.to(torch.int32))
    with pytest.raises(RuntimeError, match="same device"):
        _derive_dspark_rotated_hc_head_fn(weight, torch.empty((2, 2), device="meta"))


def _make_dspark_hc_fold_test_model(
    canonical_hc_head_fn: torch.Tensor,
    rotation: torch.Tensor | None,
    *,
    draft_basis: str = "rotated_decoder",
    hc_head_basis: str = "canonical",
):
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    torch.nn.Module.__init__(model)
    model.hc_head_fn = torch.nn.Parameter(canonical_hc_head_fn.clone(), requires_grad=False)
    model.hc_head_scale = torch.nn.Parameter(torch.tensor([0.8, 1.2]), requires_grad=False)
    model.hc_head_base = torch.nn.Parameter(torch.tensor([-0.1, 0.2]), requires_grad=False)
    model.config = SimpleNamespace(rms_norm_eps=1e-6, hc_eps=1e-4)
    model._dspark_quarot_draft_basis = draft_basis
    model._dspark_quarot_hc_head_basis = hc_head_basis
    model.register_buffer("_dspark_quarot_rotation", rotation, persistent=False)
    model.register_buffer("_dspark_quarot_rotated_hc_head_fn", None, persistent=False)
    return model


def test_dspark_rotated_hc_head_fold_reload_is_idempotent_and_non_persistent():
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    canonical_v1 = torch.tensor(
        [
            [0.5, -0.25, 0.75, 0.125],
            [-0.5, 0.25, 0.125, 0.75],
        ]
    )
    model = _make_dspark_hc_fold_test_model(canonical_v1, rotation)
    source_parameter = model.hc_head_fn

    model.install_quarot_hc_head_fold()
    installed_buffer = model._dspark_quarot_rotated_hc_head_fn
    assert installed_buffer is not None
    installed_data_ptr = installed_buffer.data_ptr()
    first_derived = model._dspark_quarot_rotated_hc_head_fn.clone()
    torch.testing.assert_close(first_derived, _derive_dspark_rotated_hc_head_fn(canonical_v1, rotation))
    torch.testing.assert_close(model.hc_head_fn, canonical_v1)
    assert model.hc_head_fn is source_parameter
    assert "_dspark_quarot_rotated_hc_head_fn" not in model.state_dict()

    canonical_v2 = canonical_v1 + 0.25
    with torch.no_grad():
        model.hc_head_fn.copy_(canonical_v2)
    model.install_quarot_hc_head_fold()
    expected_v2 = _derive_dspark_rotated_hc_head_fn(canonical_v2, rotation)
    torch.testing.assert_close(model._dspark_quarot_rotated_hc_head_fn, expected_v2)
    torch.testing.assert_close(model.hc_head_fn, canonical_v2)
    assert model._dspark_quarot_rotated_hc_head_fn is installed_buffer
    assert model._dspark_quarot_rotated_hc_head_fn.data_ptr() == installed_data_ptr
    assert not torch.equal(model._dspark_quarot_rotated_hc_head_fn, first_derived)

    model.install_quarot_hc_head_fold()
    torch.testing.assert_close(model._dspark_quarot_rotated_hc_head_fn, expected_v2)
    torch.testing.assert_close(model.hc_head_fn, canonical_v2)
    assert model._dspark_quarot_rotated_hc_head_fn is installed_buffer
    assert model._dspark_quarot_rotated_hc_head_fn.data_ptr() == installed_data_ptr


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_dspark_rotated_hc_head_fold_replaces_incompatible_buffer_before_capture(mismatch):
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    canonical = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    model = _make_dspark_hc_fold_test_model(canonical, rotation)
    expected = _derive_dspark_rotated_hc_head_fn(canonical, rotation)

    if mismatch == "shape":
        incompatible = torch.empty((1, 1), dtype=expected.dtype)
    elif mismatch == "dtype":
        incompatible = torch.empty_like(expected, dtype=torch.float64)
    else:
        incompatible = torch.empty_like(expected, device="meta")
    model._dspark_quarot_rotated_hc_head_fn = incompatible

    model.install_quarot_hc_head_fold()

    assert model._dspark_quarot_rotated_hc_head_fn is not incompatible
    assert model._dspark_quarot_rotated_hc_head_fn.shape == expected.shape
    assert model._dspark_quarot_rotated_hc_head_fn.dtype == expected.dtype
    assert model._dspark_quarot_rotated_hc_head_fn.device == expected.device
    torch.testing.assert_close(model._dspark_quarot_rotated_hc_head_fn, expected)


@pytest.mark.parametrize(
    ("draft_basis", "hc_head_basis", "with_rotation"),
    [
        ("legacy", "canonical", True),
        ("unrotated", "canonical", True),
        ("rotated", "rotated", True),
        ("rotated_decoder", "rotated", True),
        ("rotated", "canonical", False),
        ("rotated_decoder", "canonical", False),
    ],
)
def test_dspark_rotated_hc_head_fold_only_installs_for_canonical_head_in_rotated_decoder(
    draft_basis,
    hc_head_basis,
    with_rotation,
):
    canonical = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    rotation = torch.eye(2) if with_rotation else None
    model = _make_dspark_hc_fold_test_model(
        canonical,
        rotation,
        draft_basis=draft_basis,
        hc_head_basis=hc_head_basis,
    )
    model._dspark_quarot_rotated_hc_head_fn = torch.full_like(canonical, -123.0)

    model.install_quarot_hc_head_fold()

    assert model._dspark_quarot_rotated_hc_head_fn is None
    torch.testing.assert_close(model.hc_head_fn, canonical)


@pytest.mark.parametrize("draft_basis", ["rotated", "rotated_decoder"])
def test_dspark_rotated_hc_head_fold_enables_runtime_fast_path(monkeypatch, draft_basis):
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    canonical_hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    canonical_hc_head_fn = torch.tensor(
        [
            [0.5, -0.25, 0.75, 0.125],
            [-0.5, 0.25, 0.125, 0.75],
        ]
    )
    model = _make_dspark_hc_fold_test_model(
        canonical_hc_head_fn,
        rotation,
        draft_basis=draft_basis,
    )
    model.install_quarot_hc_head_fold()
    rotated_hidden = torch.matmul(canonical_hidden, rotation)
    expected = _hc_head_torch(
        rotated_hidden,
        model._dspark_quarot_rotated_hc_head_fn,
        model.hc_head_scale,
        model.hc_head_base,
        model.config.rms_norm_eps,
        model.config.hc_eps,
    )

    def fail_runtime_rotation(*_args, **_kwargs):
        raise AssertionError("load-time folded HC head must not rotate hidden states at runtime")

    monkeypatch.setattr(dspark_model_module, "_apply_dspark_quarot_rotation", fail_runtime_rotation)

    actual = DeepseekV4DSparkModel.compute_head_hidden(model, rotated_hidden)

    torch.testing.assert_close(actual, expected)


def test_dspark_quarot_main_proj_weight_contracts_are_equivalent():
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    canonical_context = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    canonical_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    rotated_context = torch.matmul(canonical_context.reshape(1, 2, 2), rotation).reshape(1, 4)
    canonical_output = F.linear(canonical_context, canonical_weight)

    runtime_unrotated_context = _prepare_dspark_main_proj_input(
        rotated_context,
        rotation,
        draft_basis="unrotated",
        hidden_size=2,
    )
    torch.testing.assert_close(F.linear(runtime_unrotated_context, canonical_weight), canonical_output)

    rotated_decoder_output = _prepare_dspark_main_proj_output(
        F.linear(runtime_unrotated_context, canonical_weight),
        rotation,
        draft_basis="rotated_decoder",
    )
    torch.testing.assert_close(rotated_decoder_output, torch.matmul(canonical_output, rotation))

    legacy_weight = torch.matmul(canonical_weight.reshape(2, 2, 2), rotation).reshape(2, 4)
    torch.testing.assert_close(F.linear(rotated_context, legacy_weight), canonical_output)

    rotated_weight = torch.matmul(rotation.t(), legacy_weight)
    torch.testing.assert_close(
        F.linear(rotated_context, rotated_weight),
        torch.matmul(canonical_output, rotation),
    )


def test_dspark_quantized_quarot_main_proj_requires_explicit_basis():
    description = _make_dspark_quant_description()
    description["optional"] = {
        "quarot": {
            "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
        }
    }
    vllm_config = _make_dspark_quant_vllm_config(description)

    with pytest.raises(ValueError, match="dspark_quarot_draft_basis"):
        _get_dspark_quarot_draft_basis(vllm_config)

    explicit = _make_dspark_quant_vllm_config(description, basis="rotated")
    assert _get_dspark_quarot_draft_basis(explicit) == "rotated"

    split_contract = _make_dspark_quant_vllm_config(description, basis="rotated_decoder")
    assert _get_dspark_quarot_draft_basis(split_contract) == "rotated_decoder"


def test_dspark_rotated_decoder_requires_explicit_hc_head_basis():
    description = _make_dspark_quant_description()
    description["optional"] = {
        "quarot": {
            "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
        }
    }
    ambiguous = _make_dspark_quant_vllm_config(description, basis="rotated_decoder")
    with pytest.raises(ValueError, match="requires an explicit dspark_quarot_hc_head_basis"):
        _get_dspark_quarot_hc_head_basis(ambiguous, draft_basis="rotated_decoder")

    canonical = _make_dspark_quant_vllm_config(
        description,
        basis="rotated_decoder",
        hc_head_basis="canonical",
    )
    assert _get_dspark_quarot_hc_head_basis(canonical, draft_basis="rotated_decoder") == "canonical"

    rotated = _make_dspark_quant_vllm_config(
        description,
        basis="rotated_decoder",
        hc_head_basis="rotated",
    )
    assert _get_dspark_quarot_hc_head_basis(rotated, draft_basis="rotated_decoder") == "rotated"

    dequantized = _make_dspark_quant_vllm_config(
        description,
        basis="unrotated",
        mtp_dequantized_to_bf16=True,
    )
    assert _draft_quant_config(dequantized) is None
    assert _get_dspark_quarot_draft_basis(dequantized) == "unrotated"


@pytest.mark.parametrize("main_proj_metadata", ["w8", "float", "absent"])
def test_dspark_quarot_requires_explicit_draft_basis(main_proj_metadata):
    description = _make_dspark_quant_description(main_proj_quant_type="FLOAT")
    if main_proj_metadata == "w8":
        description["model.mtp.0.main_proj.weight"] = "W8A8_DYNAMIC"
    if main_proj_metadata == "absent":
        del description["model.mtp.0.main_proj.weight"]
    description["optional"] = {
        "quarot": {
            "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
        }
    }

    with pytest.raises(ValueError, match="requires an explicit.*dspark_quarot_draft_basis"):
        _get_dspark_quarot_draft_basis(_make_dspark_quant_vllm_config(description))

    assert _get_dspark_quarot_draft_basis(_make_dspark_quant_vllm_config(description, basis="legacy")) == "legacy"


def test_dspark_quarot_without_rotation_keeps_legacy_default():
    description = _make_dspark_quant_description(main_proj_quant_type="FLOAT")

    assert _get_dspark_quarot_draft_basis(_make_dspark_quant_vllm_config(description)) == "legacy"


def test_dspark_quarot_basis_supports_metadata_and_rejects_conflicts():
    description = _make_dspark_quant_description()
    description["optional"] = {
        "quarot": {
            "rotation_map": {"global_rotation": "quarot/rotation.safetensors"},
            "dspark_draft_basis": "rotated",
        }
    }
    assert _get_dspark_quarot_draft_basis(_make_dspark_quant_vllm_config(description)) == "rotated"

    with pytest.raises(ValueError, match="Conflicting dspark_quarot_draft_basis"):
        _get_dspark_quarot_draft_basis(_make_dspark_quant_vllm_config(description, basis="unrotated"))


def test_dspark_num_mtp_layers_prefers_upstream_config_name():
    config = SimpleNamespace(n_mtp_layers=4, dspark_num_mtp_layers=2)

    assert _get_dspark_num_mtp_layers(config) == 4


def test_dspark_num_mtp_layers_keeps_legacy_config_fallback():
    config = SimpleNamespace(dspark_num_mtp_layers=2)

    assert _get_dspark_num_mtp_layers(config) == 2


def test_dspark_fp8_qdq_is_disabled_for_bf16_dequantized_checkpoints():
    assert _should_apply_dspark_fp8_qdq(SimpleNamespace(dspark_mtp_dequantized_to_bf16=True)) is False
    assert _should_apply_dspark_fp8_qdq(SimpleNamespace(dspark_full_dequantized_to_bf16=True)) is False
    assert _should_apply_dspark_fp8_qdq(SimpleNamespace()) is True


def test_dspark_draft_quant_config_supports_bf16_and_w4a8():
    quant_config = object()

    def make_config(*, parent_quant_config, mtp_dequantized_to_bf16=False):
        draft_config = SimpleNamespace(dspark_mtp_dequantized_to_bf16=mtp_dequantized_to_bf16)
        return SimpleNamespace(
            quant_config=parent_quant_config,
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(hf_config=draft_config),
            ),
        )

    assert _draft_quant_config(make_config(parent_quant_config=None)) is None
    assert _draft_quant_config(make_config(parent_quant_config=quant_config)) is quant_config
    assert (
        _draft_quant_config(
            make_config(
                parent_quant_config=quant_config,
                mtp_dequantized_to_bf16=True,
            )
        )
        is None
    )


def test_dspark_fp8_qdq_helpers_return_input_when_disabled():
    kv = torch.randn(2, 128)
    out = torch.randn(2, 2, 128)

    assert _maybe_fp8_qdq_nope_dims(kv, nope_head_dim=64, apply_fp8_qdq=False) is kv
    assert _maybe_fp8_e4m3fn_qdq(out, apply_fp8_qdq=False, block_size=128) is out


def test_dspark_fp8_qdq_rejects_incompatible_block_size():
    with pytest.raises(ValueError, match="last_dim=65, block_size=64"):
        _maybe_fp8_qdq_nope_dims(
            torch.randn(2, 65),
            nope_head_dim=65,
            apply_fp8_qdq=True,
        )


def test_dspark_markov_head_w2_uses_model_default_dtype(monkeypatch):
    captured = {}

    class FakeEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dspark_model_module, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(dspark_model_module, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(dspark_model_module, "LogitsProcessor", lambda vocab_size: object())

    dspark_model_module.DSparkMarkovHead(
        SimpleNamespace(vocab_size=16, dspark_markov_rank=4),
        prefix="model.layers.45.markov_head",
    )

    assert "params_dtype" not in captured["kwargs"]
    assert captured["kwargs"]["org_num_embeddings"] == 16


def test_dspark_attention_uses_upstream_no_compression_ratio(monkeypatch):
    def fake_base_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.dsa_attn = SimpleNamespace(compress_ratio=0)
        self.window_size = 8
        self.n_local_heads = 2
        self.head_dim = 4

    monkeypatch.setattr(dspark_model_module.DeepseekV4Attention, "__init__", fake_base_init)
    monkeypatch.setattr(dspark_model_module, "current_platform", SimpleNamespace(device_type="cpu"))

    attn = DeepseekV4DSparkAttention(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=16),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        ),
        config=SimpleNamespace(
            dspark_block_size=5,
            dspark_mtp_dequantized_to_bf16=True,
        ),
    )

    assert attn.compress_ratio == 1
    assert attn.dsa_attn.compress_ratio == 1


def test_dspark_target_layer_ids_match_upstream_aux_capture_semantics():
    target_layer_ids = {40, 41, 42}

    assert not _is_dspark_target_layer(39, target_layer_ids)
    assert _is_dspark_target_layer(40, target_layer_ids)
    assert _is_dspark_target_layer(41, target_layer_ids)
    assert _is_dspark_target_layer(42, target_layer_ids)


def test_dspark_remap_skips_unused_confidence_head_weights():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=61))

    assert (
        DeepSeekV4DSparkMTP._remap_dspark_name(
            model,
            "mtp.2.confidence_head.proj.weight",
        )
        is None
    )


def test_dspark_remap_loads_moe_gate_correction_bias():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=43))

    assert (
        DeepSeekV4DSparkMTP._remap_dspark_name(
            model,
            "mtp.1.ffn.gate.bias",
        )
        == "model.layers.44.mlp.gate.e_score_correction_bias"
    )


def test_dspark_remap_covers_representative_checkpoint_names():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=43))

    cases = {
        "mtp.0.main_proj.weight": "model.layers.43.main_proj.weight",
        "mtp.0.main_norm.weight": "model.layers.43.main_norm.weight",
        "mtp.0.attn.attn_sink": "model.layers.43.self_attn.attn_sink",
        "mtp.0.attn.wq_a.weight": "model.layers.43.self_attn.wq_a.weight",
        "mtp.0.attn.wkv.weight": "model.layers.43.self_attn.wkv.weight",
        "mtp.1.ffn.shared_experts.w1.weight": ("model.layers.44.mlp.shared_experts.gate_proj.weight"),
        "mtp.1.ffn.shared_experts.w2.weight": ("model.layers.44.mlp.shared_experts.down_proj.weight"),
        "mtp.1.ffn.shared_experts.w3.weight": ("model.layers.44.mlp.shared_experts.up_proj.weight"),
        "mtp.2.hc_head_fn": "model.layers.45.hc_head_fn",
        "mtp.2.markov_head.markov_w2.weight": ("model.layers.45.markov_head.markov_w2.weight"),
        "mtp.2.norm.weight": "model.layers.45.norm.weight",
    }

    for source_name, expected_name in cases.items():
        assert DeepSeekV4DSparkMTP._remap_dspark_name(model, source_name) == expected_name


class _FakeDSparkLoadParam:
    def __init__(self, name, *, success=True):
        self.name = name
        self.success = success
        self.calls = 0

    def weight_loader(self, _param, _loaded_weight, *args, **kwargs):
        self.calls += 1
        return self.success


def _make_dspark_expert_load_test_model(
    *,
    expert_mapping,
    expert_params,
    expert_parallel,
    local_expert_ids,
):
    required_param_names = [
        "model.layers.43.main_proj.weight",
        "model.layers.43.main_norm.weight",
        "model.layers.44.self_attn.q_norm.weight",
        "model.layers.45.norm.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.45.markov_head.markov_w1.weight",
        "model.layers.45.markov_head.markov_w2.weight",
    ]
    params = {name: _FakeDSparkLoadParam(name) for name in required_param_names}
    params.update(expert_params)
    manager = (
        SimpleNamespace(get_local_expert_ids=lambda: list(local_expert_ids)) if local_expert_ids is not None else None
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=SimpleNamespace(expert_map_manager=manager),
        )
    )
    lifecycle = []
    draft_model = SimpleNamespace(
        num_dspark_layers=3,
        mtp_start_layer_idx=43,
        layers={"43": layer, "44": layer, "45": layer},
        get_expert_mapping=lambda: expert_mapping,
        finalize_mega_moe_weights=lambda: lifecycle.append("finalize"),
        install_quarot_hc_head_fold=lambda: lifecycle.append("install_hc_fold"),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        quant_config=None,
        model=draft_model,
        _dspark_checkpoint_path="",
        _dspark_expert_parallel_enabled=expert_parallel,
        _dspark_ep_weight_filter_enabled=False,
    )
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)
    base_weights = [
        ("mtp.0.main_proj.weight", torch.ones(1)),
        ("mtp.0.main_norm.weight", torch.ones(1)),
        ("mtp.1.attn.q_norm.weight", torch.ones(1)),
        ("mtp.2.norm.weight", torch.ones(1)),
        ("mtp.2.hc_head_fn", torch.ones(1)),
        ("mtp.2.hc_head_base", torch.ones(1)),
        ("mtp.2.hc_head_scale", torch.ones(1)),
        ("mtp.2.markov_head.markov_w1.weight", torch.ones(1)),
        ("mtp.2.markov_head.markov_w2.weight", torch.ones(1)),
    ]
    return model, params, base_weights, lifecycle


@pytest.mark.parametrize(
    ("expert_parallel", "local_expert_ids"),
    [(False, None), (True, {0})],
)
def test_dspark_load_weights_rejects_all_false_local_expert_mapping(
    monkeypatch,
    expert_parallel,
    local_expert_ids,
):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    mapped_name = "model.layers.43.mlp.experts.fused.weight"
    model, _, base_weights, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[("mlp.experts.fused", "mlp.experts.0.gate_proj", 0, "w1")],
        expert_params={mapped_name: _FakeDSparkLoadParam(mapped_name, success=False)},
        expert_parallel=expert_parallel,
        local_expert_ids=local_expert_ids,
    )

    with pytest.raises(ValueError, match="experts.fused.weight"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.ffn.experts.0.w1.weight", torch.ones(1)), *base_weights],
        )


def test_dspark_load_weights_allows_all_false_nonlocal_ep_expert(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    mapped_name = "model.layers.43.mlp.experts.fused.weight"
    model, params, base_weights, lifecycle = _make_dspark_expert_load_test_model(
        expert_mapping=[("mlp.experts.fused", "mlp.experts.7.gate_proj", 7, "w1")],
        expert_params={mapped_name: _FakeDSparkLoadParam(mapped_name, success=False)},
        expert_parallel=True,
        local_expert_ids={0},
    )

    loaded = DeepSeekV4DSparkMTP.load_weights(
        model,
        [("mtp.0.ffn.experts.7.w1.weight", torch.ones(1)), *base_weights],
    )

    assert mapped_name not in loaded
    assert params[mapped_name].calls == 1
    assert lifecycle == ["finalize", "install_hc_fold"]


def test_dspark_load_weights_accepts_later_successful_expert_replica(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    first_name = "model.layers.43.mlp.experts.fused_first.weight"
    second_name = "model.layers.43.mlp.experts.fused_second.weight"
    model, params, base_weights, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[
            ("mlp.experts.fused_first", "mlp.experts.0.gate_proj", 0, "w1"),
            ("mlp.experts.fused_second", "mlp.experts.0.gate_proj", 0, "w1"),
        ],
        expert_params={
            first_name: _FakeDSparkLoadParam(first_name, success=False),
            second_name: _FakeDSparkLoadParam(second_name, success=True),
        },
        expert_parallel=True,
        local_expert_ids={0},
    )

    loaded = DeepSeekV4DSparkMTP.load_weights(
        model,
        [("mtp.0.ffn.experts.0.w1.weight", torch.ones(1)), *base_weights],
    )

    assert second_name in loaded
    assert params[first_name].calls == 1
    assert params[second_name].calls == 1


def test_dspark_load_weights_rejects_false_local_expert_scale_alias(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    mapped_name = "model.layers.43.mlp.experts.fused.weight_scale"
    model, _, base_weights, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[("mlp.experts.fused", "mlp.experts.0.gate_proj", 0, "w1")],
        expert_params={mapped_name: _FakeDSparkLoadParam(mapped_name, success=False)},
        expert_parallel=True,
        local_expert_ids={0},
    )

    with pytest.raises(ValueError, match="experts.fused.weight_scale"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.ffn.experts.0.w1.scale", torch.ones(1)), *base_weights],
        )


def test_dspark_load_weights_fails_closed_when_ep_locality_is_unavailable(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    mapped_name = "model.layers.43.mlp.experts.fused.weight"
    model, _, base_weights, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[("mlp.experts.fused", "mlp.experts.0.gate_proj", 0, "w1")],
        expert_params={mapped_name: _FakeDSparkLoadParam(mapped_name, success=False)},
        expert_parallel=True,
        local_expert_ids=None,
    )

    with pytest.raises(ValueError, match="local expert assignment.*could not be determined"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.ffn.experts.0.w1.weight", torch.ones(1)), *base_weights],
        )


def test_dspark_load_weights_dequantized_flag_rejects_int8_stream(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    model, _, _, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[],
        expert_params={},
        expert_parallel=False,
        local_expert_ids=None,
    )
    model.config.dspark_mtp_dequantized_to_bf16 = True

    with pytest.raises(ValueError, match="declared FLOAT"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.main_proj.weight", torch.ones(1, dtype=torch.int8))],
        )


def test_dspark_load_weights_rejects_u8_quantized_stream(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    model, _, _, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[],
        expert_params={},
        expert_parallel=False,
        local_expert_ids=None,
    )
    description = {
        "mtp.0.main_proj.weight": "W8A8_DYNAMIC",
        "mtp.0.main_proj.weight_scale": "W8A8_DYNAMIC",
        "mtp.0.main_proj.weight_offset": "W8A8_DYNAMIC",
    }
    model.quant_config = SimpleNamespace(
        quant_description=description,
        source_quant_description=description,
    )

    with pytest.raises(ValueError, match="expected torch.int8"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.main_proj.weight", torch.ones(1, dtype=torch.uint8))],
        )


def test_dspark_load_weights_rejects_main_proj_shape_from_stream(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    model, _, _, _ = _make_dspark_expert_load_test_model(
        expert_mapping=[],
        expert_params={},
        expert_parallel=False,
        local_expert_ids=None,
    )
    model.config.hidden_size = 2
    model.config.dspark_target_layer_ids = [40, 41]

    with pytest.raises(ValueError, match=r"physical shape \(2, 3\); expected \(2, 4\)"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.main_proj.weight", torch.ones((2, 3)))],
        )


def test_dspark_ep_weight_filter_exempts_only_nonlocal_base_weight(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)
    mapped_name = "model.layers.43.mlp.experts.fused.weight"
    model, _, base_weights, lifecycle = _make_dspark_expert_load_test_model(
        expert_mapping=[("mlp.experts.fused", "mlp.experts.7.gate_proj", 7, "w1")],
        expert_params={mapped_name: _FakeDSparkLoadParam(mapped_name, success=False)},
        expert_parallel=True,
        local_expert_ids={0},
    )
    description = {
        "mtp.0.ffn.experts.7.w1.weight": "W8A8_DYNAMIC",
        "mtp.0.ffn.experts.7.w1.weight_scale": "W8A8_DYNAMIC",
        "mtp.0.ffn.experts.7.w1.weight_offset": "W8A8_DYNAMIC",
    }
    model.quant_config = SimpleNamespace(
        quant_description=description,
        source_quant_description=description,
    )
    model._dspark_ep_weight_filter_enabled = True

    loaded = DeepSeekV4DSparkMTP.load_weights(
        model,
        [
            ("mtp.0.ffn.experts.7.w1.weight_scale", torch.ones(1)),
            ("mtp.0.ffn.experts.7.w1.weight_offset", torch.zeros(1)),
            *base_weights,
        ],
    )

    assert mapped_name not in loaded
    assert lifecycle == ["finalize", "install_hc_fold"]


def test_dspark_load_weights_rejects_unmatched_mtp_params(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    model.named_parameters = lambda: iter(())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    with pytest.raises(ValueError, match="model\\.layers\\.43\\.self_attn\\.q_norm\\.weight"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.attn.q_norm.weight", torch.ones(1))],
        )


def test_dspark_load_weights_rejects_tensor_missing_from_modelslim_description(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    class FakeParam:
        @staticmethod
        def weight_loader(_param, _loaded_weight):
            return None

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        quant_config=SimpleNamespace(
            quant_description={
                "mtp.0.main_proj.weight": "FLOAT",
                "mtp.0.main_norm.weight": "FLOAT",
            }
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    params = {"model.layers.43.main_proj.weight": FakeParam()}
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    with pytest.raises(ValueError, match="mtp.0.main_norm.weight"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.main_proj.weight", torch.ones(1))],
        )


def test_dspark_load_weights_rejects_missing_weight_only_schema_companions(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    class FakeParam:
        @staticmethod
        def weight_loader(_param, _loaded_weight):
            return None

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        quant_config=SimpleNamespace(
            quant_description={
                "group_size": 0,
                "mtp.0.main_proj.weight": "W8A8_DYNAMIC",
            }
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    params = {"model.layers.43.main_proj.weight": FakeParam()}
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    with pytest.raises(ValueError, match="physical companion tensors"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [("mtp.0.main_proj.weight", torch.ones(1, dtype=torch.int8))],
        )


def test_dspark_load_weights_rejects_missing_direct_description_base_weight(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    class FakeParam:
        @staticmethod
        def weight_loader(_param, _loaded_weight):
            return None

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        quant_config=SimpleNamespace(
            quant_description={
                "group_size": 0,
                "model.layers.43.self_attn.wq_a.weight": "W8A8_DYNAMIC",
            }
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    params = {
        "model.layers.43.self_attn.wq_a.weight": FakeParam(),
        "model.layers.43.self_attn.wq_a.weight_scale": FakeParam(),
        "model.layers.43.self_attn.wq_a.weight_offset": FakeParam(),
    }
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    with pytest.raises(ValueError, match="mtp.0.attn.wq_a.weight"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [
                ("mtp.0.attn.wq_a.weight_scale", torch.ones(1)),
                ("mtp.0.attn.wq_a.weight_offset", torch.zeros(1)),
            ],
        )


def test_dspark_load_weights_rejects_missing_direct_description_non_weight_tensor(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    class FakeParam:
        @staticmethod
        def weight_loader(_param, _loaded_weight):
            return None

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        quant_config=SimpleNamespace(
            quant_description={
                "group_size": 0,
                "model.layers.43.self_attn.wq_a.weight": "W8A8_DYNAMIC",
                "model.layers.43.self_attn.attn_sink": "FLOAT",
            }
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: None,
        ),
    )
    params = {
        "model.layers.43.self_attn.wq_a.weight": FakeParam(),
        "model.layers.43.self_attn.wq_a.weight_scale": FakeParam(),
        "model.layers.43.self_attn.wq_a.weight_offset": FakeParam(),
        "model.layers.43.self_attn.attn_sink": FakeParam(),
    }
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    with pytest.raises(ValueError, match="mtp.0.attn.attn_sink"):
        DeepSeekV4DSparkMTP.load_weights(
            model,
            [
                ("mtp.0.attn.wq_a.weight", torch.ones(1, dtype=torch.int8)),
                ("mtp.0.attn.wq_a.weight_scale", torch.ones(1)),
                ("mtp.0.attn.wq_a.weight_offset", torch.zeros(1)),
            ],
        )


def test_dspark_load_weights_stacks_shared_expert_gate_up_and_ignores_logical_quant_metadata(monkeypatch):
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model_module, "get_tensor_model_parallel_rank", lambda: 0)

    calls = []
    lifecycle = []

    class FakeParam:
        def __init__(self, name):
            self.name = name

        def weight_loader(self, param, loaded_weight, *args, **kwargs):
            calls.append((param.name, loaded_weight.clone(), args, kwargs))
            return True

    param_names = [
        "model.layers.43.main_proj.weight",
        "model.layers.43.main_norm.weight",
        "model.layers.44.mlp.shared_experts.gate_up_proj.weight",
        "model.layers.45.norm.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.45.markov_head.markov_w1.weight",
        "model.layers.45.markov_head.markov_w2.weight",
    ]
    params = {name: FakeParam(name) for name in param_names}
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_attention_heads=8,
            expert_dtype="fp4",
        ),
        quant_config=SimpleNamespace(
            quant_description={
                "mtp.0.attn.indexer.quant_type": "INT8_DYNAMIC",
                "mtp.0.attn.indexer.wq_b_weight": "INT8_DYNAMIC",
                "model.mtp.1.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.mtp.1.self_attn.indexer.wq_b_weight": "INT8_DYNAMIC",
                "model.layers.45.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.layers.45.self_attn.indexer.wq_b_weight": "INT8_DYNAMIC",
            }
        ),
        model=SimpleNamespace(
            num_dspark_layers=3,
            get_expert_mapping=lambda: [],
            finalize_mega_moe_weights=lambda: lifecycle.append("finalize"),
            install_quarot_hc_head_fold=lambda: lifecycle.append("install_hc_fold"),
        ),
    )
    model.named_parameters = lambda: iter(params.items())
    model._remap_dspark_name = DeepSeekV4DSparkMTP._remap_dspark_name.__get__(model)

    loaded = DeepSeekV4DSparkMTP.load_weights(
        model,
        [
            ("mtp.0.main_proj.weight", torch.ones(1)),
            ("mtp.0.main_norm.weight", torch.ones(1) * 2),
            ("mtp.1.ffn.shared_experts.w1.weight", torch.ones(1) * 3),
            ("mtp.1.ffn.shared_experts.w3.weight", torch.ones(1) * 4),
            ("mtp.2.norm.weight", torch.ones(1) * 5),
            ("mtp.2.hc_head_fn", torch.ones(1) * 6),
            ("mtp.2.hc_head_base", torch.ones(1) * 7),
            ("mtp.2.hc_head_scale", torch.ones(1) * 8),
            ("mtp.2.markov_head.markov_w1.weight", torch.ones(1) * 9),
            ("mtp.2.markov_head.markov_w2.weight", torch.ones(1) * 10),
        ],
    )

    assert "model.layers.44.mlp.shared_experts.gate_up_proj.weight" in loaded
    shared_calls = [call for call in calls if call[0] == "model.layers.44.mlp.shared_experts.gate_up_proj.weight"]
    assert [call[2] for call in shared_calls] == [(0,), (1,)]
    torch.testing.assert_close(shared_calls[0][1], torch.ones(1) * 3)
    torch.testing.assert_close(shared_calls[1][1], torch.ones(1) * 4)
    assert lifecycle == ["finalize", "install_hc_fold"]


def test_dspark_model_declares_target_shared_embedding_and_lm_head():
    assert DeepSeekV4DSparkMTP.has_own_embed_tokens is False
    assert DeepSeekV4DSparkMTP.has_own_lm_head is False


def test_draft_model_without_own_lm_head_shares_target_lm_head():
    target_lm_head = object()
    draft_lm_head = object()
    proposer = SimpleNamespace(
        method="mtp",
        model=SimpleNamespace(
            has_own_lm_head=False,
            lm_head=draft_lm_head,
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=False),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(
                    has_full_cudagraphs=lambda: False,
                ),
            ),
        ),
        use_cuda_graph=False,
    )

    AscendSpecDecodeBaseProposer._maybe_share_lm_head(
        proposer,
        SimpleNamespace(lm_head=target_lm_head),
    )

    assert proposer.model.lm_head is target_lm_head


def test_dspark_exposes_draft_kv_cache_layer_names():
    def make_layer(prefix: str) -> SimpleNamespace:
        return SimpleNamespace(
            self_attn=SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=prefix),
                ),
            ),
        )

    model = SimpleNamespace(
        layers={
            "61": make_layer("model.layers.61.self_attn.swa_cache"),
            "62": make_layer("model.layers.62.self_attn.swa_cache"),
        }
    )
    model.get_draft_kv_cache_layer_names = DeepseekV4DSparkModel.get_draft_kv_cache_layer_names.__get__(model)
    wrapper = SimpleNamespace(model=model)

    expected = [
        "model.layers.61.self_attn.swa_cache",
        "model.layers.62.self_attn.swa_cache",
    ]
    assert DeepseekV4DSparkModel.get_draft_kv_cache_layer_names(model) == expected
    assert DeepSeekV4DSparkMTP.get_draft_kv_cache_layer_names(wrapper) == expected


def test_dspark_precompute_context_kv_passes_layer_slot_mappings(monkeypatch):
    calls = []

    def make_layer(name: str) -> SimpleNamespace:
        def precompute_context_kv(main_x, positions, request_slots=None, context_slot_mapping=None):
            calls.append((name, main_x, positions, request_slots, context_slot_mapping))

        return SimpleNamespace(self_attn=SimpleNamespace(precompute_context_kv=precompute_context_kv))

    monkeypatch.setattr(dspark_model_module, "_linear_output", lambda _proj, hidden_states: hidden_states + 1)
    context_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    positions = torch.tensor([4, 5, 6], dtype=torch.int32)
    request_slots = torch.tensor([1, 1, 1], dtype=torch.int32)
    layer_slot_mappings = [
        torch.tensor([10, 11, 12], dtype=torch.int32),
        torch.tensor([20, 21, 22], dtype=torch.int32),
    ]
    model = SimpleNamespace(
        main_proj=object(),
        main_norm=lambda tensor: tensor * 2,
        layers={
            "61": make_layer("61"),
            "62": make_layer("62"),
        },
    )

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        context_slot_mapping=layer_slot_mappings,
        context_request_slots=request_slots,
    )

    assert [call[0] for call in calls] == ["61", "62"]
    for idx, call in enumerate(calls):
        _, main_x, call_positions, call_request_slots, call_slot_mapping = call
        torch.testing.assert_close(main_x, (context_states + 1) * 2)
        assert call_positions is positions
        assert call_request_slots is request_slots
        assert call_slot_mapping is layer_slot_mappings[idx]


def test_dspark_precompute_context_kv_applies_unrotated_basis_at_call_site(monkeypatch):
    projected_inputs = []
    attention_inputs = []

    def fake_linear_output(_proj, hidden_states):
        projected_inputs.append(hidden_states)
        return hidden_states

    def precompute_context_kv(main_x, _positions, request_slots=None, context_slot_mapping=None):
        del request_slots, context_slot_mapping
        attention_inputs.append(main_x)

    monkeypatch.setattr(dspark_model_module, "_linear_output", fake_linear_output)
    context_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    model = SimpleNamespace(
        main_proj=object(),
        main_norm=lambda tensor: tensor,
        hidden_size=2,
        _dspark_quarot_rotation=torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        _dspark_quarot_draft_basis="unrotated",
        layers={
            "43": SimpleNamespace(
                self_attn=SimpleNamespace(precompute_context_kv=precompute_context_kv),
            )
        },
    )

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        context_positions=torch.tensor([0]),
    )

    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
    torch.testing.assert_close(projected_inputs[0], expected)
    torch.testing.assert_close(attention_inputs[0], expected)


def test_dspark_precompute_context_kv_bridges_canonical_projection_to_rotated_decoder(monkeypatch):
    projected_inputs = []
    attention_inputs = []

    def fake_linear_output(_proj, hidden_states):
        projected_inputs.append(hidden_states)
        return hidden_states[:, :2]

    def precompute_context_kv(main_x, _positions, request_slots=None, context_slot_mapping=None):
        del request_slots, context_slot_mapping
        attention_inputs.append(main_x)

    monkeypatch.setattr(dspark_model_module, "_linear_output", fake_linear_output)
    context_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    model = SimpleNamespace(
        main_proj=object(),
        main_norm=lambda tensor: tensor + torch.tensor([[1.0, 2.0]]),
        hidden_size=2,
        _dspark_quarot_rotation=torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        _dspark_quarot_draft_basis="rotated_decoder",
        layers={
            "43": SimpleNamespace(
                self_attn=SimpleNamespace(precompute_context_kv=precompute_context_kv),
            )
        },
    )

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        context_positions=torch.tensor([0]),
    )

    torch.testing.assert_close(projected_inputs[0], torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))
    # main_norm([-2, 1]) -> [-1, 3], then the decoder-side boundary applies Q.
    torch.testing.assert_close(attention_inputs[0], torch.tensor([[3.0, 1.0]]))


def test_dspark_precompute_context_kv_selects_prefix_mapped_slot_mappings(monkeypatch):
    calls = []

    def make_layer(name: str, prefix: str) -> SimpleNamespace:
        def precompute_context_kv(main_x, positions, request_slots=None, context_slot_mapping=None):
            calls.append((name, main_x, positions, request_slots, context_slot_mapping))

        return SimpleNamespace(
            self_attn=SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=prefix),
                ),
                precompute_context_kv=precompute_context_kv,
            )
        )

    monkeypatch.setattr(dspark_model_module, "_linear_output", lambda _proj, hidden_states: hidden_states + 1)
    context_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    positions = torch.tensor([4, 5, 6], dtype=torch.int32)
    request_slots = torch.tensor([1, 1, 1], dtype=torch.int32)
    slot_mapping_61 = torch.tensor([10, 11, 12], dtype=torch.int32)
    slot_mapping_62 = torch.tensor([20, 21, 22], dtype=torch.int32)
    model = SimpleNamespace(
        main_proj=object(),
        main_norm=lambda tensor: tensor * 2,
        layers={
            "61": make_layer("61", "model.layers.61.self_attn.swa_cache"),
            "62": make_layer("62", "model.layers.62.self_attn.swa_cache"),
        },
    )

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        context_slot_mapping={
            "model.layers.61.self_attn.swa_cache": slot_mapping_61,
            "model.layers.62.self_attn.swa_cache": slot_mapping_62,
        },
        context_request_slots=request_slots,
    )

    assert calls[0][4] is slot_mapping_61
    assert calls[1][4] is slot_mapping_62


def test_dspark_precompute_context_kv_skips_private_cache_for_standard_dsa(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    calls = []
    shared_kv = torch.ones(2, 1, 4)

    def fail_expand(_shared_kv):
        raise AssertionError("private cache should not be materialized on the standard DSA path")

    attention = SimpleNamespace(
        _project_shared_kv=lambda _main_x, _positions: shared_kv,
        _expand_private_kv=fail_expand,
        _store_standard_swa_kv=lambda kv, mapping: calls.append((kv, mapping)),
    )
    positions = torch.tensor([4, 5], dtype=torch.int32)
    slot_mapping = torch.tensor([40, 41], dtype=torch.int32)

    DeepseekV4DSparkAttention.precompute_context_kv(
        attention,
        torch.zeros(2, 4),
        positions,
        request_slots=torch.tensor([0, 0], dtype=torch.int32),
        context_slot_mapping=slot_mapping,
    )

    assert calls == [(shared_kv, slot_mapping)]


def test_dspark_precompute_context_kv_keeps_private_cache_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", "1")

    shared_kv = torch.ones(2, 1, 4)
    private_k = torch.full((2, 1, 4), 2.0)
    private_v = torch.full((2, 1, 4), 3.0)
    expand_calls = []

    def expand_private_kv(kv):
        expand_calls.append(kv)
        return private_k, private_v

    attention = SimpleNamespace(
        _project_shared_kv=lambda _main_x, _positions: shared_kv,
        _expand_private_kv=expand_private_kv,
        _store_standard_swa_kv=lambda *_args: None,
        _ensure_dspark_cache=lambda *_args: None,
        _dspark_cache_capacity=8,
        _dspark_max_request_slots=2,
        _dspark_k_cache=torch.zeros(2, 8, 1, 4),
        _dspark_v_cache=torch.zeros(2, 8, 1, 4),
        _dspark_cache_positions=torch.zeros(2, 8, dtype=torch.int32),
        _dspark_cache_valid=torch.zeros(2, 8, dtype=torch.bool),
    )
    positions = torch.tensor([4, 5], dtype=torch.int32)
    request_slots = torch.tensor([1, 1], dtype=torch.int32)

    DeepseekV4DSparkAttention.precompute_context_kv(
        attention,
        torch.zeros(2, 4),
        positions,
        request_slots=request_slots,
        context_slot_mapping=torch.tensor([40, 41], dtype=torch.int32),
    )

    assert expand_calls == [shared_kv]
    torch.testing.assert_close(attention._dspark_k_cache[1, 4:6], private_k)
    torch.testing.assert_close(attention._dspark_v_cache[1, 4:6], private_v)
    torch.testing.assert_close(attention._dspark_cache_positions[1, 4:6], positions)
    torch.testing.assert_close(attention._dspark_cache_valid[1, 4:6], torch.ones(2, dtype=torch.bool))


def test_dspark_forward_passes_query_slot_mapping_to_layers():
    calls = []

    class FakeLayer:
        def __call__(
            self,
            *,
            positions,
            hidden_states,
            residual=None,
            post_mix=None,
            res_mix=None,
            input_ids,
            request_slots=None,
            slot_mapping=None,
            block_table=None,
        ):
            del residual, post_mix, res_mix
            calls.append((positions, hidden_states, input_ids, request_slots, slot_mapping, block_table))
            return hidden_states + 1

    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([10, 11], dtype=torch.int32)
    inputs_embeds = torch.ones(2, 3)
    request_slots = torch.tensor([4, 4], dtype=torch.int32)
    slot_mapping = torch.tensor([80, 81], dtype=torch.int32)
    model = SimpleNamespace(
        embed_tokens=None,
        hc_mult=2,
        layers={
            "61": FakeLayer(),
            "62": FakeLayer(),
        },
        compute_head_hidden=lambda hidden_states, *args: hidden_states,
    )

    output = DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        request_slots=request_slots,
        slot_mapping=slot_mapping,
    )

    assert len(calls) == 2
    for call in calls:
        call_positions, _, call_input_ids, call_request_slots, call_slot_mapping, call_block_table = call
        assert call_positions is positions
        assert call_input_ids is input_ids
        assert call_request_slots is request_slots
        assert call_slot_mapping is slot_mapping
        assert call_block_table is None
    torch.testing.assert_close(output, inputs_embeds.unsqueeze(-2).repeat(1, 2, 1) + 2)


def test_dspark_forward_carries_mhc_state_between_layers_and_head():
    calls = []
    head_call = {}

    class FakeLayer:
        def __init__(self, value: float):
            self.value = value

        def __call__(
            self,
            *,
            positions,
            hidden_states,
            residual=None,
            post_mix=None,
            res_mix=None,
            input_ids,
            request_slots=None,
            slot_mapping=None,
            block_table=None,
        ):
            del positions, input_ids, request_slots, slot_mapping, block_table
            calls.append((residual, post_mix, res_mix))
            state = hidden_states + self.value
            if residual is None:
                return state, state + 10, state + 20, state + 30
            return state, residual + self.value, post_mix + self.value, res_mix + self.value

    def fake_compute_head(hidden_states, residual=None, post_mix=None, res_mix=None):
        head_call["hidden_states"] = hidden_states
        head_call["residual"] = residual
        head_call["post_mix"] = post_mix
        head_call["res_mix"] = res_mix
        return hidden_states

    input_ids = torch.tensor([1], dtype=torch.int64)
    positions = torch.tensor([10], dtype=torch.int32)
    inputs_embeds = torch.ones(1, 3)
    model = SimpleNamespace(
        embed_tokens=None,
        hc_mult=2,
        layers={
            "61": FakeLayer(1),
            "62": FakeLayer(2),
        },
        compute_head_hidden=fake_compute_head,
    )

    output = DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
    )

    expanded = inputs_embeds.unsqueeze(-2).repeat(1, 2, 1)
    first_hidden = expanded + 1
    second_hidden = first_hidden + 2
    assert calls[0] == (None, None, None)
    torch.testing.assert_close(calls[1][0], first_hidden + 10)
    torch.testing.assert_close(calls[1][1], first_hidden + 20)
    torch.testing.assert_close(calls[1][2], first_hidden + 30)
    torch.testing.assert_close(head_call["hidden_states"], second_hidden)
    torch.testing.assert_close(head_call["residual"], first_hidden + 12)
    torch.testing.assert_close(head_call["post_mix"], first_hidden + 22)
    torch.testing.assert_close(head_call["res_mix"], first_hidden + 32)
    torch.testing.assert_close(output, second_hidden)


def test_dspark_decoder_layer_uses_upstream_style_mhc_state_flow():
    layer = object.__new__(DeepseekV4DSparkDecoderLayer)
    layer.hc_attn_fn = torch.tensor(1.0)
    layer.hc_attn_scale = torch.tensor(2.0)
    layer.hc_attn_base = torch.tensor(3.0)
    layer.hc_ffn_fn = torch.tensor(4.0)
    layer.hc_ffn_scale = torch.tensor(5.0)
    layer.hc_ffn_base = torch.tensor(6.0)
    layer.input_layernorm = lambda x: x + 10
    layer.post_attention_layernorm = lambda x: x + 100
    layer.self_attn = lambda positions, hidden_states, _kv_cache, **kwargs: hidden_states + 1000
    layer.mlp = lambda hidden_states, input_ids: hidden_states + 10000
    calls: list[tuple[Any, ...]] = []

    def fake_pre(hidden_states, hc_fn, hc_scale, hc_base):
        calls.append(("pre", hidden_states.clone(), hc_fn, hc_scale, hc_base))
        return hidden_states + 1, hidden_states + 2, hidden_states + 3

    def fake_fused(hidden_states, residual, post_mix, res_mix, hc_fn, hc_scale, hc_base):
        calls.append(
            (
                "fused",
                hidden_states.clone(),
                residual.clone(),
                post_mix.clone(),
                res_mix.clone(),
                hc_fn,
                hc_scale,
                hc_base,
            )
        )
        return residual + 4, post_mix + 5, res_mix + 6, hidden_states + 7

    layer._mhc_pre = fake_pre
    layer._mhc_fused_post_pre = fake_fused
    hidden_states = torch.zeros(1, 2, 3)
    positions = torch.tensor([0], dtype=torch.int32)
    input_ids = torch.tensor([1], dtype=torch.int64)

    out, residual, post_mix, res_mix = DeepseekV4DSparkDecoderLayer.forward(
        layer,
        positions=positions,
        hidden_states=hidden_states,
        input_ids=input_ids,
    )

    assert [call[0] for call in calls] == ["pre", "fused"]
    torch.testing.assert_close(calls[1][1], hidden_states + 1 + 10 + 1000)
    torch.testing.assert_close(calls[1][2], hidden_states)
    torch.testing.assert_close(calls[1][3], hidden_states + 2)
    torch.testing.assert_close(calls[1][4], hidden_states + 3)
    torch.testing.assert_close(out, hidden_states + 1 + 10 + 1000 + 7 + 100 + 10000)
    torch.testing.assert_close(residual, hidden_states + 4)
    torch.testing.assert_close(post_mix, hidden_states + 7)
    torch.testing.assert_close(res_mix, hidden_states + 9)


def test_dspark_attention_rebuilds_standard_query_slot_mapping(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.block_size = 5
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(block_size=32),
    )
    positions = torch.tensor([26, 27, 28, 29, 30, 0], dtype=torch.int32)
    stale_slot_mapping = torch.tensor([90, 91, 92, 93, 94, -1], dtype=torch.int32)
    block_table = torch.tensor([[3, 17, 0, 0]], dtype=torch.int32)

    slot_mapping = DeepseekV4DSparkAttention._standard_query_slot_mapping_from_block_table(
        attn,
        positions,
        stale_slot_mapping,
        block_table,
    )

    torch.testing.assert_close(
        slot_mapping,
        torch.tensor([122, 123, 124, 125, 126, -1], dtype=torch.int32),
    )


def test_dspark_attention_rebuilds_standard_query_slot_mapping_with_token_to_req(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.block_size = 2
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(block_size=4),
    )
    positions = torch.tensor([8, 20, 9, 21], dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    token_to_req_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 4, 5, 6, 7, 8],
            [10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )

    def fail_nonzero(*args, **kwargs):
        raise AssertionError("torch.nonzero is not ACLGraph-capture safe")

    with monkeypatch.context() as m:
        m.setattr(torch, "nonzero", fail_nonzero)
        rebuilt = DeepseekV4DSparkAttention._standard_query_slot_mapping_from_block_table(
            attn,
            positions,
            slot_mapping,
            block_table,
            token_to_req_indices,
        )

    torch.testing.assert_close(
        rebuilt,
        torch.tensor([20, 60, 21, 61], dtype=torch.int32),
    )


def test_dspark_forward_selects_prefix_mapped_slot_mapping_and_block_table():
    calls = []

    class FakeLayer:
        def __init__(self, prefix: str):
            self.self_attn = SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    swa_cache_layer=SimpleNamespace(prefix=prefix),
                )
            )

        def __call__(
            self,
            *,
            positions,
            hidden_states,
            residual=None,
            post_mix=None,
            res_mix=None,
            input_ids,
            request_slots=None,
            slot_mapping=None,
            block_table=None,
        ):
            del residual, post_mix, res_mix
            calls.append((positions, hidden_states, input_ids, request_slots, slot_mapping, block_table))
            return hidden_states + 1

    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([10, 11], dtype=torch.int32)
    inputs_embeds = torch.ones(2, 3)
    request_slots = torch.tensor([4, 4], dtype=torch.int32)
    slot_mapping_61 = torch.tensor([80, 81], dtype=torch.int32)
    slot_mapping_62 = torch.tensor([180, 181], dtype=torch.int32)
    block_table_61 = torch.tensor([[1, 2]], dtype=torch.int32)
    block_table_62 = torch.tensor([[11, 12]], dtype=torch.int32)
    model = SimpleNamespace(
        embed_tokens=None,
        hc_mult=2,
        layers={
            "61": FakeLayer("model.layers.61.self_attn.swa_cache"),
            "62": FakeLayer("model.layers.62.self_attn.swa_cache"),
        },
        compute_head_hidden=lambda hidden_states, *args: hidden_states,
    )

    output = DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        request_slots=request_slots,
        slot_mapping={
            "model.layers.61.self_attn.swa_cache": slot_mapping_61,
            "model.layers.62.self_attn.swa_cache": slot_mapping_62,
        },
        block_table={
            "model.layers.61.self_attn.swa_cache": block_table_61,
            "model.layers.62.self_attn.swa_cache": block_table_62,
        },
    )

    assert len(calls) == 2
    assert calls[0][4] is slot_mapping_61
    assert calls[0][5] is block_table_61
    assert calls[1][4] is slot_mapping_62
    assert calls[1][5] is block_table_62
    torch.testing.assert_close(output, inputs_embeds.unsqueeze(-2).repeat(1, 2, 1) + 2)


def test_dspark_store_standard_swa_kv_uses_dsa_slot_mapping(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    calls: list[tuple[Any, ...]] = []

    def fake_format(slot_mapping, block_size):
        calls.append(("format", slot_mapping.clone(), block_size))
        return torch.stack([slot_mapping // block_size, slot_mapping % block_size], dim=-1)

    def fake_scatter(cache, shared_kv, slot_mapping):
        calls.append(("scatter", cache, shared_kv.clone(), slot_mapping.clone()))

    monkeypatch.setattr(device_op_module.DeviceOperator, "format_dsa_slot_mapping", staticmethod(fake_format))
    monkeypatch.setattr(device_op_module.DeviceOperator, "dsa_kv_compress_scatter", staticmethod(fake_scatter))
    cache = torch.zeros(4, 8, 1, 3)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                kv_cache=cache,
                block_size=8,
            )
        )
    )
    shared_kv = torch.arange(6, dtype=torch.float32).view(2, 1, 3)
    slot_mapping = torch.tensor([9, 18], dtype=torch.int64)

    DeepseekV4DSparkAttention._store_standard_swa_kv(attn, shared_kv, slot_mapping)

    assert calls[0][0] == "format"
    torch.testing.assert_close(calls[0][1], torch.tensor([9, 18], dtype=torch.int32))
    assert calls[0][2] == 8
    assert calls[1][0] == "scatter"
    assert calls[1][1] is cache
    torch.testing.assert_close(calls[1][2], shared_kv)
    torch.testing.assert_close(calls[1][3], torch.tensor([[1, 1], [2, 2]], dtype=torch.int32))


def test_dspark_store_standard_swa_kv_preserves_explicit_slot_semantics(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)

    def fake_format(slot_mapping, block_size):
        return torch.stack([slot_mapping // block_size, slot_mapping % block_size], dim=-1)

    def fake_scatter(cache, shared_kv, slot_mapping):
        for token_idx, (block_idx, block_offset) in enumerate(slot_mapping.tolist()):
            cache[block_idx, block_offset].copy_(shared_kv[token_idx])

    monkeypatch.setattr(device_op_module.DeviceOperator, "format_dsa_slot_mapping", staticmethod(fake_format))
    monkeypatch.setattr(device_op_module.DeviceOperator, "dsa_kv_compress_scatter", staticmethod(fake_scatter))

    block_size = 4
    cache = torch.zeros(5, block_size, 1, 2)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                kv_cache=cache,
                block_size=block_size,
            )
        )
    )
    shared_kv = torch.arange(12, dtype=torch.float32).view(6, 1, 2)
    slot_mapping = torch.tensor([3, 8, 9, 1, 14, 4], dtype=torch.int32)

    DeepseekV4DSparkAttention._store_standard_swa_kv(attn, shared_kv, slot_mapping)

    expected = torch.zeros_like(cache)
    for token_idx, slot in enumerate(slot_mapping.tolist()):
        expected[slot // block_size, slot % block_size].copy_(shared_kv[token_idx])
    torch.testing.assert_close(cache, expected)


def test_dspark_store_standard_swa_kv_capture_slices_padding(monkeypatch):
    from vllm_ascend.device import device_op as device_op_module

    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setattr(dspark_model_module.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.NONE,
            num_actual_tokens=2,
        ),
    )
    monkeypatch.setattr(
        dspark_model_module,
        "_sync_npu_device_for_standard_pta",
        lambda tensor: (_ for _ in ()).throw(AssertionError("capture path must not synchronize")),
    )
    calls: list[tuple[Any, ...]] = []

    def fake_format(slot_mapping, block_size):
        calls.append(("format", slot_mapping.clone(), block_size))
        return torch.stack([slot_mapping // block_size, slot_mapping % block_size], dim=-1)

    def fake_scatter(cache, shared_kv, slot_mapping):
        calls.append(("scatter", cache, shared_kv.clone(), slot_mapping.clone()))

    monkeypatch.setattr(device_op_module.DeviceOperator, "format_dsa_slot_mapping", staticmethod(fake_format))
    monkeypatch.setattr(device_op_module.DeviceOperator, "dsa_kv_compress_scatter", staticmethod(fake_scatter))
    cache = torch.zeros(4, 8, 1, 3)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                kv_cache=cache,
                block_size=8,
            )
        )
    )
    shared_kv = torch.arange(12, dtype=torch.float32).view(4, 1, 3)
    slot_mapping = torch.tensor([9, 18, -1, -1], dtype=torch.int32)

    DeepseekV4DSparkAttention._store_standard_swa_kv(attn, shared_kv, slot_mapping)

    assert calls[0][0] == "format"
    torch.testing.assert_close(calls[0][1], torch.tensor([9, 18], dtype=torch.int32))
    assert calls[1][0] == "scatter"
    torch.testing.assert_close(calls[1][2], shared_kv[:2])
    torch.testing.assert_close(calls[1][3], torch.tensor([[1, 1], [2, 2]], dtype=torch.int32))


def test_dspark_standard_attention_can_fallback_to_pta(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", "1")

    expected = torch.ones(2, 4, 8)
    calls = []

    def fake_sas(*args, **kwargs):
        raise AssertionError("SAS fast path must not run when disabled")

    def fake_pta(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 2
    attn.window_size = 6
    attn.scale = 0.5

    out = DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attn,
        q=torch.zeros(2, 4, 8),
        positions=torch.tensor([6, 7], dtype=torch.int32),
        slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert out is expected
    assert len(calls) == 1


def test_dspark_standard_attention_uses_sas_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", raising=False)

    expected = torch.ones(2, 4, 8)
    dspark_swa_indices = torch.full((2, 1, 8), -1, dtype=torch.int32)
    dspark_swa_lens = torch.tensor([2, 2], dtype=torch.int32)
    sas_metadata = torch.tensor([123], dtype=torch.int32)
    calls = []

    def fake_sas(*args, **kwargs):
        calls.append(("sas", args, kwargs))
        return expected

    def fake_pta(*args, **kwargs):
        raise AssertionError("PTA should not run when SAS returns an output")

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.FULL,
            draft_attn_metadatas=[
                {
                    "layers.0.self_attn": SimpleNamespace(
                        decode=SimpleNamespace(
                            dspark_swa_indices=dspark_swa_indices,
                            dspark_swa_lens=dspark_swa_lens,
                            sas_metadata=sas_metadata,
                        )
                    )
                }
            ],
        ),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
            prefix="layers.0.self_attn",
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 2
    attn.window_size = 6
    attn.scale = 0.5

    out = DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attn,
        q=torch.zeros(2, 4, 8),
        positions=torch.tensor([6, 7], dtype=torch.int32),
        slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert out is expected
    assert calls[0][0] == "sas"
    assert calls[0][2]["dspark_swa_indices"] is dspark_swa_indices
    assert calls[0][2]["dspark_swa_lens"] is dspark_swa_lens
    assert calls[0][2]["sas_metadata"] is sas_metadata
    assert calls[0][2]["skip_scheduling_guard"] is True
    assert calls[0][2]["raise_on_error"] is True


def test_dspark_standard_attention_passes_actual_query_tokens_to_sas(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", raising=False)

    expected = torch.ones(8, 4, 8)
    calls = []

    def fake_sas(*args, **kwargs):
        calls.append(("sas", args, kwargs))
        return expected

    def fake_pta(*args, **kwargs):
        raise AssertionError("PTA should not run when SAS returns an output")

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.NONE,
            num_actual_tokens=5,
        ),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 5
    attn.window_size = 6
    attn.scale = 0.5

    out = DeepseekV4DSparkAttention._run_standard_dspark_attention(
        attn,
        q=torch.zeros(8, 4, 8),
        positions=torch.arange(8, dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        dspark_query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert out is expected
    assert calls[0][2]["num_query_tokens"] == 5
    assert calls[0][2]["skip_scheduling_guard"] is False


def test_dspark_standard_attention_does_not_fallback_to_pta_during_capture(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", raising=False)

    def fake_sas(*args, **kwargs):
        return None

    def fake_pta(*args, **kwargs):
        raise AssertionError("PTA fallback must not run during graph capture")

    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache_sas", fake_sas)
    monkeypatch.setattr(dspark_model_module, "dspark_attention_from_standard_cache", fake_pta)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.FULL),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros(4, 16, 1, 8),
            block_size=16,
        )
    )
    attn.attn_sink = torch.zeros(4)
    attn.n_local_heads = 4
    attn.block_size = 2
    attn.window_size = 6
    attn.scale = 0.5

    with pytest.raises(RuntimeError, match="requires prebuilt SWA indices"):
        DeepseekV4DSparkAttention._run_standard_dspark_attention(
            attn,
            q=torch.zeros(2, 4, 8),
            positions=torch.tensor([6, 7], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
        )


def test_dspark_standard_attention_requires_standard_cache_metadata_during_capture(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE", raising=False)
    monkeypatch.setattr(
        dspark_model_module,
        "_maybe_get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=dspark_model_module.CUDAGraphMode.FULL),
    )

    attn = object.__new__(DeepseekV4DSparkAttention)
    attn.dsa_attn = SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=None, block_size=16))
    attn.block_size = 2
    attn.window_size = 6

    with pytest.raises(RuntimeError, match="requires block_table"):
        DeepseekV4DSparkAttention._run_standard_dspark_attention(
            attn,
            q=torch.zeros(2, 4, 8),
            positions=torch.tensor([6, 7], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
            block_table=None,
            dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
        )

    with pytest.raises(RuntimeError, match="requires SWA kv_cache"):
        DeepseekV4DSparkAttention._run_standard_dspark_attention(
            attn,
            q=torch.zeros(2, 4, 8),
            positions=torch.tensor([6, 7], dtype=torch.int32),
            slot_mapping=torch.tensor([6, 7], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            dspark_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            dspark_seq_lens=torch.tensor([8], dtype=torch.int32),
        )
