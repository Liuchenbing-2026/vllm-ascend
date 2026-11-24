# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextvars
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.overrides import TorchFunctionMode

from vllm_ascend import ascend_config as ascend_config_mod
from vllm_ascend.patch.worker import patch_kda_checkpoint as mod


@pytest.fixture(autouse=True)
def _reset_ascend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ascend_config_mod, "_ASCEND_CONFIG", None)


def _build_legacy_kda_modules():
    def _legacy_align(backend: str) -> int | None:
        return 16 if backend == "flashkda" else None

    def _legacy_chunk_kda_with_fused_gate(
        q: Any,
        k: Any,
        v: Any,
        raw_g: Any,
        raw_beta: Any,
        A_log: Any,
        g_bias: Any,
        **kwargs: Any,
    ):
        return q, v

    class _LegacyKimiGDN:
        def __init__(self, prefix: str, vllm_config: Any, *, enable_align: bool = False) -> None:
            self.prefix = prefix
            self.vllm_config = vllm_config
            self.head_dim = 16
            self.conv_size = 4
            self.local_num_heads = 2
            self.kv_cache = (
                torch.zeros((2, 6, 3), dtype=torch.bfloat16),
                torch.zeros((2, 2, 16, 16), dtype=torch.bfloat16),
            )

        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)

        def get_kv_cache_spec(self, vllm_config: Any):
            return "base-spec"

        def _forward(self, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any):
            return "orig-forward-result"

    kda_ckpt_mod = types.ModuleType("vllm.model_executor.layers.mamba.kda_checkpoint")
    kda_ckpt_mod.kda_prefill_checkpoint_alignment = _legacy_align  # type: ignore[attr-defined]

    nvidia_kda_mod = types.ModuleType("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    nvidia_kda_mod.chunk_kda_with_fused_gate = _legacy_chunk_kda_with_fused_gate  # type: ignore[attr-defined]

    kimi_gdn_mod = types.ModuleType("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    kimi_gdn_mod.KimiGatedDeltaNetAttention = _LegacyKimiGDN  # type: ignore[attr-defined]
    return kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, _LegacyKimiGDN


def test_patch_kda_checkpoint_legacy_installs_and_skips_non_align_runtime() -> None:
    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    orig_align = kda_ckpt_mod.kda_prefill_checkpoint_alignment
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }

    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()

    # Global kda_checkpoint helpers must NOT be mutated (P0 fix).
    assert kda_ckpt_mod.kda_prefill_checkpoint_alignment is orig_align
    assert kda_ckpt_mod.kda_prefill_checkpoint_alignment("triton") is None
    assert not hasattr(kda_ckpt_mod, "reserve_triton_kda_checkpoint_buffer")
    assert not hasattr(kda_ckpt_mod, "get_triton_kda_checkpoint_buffer")
    assert getattr(nvidia_kda_mod.chunk_kda_with_fused_gate, mod._KDA_CKPT_PATCH_ATTR, False) is True
    assert hasattr(layer_cls, "get_kv_cache_checkpoint_alignment")
    assert getattr(layer_cls._forward, mod._KDA_CKPT_PATCH_ATTR, False) is True

    # Non-align or opted-out layer must stay a fast-path no-op and clear non-zero MambaSpec blocks.
    from vllm.v1.kv_cache_interface import MambaSpec

    non_align_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, use_ubatching=False),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        kv_transfer_config=None,
        speculative_config=None,
        num_hash_lookahead_tokens=0,
        additional_config={"kda_internal_prefill_checkpoints": True},
    )
    layer = layer_cls("layer.0", non_align_cfg)
    assert layer._ascend_internal_checkpoint_enabled is False
    assert layer.get_kv_cache_checkpoint_alignment(non_align_cfg) is None
    assert layer.get_kv_cache_spec(non_align_cfg) == "base-spec"
    assert layer._forward(None, None, None, None, None) == "orig-forward-result"

    if "num_prefill_checkpoint_blocks" in MambaSpec.__dataclass_fields__:

        def _nonzero_get_kv_cache_spec(self_obj: Any, vllm_config: Any) -> MambaSpec:
            return MambaSpec(
                block_size=256,
                shapes=((8, 3), (2, 16, 16)),
                dtypes=(torch.bfloat16, torch.bfloat16),
                mamba_cache_mode="none",
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=64,
            )

        nonzero_cls = type(
            "_NonzeroSpecKimiGDN",
            (layer_cls,),
            {"get_kv_cache_spec": _nonzero_get_kv_cache_spec},
        )
        nonzero_mod = types.ModuleType("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
        nonzero_mod.KimiGatedDeltaNetAttention = nonzero_cls  # type: ignore[attr-defined]
        with patch(
            "importlib.import_module",
            side_effect=lambda name: {
                "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
                "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
                "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": nonzero_mod,
            }[name],
        ):
            mod.apply_patch()
        cleared = nonzero_cls("layer.0", non_align_cfg).get_kv_cache_spec(non_align_cfg)
        assert cleared.num_prefill_checkpoint_blocks == 0
        assert cleared.prefill_checkpoint_alignment is None


def test_patch_kda_checkpoint_strict_noop_post_upstream_pin() -> None:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    def _native_align(backend: str) -> int | None:
        return 64 if backend == "triton" else 16

    def _native_reserve(*args: Any, **kwargs: Any) -> None:
        return None

    def _native_get(*args: Any, **kwargs: Any) -> tuple[None, None]:
        return None, None

    def _native_chunk_kda_with_fused_gate(
        q: Any,
        k: Any,
        v: Any,
        raw_g: Any,
        raw_beta: Any,
        A_log: Any,
        g_bias: Any,
        checkpoint_state: Any = None,
        checkpoint_offsets: Any = None,
        **kwargs: Any,
    ):
        return q, v

    class _NativeKimiGDN:
        def __init__(self, vllm_config: Any) -> None:
            self.vllm_config = vllm_config

        def get_kv_cache_checkpoint_alignment(self, vllm_config: Any) -> int | None:
            return 64

        def get_kv_cache_spec(self, vllm_config: Any) -> str:
            return "native-spec"

        @eager_break_during_capture
        def _forward(self, *args: Any, **kwargs: Any):
            checkpoint_state = kwargs.get("checkpoint_state")
            return checkpoint_state

    orig_init = _NativeKimiGDN.__init__
    orig_align = _NativeKimiGDN.get_kv_cache_checkpoint_alignment
    orig_spec = _NativeKimiGDN.get_kv_cache_spec
    orig_forward = _NativeKimiGDN._forward

    kda_ckpt_mod = types.ModuleType("vllm.model_executor.layers.mamba.kda_checkpoint")
    kda_ckpt_mod.kda_prefill_checkpoint_alignment = _native_align  # type: ignore[attr-defined]
    kda_ckpt_mod.reserve_triton_kda_checkpoint_buffer = _native_reserve  # type: ignore[attr-defined]
    kda_ckpt_mod.get_triton_kda_checkpoint_buffer = _native_get  # type: ignore[attr-defined]

    nvidia_kda_mod = types.ModuleType("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    nvidia_kda_mod.chunk_kda_with_fused_gate = _native_chunk_kda_with_fused_gate  # type: ignore[attr-defined]

    kimi_gdn_mod = types.ModuleType("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    kimi_gdn_mod.KimiGatedDeltaNetAttention = _NativeKimiGDN  # type: ignore[attr-defined]

    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }
    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()

    # Every module symbol and class method must remain strictly identical (0 replacements).
    assert kda_ckpt_mod.kda_prefill_checkpoint_alignment is _native_align
    assert kda_ckpt_mod.reserve_triton_kda_checkpoint_buffer is _native_reserve
    assert kda_ckpt_mod.get_triton_kda_checkpoint_buffer is _native_get
    assert nvidia_kda_mod.chunk_kda_with_fused_gate is _native_chunk_kda_with_fused_gate
    assert _NativeKimiGDN.__init__ is orig_init
    assert _NativeKimiGDN.get_kv_cache_checkpoint_alignment is orig_align
    assert _NativeKimiGDN.get_kv_cache_spec is orig_spec
    assert _NativeKimiGDN._forward is orig_forward


def test_patch_kda_checkpoint_failsafe_on_compiled_or_raising_targets() -> None:
    def _raising_align(backend: str) -> int | None:
        raise RuntimeError("incompatible probe")

    builtin_callable = MagicMock(side_effect=TypeError("no signature"))

    class _OpaqueKimiGDN:
        def __init__(self) -> None:
            pass

        def get_kv_cache_spec(self, vllm_config: Any) -> str:
            return "spec"

        _forward = len  # Builtin/C callable without Python __code__

    kda_ckpt_mod = types.ModuleType("vllm.model_executor.layers.mamba.kda_checkpoint")
    kda_ckpt_mod.kda_prefill_checkpoint_alignment = _raising_align  # type: ignore[attr-defined]
    nvidia_kda_mod = types.ModuleType("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    nvidia_kda_mod.chunk_kda_with_fused_gate = builtin_callable  # type: ignore[attr-defined]
    kimi_gdn_mod = types.ModuleType("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    kimi_gdn_mod.KimiGatedDeltaNetAttention = _OpaqueKimiGDN  # type: ignore[attr-defined]

    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }
    with (
        patch.object(mod, "_safe_param_names", return_value=None),
        patch("importlib.import_module", side_effect=lambda name: fake_modules[name]),
    ):
        mod.apply_patch()

    assert kda_ckpt_mod.kda_prefill_checkpoint_alignment is _raising_align
    assert nvidia_kda_mod.chunk_kda_with_fused_gate is builtin_callable
    assert _OpaqueKimiGDN._forward is len


def test_patch_kda_checkpoint_alignment_does_not_mutate_shared_helper() -> None:
    import vllm_ascend.ops.kda_checkpoint as ascend_kda_ckpt

    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    orig_align = kda_ckpt_mod.kda_prefill_checkpoint_alignment
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }
    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()
        first_forward = layer_cls._forward
        # Repeated patch application remains idempotent and leaves global alignment helper untouched.
        mod.apply_patch()

    assert kda_ckpt_mod.kda_prefill_checkpoint_alignment is orig_align
    assert layer_cls._forward is first_forward
    assert not hasattr(ascend_kda_ckpt.kda_prefill_checkpoint_alignment, mod._KDA_CKPT_PATCH_ATTR)
    assert not hasattr(ascend_kda_ckpt.kda_prefill_checkpoint_alignment, "__wrapped__")
    assert orig_align("triton") is None
    assert ascend_kda_ckpt.kda_prefill_checkpoint_alignment("triton") == 64
    assert ascend_kda_ckpt.kda_prefill_checkpoint_alignment("flashkda") == 16


def _align_vllm_config(*, max_num_seqs: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, use_ubatching=False),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_transfer_config=None,
        speculative_config=None,
        num_hash_lookahead_tokens=0,
        additional_config={"kda_internal_prefill_checkpoints": True},
    )


def test_patch_kda_checkpoint_mixed_decode_prefill_slices_raw_qkv_and_keeps_module_immutable() -> None:
    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }
    captured_mod_bindings: list[tuple[Any, Any]] = []

    def _peeled_mixed_forward(self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any):
        captured_mod_bindings.append(
            (nvidia_kda_mod.chunk_kda_with_fused_gate, getattr(kimi_gdn_mod, "chunk_kda_with_fused_gate", None))
        )
        # Simulated peeled mixed decode (2) + prefill (128) call using 0-based prefill cu_seqlens.
        prefill_q = mixed_qkv[2:130].view(1, 128, 2, 16)
        return nvidia_kda_mod.chunk_kda_with_fused_gate(
            q=prefill_q,
            k=prefill_q,
            v=prefill_q,
            raw_g=prefill_q,
            raw_beta=torch.zeros((1, 128, 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,), dtype=torch.float32),
            g_bias=None,
            initial_state=torch.zeros((1, 2, 16, 16), dtype=torch.float32),
            cu_seqlens=torch.tensor([0, 128], dtype=torch.int32),
        )

    layer_cls._forward = _peeled_mixed_forward

    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()
        installed_dispatch = nvidia_kda_mod.chunk_kda_with_fused_gate
        assert getattr(kimi_gdn_mod, "chunk_kda_with_fused_gate", None) is None
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer = layer_cls("layer.0", _align_vllm_config())

    assert layer._ascend_internal_checkpoint_enabled is True
    mixed_qkv = torch.arange(130 * 32, dtype=torch.bfloat16).view(130, 32)
    mixed_qkv[:2] = -999.0
    checkpoint = SimpleNamespace(
        checkpoint_offsets=torch.tensor([0, 0, 64], dtype=torch.int32),
        state_indices=torch.tensor([-1, -1, 5], dtype=torch.int64),
        offsets=[0, 0, 64],
        query_lens=[1, 1, 128],
    )
    attn_meta = SimpleNamespace(
        num_prefills=1,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=130,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        checkpoint=checkpoint,
    )
    fwd_ctx = SimpleNamespace(attn_metadata={layer.prefix: attn_meta})

    observed_chunk_calls: list[dict[str, Any]] = []

    def _fake_ascend_chunk(*args: Any, **kwargs: Any):
        observed_chunk_calls.append(kwargs)
        return kwargs["q"], torch.ones((1, 2, 16, 16), dtype=torch.float32)

    with (
        patch("vllm.forward_context.get_forward_context", return_value=fwd_ctx),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(mod, "ascend_chunk_kda_with_fused_gate", side_effect=_fake_ascend_chunk),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export,
    ):
        layer._forward(mixed_qkv, None, None, None, None)

    assert captured_mod_bindings == [(installed_dispatch, None)]
    assert nvidia_kda_mod.chunk_kda_with_fused_gate is installed_dispatch
    assert getattr(kimi_gdn_mod, "chunk_kda_with_fused_gate", None) is None
    assert len(observed_chunk_calls) == 1
    assert observed_chunk_calls[0]["checkpoint_offsets"].tolist() == [64]
    assert tuple(observed_chunk_calls[0]["checkpoint_state"].shape) == (1, 2, 16, 16)
    mock_export.assert_called_once()
    export_kwargs = mock_export.call_args.kwargs
    assert torch.equal(export_kwargs["raw_qkv"], mixed_qkv[2:130])
    assert export_kwargs["raw_qkv"].shape == (128, 32)
    assert mock_export.call_args.args[1].state_indices.tolist() == [5]


def test_patch_kda_checkpoint_by_value_binding_reentrancy_and_exception_reset() -> None:
    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    # Simulate `from ... import chunk_kda_with_fused_gate` bound on kimi_gdn_mod before patching.
    kimi_gdn_mod.chunk_kda_with_fused_gate = nvidia_kda_mod.chunk_kda_with_fused_gate  # type: ignore[attr-defined]
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }
    should_raise = False

    def _by_value_forward(self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any):
        assert nvidia_kda_mod.chunk_kda_with_fused_gate is kimi_gdn_mod.chunk_kda_with_fused_gate
        if should_raise:
            raise RuntimeError("boom")
        prefill_q = mixed_qkv[2:130].view(1, 128, 2, 16)
        return kimi_gdn_mod.chunk_kda_with_fused_gate(
            q=prefill_q,
            k=prefill_q,
            v=prefill_q,
            raw_g=prefill_q,
            raw_beta=torch.zeros((1, 128, 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,), dtype=torch.float32),
            g_bias=None,
            initial_state=torch.zeros((1, 2, 16, 16), dtype=torch.float32),
            cu_seqlens=torch.tensor([0, 128], dtype=torch.int32),
        )

    layer_cls._forward = _by_value_forward

    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()
        installed_dispatch = nvidia_kda_mod.chunk_kda_with_fused_gate
        assert kimi_gdn_mod.chunk_kda_with_fused_gate is installed_dispatch
        hook_var = getattr(installed_dispatch, mod._KDA_CKPT_HOOK_VAR_ATTR)
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer = layer_cls("layer.0", _align_vllm_config())

    mixed_qkv = torch.arange(130 * 32, dtype=torch.bfloat16).view(130, 32)
    attn_meta = SimpleNamespace(
        num_prefills=1,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=130,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        checkpoint=SimpleNamespace(checkpoint_offsets=torch.tensor([64], dtype=torch.int32)),
    )
    fwd_ctx = SimpleNamespace(attn_metadata={layer.prefix: attn_meta})

    with (
        patch("vllm.forward_context.get_forward_context", return_value=fwd_ctx),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=True),
        patch.object(
            mod,
            "ascend_chunk_kda_with_fused_gate",
            side_effect=lambda **kwargs: (kwargs["q"], torch.ones((1, 2, 16, 16))),
        ),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export,
    ):
        layer._forward(mixed_qkv, None, None, None, None)
        assert mock_export.call_count == 1
        assert torch.equal(mock_export.call_args.kwargs["raw_qkv"], mixed_qkv[2:130])
        assert hook_var.get() is None

        # Isolated context runs keep hooks scoped strictly to that ContextVar frame.
        ctx = contextvars.copy_context()
        ctx.run(layer._forward, mixed_qkv, None, None, None, None)
        assert mock_export.call_count == 2
        assert hook_var.get() is None

        # Exception during forward resets the ContextVar hook so subsequent direct calls do not export.
        should_raise = True
        with pytest.raises(RuntimeError, match="boom"):
            layer._forward(mixed_qkv, None, None, None, None)
        assert hook_var.get() is None

        kimi_gdn_mod.chunk_kda_with_fused_gate(q=mixed_qkv[:64].view(1, 64, 2, 16))
        assert mock_export.call_count == 2


def test_patch_kda_checkpoint_unpeeled_mixed_spec_prefill_and_bool_guard() -> None:
    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }

    def _unpeeled_or_full_ns_forward(self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any):
        tokens = core_attn_out  # Test passes pre-sliced token tensor via core_attn_out helper slot.
        cu = g1
        num_seqs = cu.numel() - 1
        return nvidia_kda_mod.chunk_kda_with_fused_gate(
            q=tokens,
            k=tokens,
            v=tokens,
            raw_g=tokens,
            raw_beta=torch.zeros((1, tokens.shape[1], 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,), dtype=torch.float32),
            g_bias=None,
            initial_state=torch.zeros((num_seqs, 2, 16, 16), dtype=torch.float32),
            cu_seqlens=cu,
        )

    layer_cls._forward = _unpeeled_or_full_ns_forward

    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer = layer_cls("layer.0", _align_vllm_config())

    # 1) Unpeeled upstream KimiGatedDeltaNetAttention (decodes + prefill together) pads prefill-only checkpoint rows.
    mixed_qkv = torch.arange(130 * 32, dtype=torch.bfloat16).view(130, 32)
    unpeeled_meta = SimpleNamespace(
        num_prefills=1,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=130,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        checkpoint=SimpleNamespace(
            checkpoint_offsets=torch.tensor([64], dtype=torch.int32),
            state_indices=torch.tensor([9], dtype=torch.int64),
        ),
    )
    chunk_calls: list[dict[str, Any]] = []

    def _fake_unpeeled_chunk(**kwargs: Any) -> tuple[Any, torch.Tensor]:
        chunk_calls.append(kwargs)
        return kwargs["q"], torch.ones((3, 2, 16, 16))

    unpeeled_ctx = SimpleNamespace(attn_metadata={layer.prefix: unpeeled_meta})
    with (
        patch("vllm.forward_context.get_forward_context", return_value=unpeeled_ctx),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(
            mod,
            "ascend_chunk_kda_with_fused_gate",
            side_effect=_fake_unpeeled_chunk,
        ),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export,
    ):
        layer._forward(
            mixed_qkv,
            torch.tensor([0, 1, 2, 130], dtype=torch.int32),
            None,
            None,
            mixed_qkv.view(1, 130, 2, 16),
        )

    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    assert chunk_calls[0]["checkpoint_offsets"].tolist() == [0, 0, 64]
    assert torch.equal(mock_export.call_args.kwargs["raw_qkv"], mixed_qkv)
    assert mock_export.call_args.args[1].state_indices.tolist() == [NULL_BLOCK_ID, NULL_BLOCK_ID, 9]

    # 2) Spec + prefill non-spec selection, plus bool guard on num_decodes/num_decode_tokens.
    spec_mixed_qkv = torch.arange(68 * 32, dtype=torch.bfloat16).view(68, 32)
    non_spec_indx = torch.arange(4, 68, dtype=torch.int64)
    spec_meta = SimpleNamespace(
        num_prefills=1,
        num_decodes=True,
        num_decode_tokens=True,
        num_actual_tokens=68,
        spec_sequence_masks=torch.tensor([True, False]),
        non_spec_token_indx=non_spec_indx,
        checkpoint=SimpleNamespace(checkpoint_offsets=torch.tensor([64], dtype=torch.int32)),
    )
    spec_ctx = SimpleNamespace(attn_metadata={layer.prefix: spec_meta})
    with (
        patch("vllm.forward_context.get_forward_context", return_value=spec_ctx),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(
            mod,
            "ascend_chunk_kda_with_fused_gate",
            side_effect=lambda **kwargs: (kwargs["q"], torch.ones((1, 2, 16, 16))),
        ),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export_spec,
    ):
        layer._forward(
            spec_mixed_qkv,
            torch.tensor([0, 64], dtype=torch.int32),
            None,
            None,
            spec_mixed_qkv[4:68].view(1, 64, 2, 16),
        )

    assert torch.equal(mock_export_spec.call_args.kwargs["raw_qkv"], spec_mixed_qkv[4:68])


def test_patch_kda_checkpoint_preserves_outermost_eager_break_during_capture_and_replay() -> None:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, eager_break_during_capture
    from vllm.config import CUDAGraphMode

    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }
    fwd_ctx_holder: list[Any] = []
    capture_events: list[str] = []
    capturing_at_export: list[bool] = []

    class _FakeCUDAGraph:
        def __init__(self) -> None:
            self.graph_id = len([ev for ev in capture_events if ev == "begin"])

        def capture_begin(self, pool: Any = None) -> None:
            capture_events.append("begin")

        def capture_end(self) -> None:
            capture_events.append("end")

        def replay(self) -> None:
            capture_events.append(f"replay_{self.graph_id}")

    class _AssertNoCaptureIndexSelect(TorchFunctionMode):
        def __init__(self, cap: BreakableCUDAGraphCapture) -> None:
            super().__init__()
            self._cap = cap
            self.index_select_calls = 0

        def __torch_function__(self, func, types_, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.index_select or func is torch.index_select:
                self.index_select_calls += 1
                assert not self._cap._capturing, "index_select executed while BreakableCUDAGraph was capturing"
            return func(*args, **kwargs)

    @eager_break_during_capture
    def _eager_gdn_forward(
        self: Any,
        mixed_qkv: Any,
        g1: Any,
        g2: Any,
        beta: Any,
        core_attn_out: Any,
    ) -> None:
        attn_meta = fwd_ctx_holder[0].attn_metadata[self.prefix]
        if attn_meta.num_prefills <= 0:
            core_attn_out[:2].copy_(mixed_qkv[:2])
            return
        prefill_q = mixed_qkv[6:70].view(1, 64, 2, 16)
        nvidia_kda_mod.chunk_kda_with_fused_gate(
            q=prefill_q,
            k=prefill_q,
            v=prefill_q,
            raw_g=prefill_q,
            raw_beta=torch.zeros((1, 64, 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,), dtype=torch.float32),
            g_bias=None,
            initial_state=torch.zeros((1, 2, 16, 16), dtype=torch.float32),
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
        )
        core_attn_out[:64].copy_(mixed_qkv[6:70])

    orig_undecorated = _eager_gdn_forward.__wrapped__
    layer_cls._forward = _eager_gdn_forward

    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()
        first_wrapped = layer_cls._forward
        installed_dispatch = nvidia_kda_mod.chunk_kda_with_fused_gate
        hook_var = getattr(installed_dispatch, mod._KDA_CKPT_HOOK_VAR_ATTR)
        # Re-applying must be idempotent and keep @eager_break_during_capture outermost.
        mod.apply_patch()
        assert layer_cls._forward is first_wrapped

    assert "BreakableCUDAGraphCapture" in first_wrapped.__code__.co_names
    assert "add_eager" in first_wrapped.__code__.co_names
    assert getattr(first_wrapped, mod._KDA_CKPT_PATCH_ATTR, False) is True
    assert getattr(first_wrapped.__wrapped__, mod._KDA_CKPT_PATCH_ATTR, False) is True
    assert first_wrapped.__wrapped__.__wrapped__ is orig_undecorated

    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer = layer_cls("layer.0", _align_vllm_config())

    static_mixed_qkv = torch.zeros((70, 32), dtype=torch.bfloat16)
    static_mixed_qkv[4:6] = -1.0
    static_mixed_qkv[6:70] = 3.0
    static_out = torch.zeros((70, 32), dtype=torch.bfloat16)
    non_spec_indx = torch.arange(4, 70, dtype=torch.int64)

    mixed_checkpoint = SimpleNamespace(
        checkpoint_offsets=torch.tensor([0, 0, 64], dtype=torch.int32),
        state_indices=torch.tensor([-1, -1, 5], dtype=torch.int64),
        offsets=[0, 0, 64],
        query_lens=[1, 1, 64],
    )
    mixed_meta = SimpleNamespace(
        num_prefills=1,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=70,
        spec_sequence_masks=torch.tensor([True, False]),
        non_spec_token_indx=non_spec_indx,
        checkpoint=mixed_checkpoint,
    )
    decode_only_meta = SimpleNamespace(
        num_prefills=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=70,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        checkpoint=None,
    )
    observed_chunk_calls: list[dict[str, Any]] = []

    def _fake_ascend_chunk(*args: Any, **kwargs: Any) -> tuple[Any, torch.Tensor]:
        observed_chunk_calls.append(kwargs)
        return kwargs["q"], torch.ones((1, 2, 16, 16), dtype=torch.float32)

    def _record_export(*args: Any, **kwargs: Any) -> None:
        cap = BreakableCUDAGraphCapture.current()
        capturing_at_export.append(bool(cap is not None and cap._capturing))

    with (
        patch("torch.cuda.CUDAGraph", _FakeCUDAGraph),
        patch("vllm.compilation.breakable_cudagraph.weak_ref_tensor", side_effect=lambda x: x),
        patch("vllm.compilation.breakable_cudagraph.is_forward_context_available", return_value=True),
        patch("vllm.compilation.breakable_cudagraph.get_forward_context", side_effect=lambda: fwd_ctx_holder[0]),
        patch("vllm.forward_context.get_forward_context", side_effect=lambda: fwd_ctx_holder[0]),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(mod, "ascend_chunk_kda_with_fused_gate", side_effect=_fake_ascend_chunk),
        patch.object(mod, "export_kda_prefill_checkpoint", side_effect=_record_export) as mock_export,
    ):
        # 1) During capture with mixed+spec metadata, eager break exits capture BEFORE index_select & export.
        fwd_ctx_holder.append(
            SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, attn_metadata={layer.prefix: mixed_meta})
        )
        cap_mixed = BreakableCUDAGraphCapture()
        with cap_mixed, _AssertNoCaptureIndexSelect(cap_mixed) as guard:
            layer._forward(static_mixed_qkv, None, None, None, static_out)

        assert guard.index_select_calls == 1
        assert cap_mixed.num_eager_breaks == 1
        assert cap_mixed.num_graphs == 2
        assert mock_export.call_count == 1
        assert capturing_at_export == [False]
        assert torch.all(mock_export.call_args.kwargs["raw_qkv"] == 3.0)
        assert hook_var.get() is None

        # 2) Capture under decode-only (no checkpoint), then replay under mixed+spec: checkpoint hook must run.
        mock_export.reset_mock()
        observed_chunk_calls.clear()
        fwd_ctx_holder[0] = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={layer.prefix: decode_only_meta},
        )
        cap_replay = BreakableCUDAGraphCapture()
        with cap_replay:
            layer._forward(static_mixed_qkv, None, None, None, static_out)
        assert cap_replay.num_eager_breaks == 1
        assert cap_replay.num_graphs == 2
        assert mock_export.call_count == 0

        fwd_ctx_holder[0] = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            attn_metadata={layer.prefix: mixed_meta},
        )
        static_mixed_qkv[6:70].fill_(7.0)
        cap_replay.replay()
        assert mock_export.call_count == 1
        assert observed_chunk_calls[-1]["checkpoint_offsets"].tolist() == [64]
        assert mock_export.call_args.kwargs["raw_qkv"].shape == (64, 32)
        assert torch.all(mock_export.call_args.kwargs["raw_qkv"] == 7.0)
        assert hook_var.get() is None

        static_mixed_qkv[6:70].fill_(11.0)
        cap_replay.replay()
        assert mock_export.call_count == 2
        assert torch.all(mock_export.call_args.kwargs["raw_qkv"] == 11.0)
        assert hook_var.get() is None


def test_patch_kda_checkpoint_real_kimi_gdn_class_keeps_eager_break_outermost() -> None:
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention

    mod.apply_patch()
    forward_fn = KimiGatedDeltaNetAttention._forward
    assert "BreakableCUDAGraphCapture" in forward_fn.__code__.co_names
    assert "add_eager" in forward_fn.__code__.co_names
    if not mod._layer_has_native_kda_checkpoint(KimiGatedDeltaNetAttention):
        assert getattr(forward_fn, mod._KDA_CKPT_PATCH_ATTR, False) is True
        ckpt_wrapper = getattr(forward_fn, "__wrapped__", None)
        assert ckpt_wrapper is not None
        assert getattr(ckpt_wrapper, mod._KDA_CKPT_PATCH_ATTR, False) is True
        raw_inner = getattr(ckpt_wrapper, "__wrapped__", None)
        assert raw_inner is not None
        assert "BreakableCUDAGraphCapture" not in raw_inner.__code__.co_names

    mod.apply_patch()
    assert KimiGatedDeltaNetAttention._forward is forward_fn


def test_patch_kda_checkpoint_zero_len_padding_rows_prefill_only_and_mixed() -> None:
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    kda_ckpt_mod, nvidia_kda_mod, kimi_gdn_mod, layer_cls = _build_legacy_kda_modules()
    fake_modules = {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt_mod,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nvidia_kda_mod,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn_mod,
    }

    def _padded_cu_seqlens_forward(self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any):
        cu = g1
        num_seqs = cu.numel() - 1
        return nvidia_kda_mod.chunk_kda_with_fused_gate(
            q=core_attn_out,
            k=core_attn_out,
            v=core_attn_out,
            raw_g=core_attn_out,
            raw_beta=torch.zeros((1, core_attn_out.shape[1], 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,), dtype=torch.float32),
            g_bias=None,
            initial_state=torch.zeros((num_seqs, 2, 16, 16), dtype=torch.float32),
            cu_seqlens=cu,
        )

    layer_cls._forward = _padded_cu_seqlens_forward
    with patch("importlib.import_module", side_effect=lambda name: fake_modules[name]):
        mod.apply_patch()
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer = layer_cls("layer.0", _align_vllm_config())

    # 1) Pure prefill (num_decodes=0) with a middle zero-length pad row: (2,) -> (3,)
    prefill_qkv = torch.zeros((192, 32), dtype=torch.bfloat16)
    cu_prefill = torch.tensor([0, 64, 64, 192], dtype=torch.int32)
    meta_prefill = SimpleNamespace(
        num_prefills=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_actual_tokens=192,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        non_spec_query_start_loc_cpu=cu_prefill,
        checkpoint=SimpleNamespace(
            checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32),
            state_indices=torch.tensor([5, 9], dtype=torch.int64),
            request_rows=torch.tensor([0, 2], dtype=torch.int64),
            block_cols=torch.tensor([3, 5], dtype=torch.int64),
            offsets=[64, 128],
            query_lens=[64, 128],
        ),
    )
    chunk_calls: list[dict[str, Any]] = []

    def _make_chunk_recorder(num_seqs: int):
        def _record(**kwargs: Any) -> tuple[Any, torch.Tensor]:
            chunk_calls.append(kwargs)
            return kwargs["q"], torch.ones((num_seqs, 2, 16, 16))

        return _record

    with (
        patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={layer.prefix: meta_prefill}),
        ),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(
            mod,
            "ascend_chunk_kda_with_fused_gate",
            side_effect=_make_chunk_recorder(3),
        ),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export,
    ):
        layer._forward(prefill_qkv, cu_prefill, None, None, prefill_qkv.view(1, 192, 2, 16))

    assert chunk_calls[0]["checkpoint_offsets"].tolist() == [64, 0, 128]
    exported_ckpt = mock_export.call_args.args[1]
    assert exported_ckpt.state_indices.tolist() == [5, NULL_BLOCK_ID, 9]
    assert exported_ckpt.request_rows.tolist() == [0, 0, 2]
    assert exported_ckpt.block_cols.tolist() == [3, -1, 5]
    assert exported_ckpt.offsets == [64, 0, 128]
    assert exported_ckpt.query_lens == [64, 0, 128]

    # 2) Mixed decode + zero-length prefill pad row: num_decodes=2, base_len=2, num_seqs=5
    mixed_qkv = torch.zeros((194, 32), dtype=torch.bfloat16)
    cu_mixed = torch.tensor([0, 1, 2, 66, 66, 194], dtype=torch.int32)
    meta_mixed = SimpleNamespace(
        num_prefills=2,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=194,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        non_spec_query_start_loc_cpu=cu_mixed,
        checkpoint=SimpleNamespace(
            checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32),
            state_indices=torch.tensor([5, 9], dtype=torch.int64),
            block_cols=torch.tensor([3, 5], dtype=torch.int64),
            offsets=[64, 128],
            query_lens=[64, 128],
        ),
    )
    chunk_calls.clear()
    with (
        patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={layer.prefix: meta_mixed}),
        ),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(
            mod,
            "ascend_chunk_kda_with_fused_gate",
            side_effect=_make_chunk_recorder(5),
        ),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export_mixed,
    ):
        layer._forward(mixed_qkv, cu_mixed, None, None, mixed_qkv.view(1, 194, 2, 16))

    assert chunk_calls[0]["checkpoint_offsets"].tolist() == [0, 0, 64, 0, 128]
    exported_mixed = mock_export_mixed.call_args.args[1]
    assert exported_mixed.state_indices.tolist() == [NULL_BLOCK_ID, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 9]
    assert exported_mixed.block_cols.tolist() == [-1, -1, 3, -1, 5]
    assert exported_mixed.offsets == [0, 0, 64, 0, 128]

    # 3) Device-only fallback without cu_seqlens_cpu triggers zero .tolist() / .item() D2H syncs.
    class _NoSyncTensor(torch.Tensor):
        _fake_device = SimpleNamespace(type="fake_npu")

        @property
        def device(self) -> Any:  # type: ignore[override]
            return self._fake_device

        def to(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return self

        def tolist(self) -> Any:
            raise AssertionError("unexpected D2H .tolist() sync in _align_checkpoint_rows")

        def item(self) -> Any:
            raise AssertionError("unexpected D2H .item() sync in _align_checkpoint_rows")

    dev_cu = torch.tensor([0, 1, 2, 66, 66, 194], dtype=torch.int32).as_subclass(_NoSyncTensor)
    dev_ckpt = SimpleNamespace(
        checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32).as_subclass(_NoSyncTensor),
        state_indices=torch.tensor([5, 9], dtype=torch.int64).as_subclass(_NoSyncTensor),
        offsets=[64, 128],
        query_lens=[64, 128],
    )
    aligned_dev = mod._align_checkpoint_rows(
        dev_ckpt,
        num_seqs=5,
        num_decodes=2,
        peeled_decodes=False,
        cu_seqlens=dev_cu,
        cu_seqlens_cpu=None,
    )
    assert torch.equal(
        aligned_dev.checkpoint_offsets.as_subclass(torch.Tensor),
        torch.tensor([0, 0, 64, 0, 128], dtype=torch.int32),
    )
    assert torch.equal(
        aligned_dev.state_indices.as_subclass(torch.Tensor),
        torch.tensor([NULL_BLOCK_ID, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 9], dtype=torch.int64),
    )
    assert aligned_dev.offsets is None
    assert aligned_dev.query_lens is None
