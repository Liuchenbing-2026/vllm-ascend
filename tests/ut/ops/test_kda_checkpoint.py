# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend import ascend_config as ascend_config_mod
from vllm_ascend.ops import kda_checkpoint
from vllm_ascend.ops.triton.kda import chunk_delta_h
from vllm_ascend.ops.triton.kda import kda as kda_triton


@pytest.fixture(autouse=True)
def _reset_ascend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ascend_config_mod, "_ASCEND_CONFIG", None)


def _valid_vllm_config(
    *,
    mamba_cache_mode: str = "align",
    pcp_size: int = 1,
    use_ubatching: bool = False,
    kda_internal_prefill_checkpoints: Any = True,
):
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            use_ubatching=use_ubatching,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        kv_transfer_config=None,
        speculative_config=None,
        num_hash_lookahead_tokens=0,
    )
    if kda_internal_prefill_checkpoints is not None:
        cfg.additional_config = {"kda_internal_prefill_checkpoints": kda_internal_prefill_checkpoints}
    return cfg


def test_kda_prefill_checkpoint_alignment() -> None:
    assert kda_checkpoint.kda_prefill_checkpoint_alignment("triton") == 64
    assert kda_checkpoint.kda_prefill_checkpoint_alignment("flashkda") == 16
    assert kda_checkpoint.kda_prefill_checkpoint_alignment("unknown") is None


def test_supports_kda_prefill_checkpoint_guards() -> None:
    cfg = _valid_vllm_config()
    assert kda_checkpoint.supports_kda_prefill_checkpoint(cfg, head_dim=128, conv_size=4, local_num_heads=8) is True
    assert (
        kda_checkpoint.supports_kda_prefill_checkpoint(
            _valid_vllm_config(mamba_cache_mode="none"), head_dim=128, conv_size=4, local_num_heads=8
        )
        is False
    )
    assert (
        kda_checkpoint.supports_kda_prefill_checkpoint(
            _valid_vllm_config(pcp_size=2), head_dim=128, conv_size=4, local_num_heads=8
        )
        is False
    )
    assert (
        kda_checkpoint.supports_kda_prefill_checkpoint(
            _valid_vllm_config(use_ubatching=True), head_dim=128, conv_size=4, local_num_heads=8
        )
        is False
    )
    assert kda_checkpoint.supports_kda_prefill_checkpoint(cfg, head_dim=512, conv_size=4, local_num_heads=8) is False
    # Minimal UT doubles omitting cache_config/parallel_config must safely return False.
    assert (
        kda_checkpoint.supports_kda_prefill_checkpoint(
            SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
            head_dim=128,
            conv_size=4,
            local_num_heads=8,
        )
        is False
    )


def test_initialize_and_upgrade_kda_kv_cache_spec() -> None:
    cfg = _valid_vllm_config()
    layer = SimpleNamespace(
        head_dim=128,
        conv_size=4,
        local_num_heads=4,
        get_state_dtype=lambda: (torch.bfloat16, torch.bfloat16),
    )
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        enabled = kda_checkpoint.initialize_kda_checkpoints(layer, cfg)

    assert enabled is True
    assert layer._ascend_internal_checkpoint_enabled is True
    assert layer._triton_checkpoint_spec == ((8, 4, 128, 128), torch.bfloat16)

    base_spec = MambaSpec(
        block_size=256,
        shapes=((8, 3), (4, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=0,
        prefill_checkpoint_alignment=None,
    )
    upgraded = kda_checkpoint.build_kda_kv_cache_spec(layer, cfg, base_spec)
    assert upgraded.num_prefill_checkpoint_blocks == 1
    assert upgraded.prefill_checkpoint_alignment == 64

    already_set = replace(base_spec, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=64)
    assert kda_checkpoint.build_kda_kv_cache_spec(layer, cfg, already_set) == already_set


@pytest.mark.parametrize("conv_state_is_dim_first", [True, False])
def test_export_kda_prefill_checkpoint_normalizes_conv_layout(conv_state_is_dim_first: bool) -> None:
    exporter = MagicMock()
    checkpoint = SimpleNamespace(checkpoint_offsets=torch.tensor([64], dtype=torch.int32))
    raw_qkv = torch.zeros((128, 12), dtype=torch.bfloat16)
    conv_state = (
        torch.zeros((4, 12, 3), dtype=torch.bfloat16)
        if conv_state_is_dim_first
        else torch.zeros((4, 3, 12), dtype=torch.bfloat16)
    )
    recurrent_checkpoint = torch.zeros((1, 2, 16, 16), dtype=torch.bfloat16)
    recurrent_state = torch.zeros((4, 2, 16, 16), dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)

    kda_checkpoint.export_kda_prefill_checkpoint(
        exporter,
        checkpoint,
        raw_qkv=raw_qkv,
        conv_state=conv_state,
        conv_state_is_dim_first=conv_state_is_dim_first,
        recurrent_checkpoint=recurrent_checkpoint,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
    )

    exporter.export.assert_called_once()
    kwargs = exporter.export.call_args.kwargs
    assert tuple(kwargs["conv_state"].shape) == (4, 12, 3)
    assert kwargs["raw_qkv"] is raw_qkv
    assert kwargs["recurrent_checkpoint"] is recurrent_checkpoint
    assert kwargs["recurrent_state"] is recurrent_state


def test_run_kda_checkpoint_prefill_exports_and_passes_buffers() -> None:
    exporter = MagicMock()
    spec = ((4, 2, 16, 16), torch.bfloat16)
    checkpoint = SimpleNamespace(checkpoint_offsets=torch.tensor([64], dtype=torch.int32))
    q = torch.ones((1, 128, 2, 16), dtype=torch.bfloat16)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    raw_gate = torch.zeros_like(q)
    raw_beta = torch.zeros((1, 128, 2), dtype=torch.bfloat16)
    a_log = torch.zeros((2,), dtype=torch.float32)
    initial_state = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
    raw_qkv = torch.zeros((128, 96), dtype=torch.bfloat16)
    conv_state = torch.zeros((4, 3, 96), dtype=torch.bfloat16)
    recurrent_state = torch.zeros((4, 2, 16, 16), dtype=torch.bfloat16)

    fake_out = torch.ones_like(v)
    fake_final = torch.ones((1, 2, 16, 16), dtype=torch.float32)
    with (
        patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False),
        patch.object(kda_checkpoint, "chunk_kda_with_fused_gate", return_value=(fake_out, fake_final)) as mock_kda,
    ):
        out, final_state = kda_checkpoint.run_kda_checkpoint_prefill(
            exporter=exporter,
            checkpoint_spec=spec,
            checkpoint=checkpoint,
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            a_log=a_log,
            dt_bias=None,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            raw_qkv=raw_qkv,
            conv_state=conv_state,
            conv_state_is_dim_first=False,
            recurrent_state=recurrent_state,
            lower_bound=-5.0,
        )

    assert out is fake_out
    assert final_state is fake_final
    call_kwargs = mock_kda.call_args.kwargs
    assert call_kwargs["use_qk_l2norm_in_kernel"] is True
    assert call_kwargs["checkpoint_offsets"] is checkpoint.checkpoint_offsets
    assert tuple(call_kwargs["checkpoint_state"].shape) == (1, 2, 16, 16)
    exporter.export.assert_called_once()


def test_chunk_delta_h_and_chunk_kda_with_fused_gate_contracts() -> None:
    fake_npu = _FakeDevice(type="npu")
    k = _as_fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32), fake_npu)
    w = _as_fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32), fake_npu)
    u = _as_fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32), fake_npu)
    cu = _as_fake_npu(torch.tensor([0, 128], dtype=torch.int32), fake_npu)

    with pytest.raises(ValueError, match="provided together"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k,
            w=w,
            u=u,
            cu_seqlens=cu,
            chunk_indices=torch.zeros((2, 2), dtype=torch.int32),
            chunk_offsets=torch.tensor([0, 2], dtype=torch.int32),
            checkpoint_state=_as_fake_npu(torch.zeros((1, 2, 64, 64)), fake_npu),
            checkpoint_offsets=None,
        )

    with pytest.raises(ValueError, match="checkpoint_state must have shape"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k,
            w=w,
            u=u,
            cu_seqlens=cu,
            chunk_indices=torch.zeros((2, 2), dtype=torch.int32),
            chunk_offsets=torch.tensor([0, 2], dtype=torch.int32),
            checkpoint_state=_as_fake_npu(torch.zeros((2, 2, 64, 64)), fake_npu),
            checkpoint_offsets=_as_fake_npu(torch.tensor([64], dtype=torch.int32), fake_npu),
        )

    q = torch.ones((1, 64, 2, 16), dtype=torch.float32)
    k_small = torch.ones_like(q)
    v_small = torch.ones_like(q)
    raw_g = torch.zeros_like(q)
    raw_beta = torch.zeros((1, 64, 2), dtype=torch.float32)
    a_log = torch.zeros((2,), dtype=torch.float32)
    ckpt_state = torch.empty((1, 2, 16, 16), dtype=torch.float32)
    ckpt_offsets = torch.tensor([0], dtype=torch.int32)

    with patch.object(kda_triton, "chunk_kda", return_value=(v_small, None)) as mock_chunk_kda:
        kda_triton.chunk_kda_with_fused_gate(
            q=q,
            k=k_small,
            v=v_small,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=a_log.view(1, 1, 2, 1),
            g_bias=None,
            lower_bound=-5.0,
            checkpoint_state=ckpt_state,
            checkpoint_offsets=ckpt_offsets,
        )
    forwarded = mock_chunk_kda.call_args.kwargs
    assert forwarded["checkpoint_state"] is ckpt_state
    assert forwarded["checkpoint_offsets"] is ckpt_offsets
    torch.testing.assert_close(forwarded["beta"], torch.full_like(raw_beta, 0.5))
    assert float(forwarded["g"].max()) <= 0.0


@dataclass
class _DummyCheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    state_indices: torch.Tensor
    offsets: list[int]
    query_lens: tuple[int, ...]
    request_rows: torch.Tensor | None = None
    block_cols: torch.Tensor | None = None


def test_prepare_kda_checkpoint_prefill_inputs_fast_path_and_compaction() -> None:
    state_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    has_initial_state = torch.tensor([True, False, True], dtype=torch.bool)
    query_start_loc = torch.tensor([0, 64, 64, 192], dtype=torch.int32)
    ckpt = _DummyCheckpointMetadata(
        checkpoint_offsets=torch.tensor([64, 0, 128], dtype=torch.int32),
        state_indices=torch.tensor([11, 22, 33], dtype=torch.int64),
        offsets=[64, 0, 128],
        query_lens=(64, 0, 128),
        request_rows=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_cols=torch.tensor([3, 4, 5], dtype=torch.int32),
    )

    fast_meta = SimpleNamespace(keep_meta=None, cu_seqlens_kern=None, cu_seqlens_host=(0, 64, 64, 192))
    out_si, out_hi, out_cu, out_ckpt, out_prebuilt = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc,
        chunk_meta=fast_meta,
        checkpoint=ckpt,
    )
    assert out_si is state_indices
    assert out_hi is has_initial_state
    assert out_cu is query_start_loc
    assert out_ckpt is ckpt
    assert out_prebuilt is fast_meta

    compact_meta = SimpleNamespace(
        keep_meta=torch.tensor([True, False, True], dtype=torch.bool),
        cu_seqlens_kern=(0, 64, 192),
    )
    c_si, c_hi, c_cu, c_ckpt, c_prebuilt = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc,
        chunk_meta=compact_meta,
        checkpoint=ckpt,
    )
    assert isinstance(c_cu, torch.Tensor)
    assert c_cu.dtype == torch.int32
    assert c_prebuilt is None
    assert c_si.tolist() == [1, 3]
    assert c_hi.tolist() == [True, True]
    assert c_cu.tolist() == [0, 64, 192]
    assert c_ckpt is not ckpt
    assert c_ckpt.checkpoint_offsets.tolist() == [64, 128]
    assert c_ckpt.state_indices.tolist() == [11, 33]
    assert c_ckpt.offsets == [64, 128]
    assert c_ckpt.query_lens == (64, 128)
    assert c_ckpt.request_rows.tolist() == [0, 2]
    assert c_ckpt.block_cols.tolist() == [3, 5]
    # Original checkpoint remains untouched (safe across layers).
    assert ckpt.checkpoint_offsets.tolist() == [64, 0, 128]


def test_prepare_kda_checkpoint_prefill_inputs_fallback_and_no_double_slice() -> None:
    state_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    has_initial_state = torch.tensor([True, False, True], dtype=torch.bool)
    query_start_loc = torch.tensor([0, 64, 64, 192], dtype=torch.int32)
    already_compacted_ckpt = SimpleNamespace(
        checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32),
        offsets=[64, 128],
    )
    chunk_meta = SimpleNamespace(
        keep_meta=torch.tensor([True, False, True], dtype=torch.bool),
        cu_seqlens_kern=None,
    )
    c_si, _, c_cu, c_ckpt, c_prebuilt = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc,
        chunk_meta=chunk_meta,
        checkpoint=already_compacted_ckpt,
    )
    assert c_prebuilt is None
    assert c_si.tolist() == [1, 3]
    assert isinstance(c_cu, torch.Tensor) and c_cu.tolist() == [0, 64, 192]
    assert c_ckpt is already_compacted_ckpt
    assert c_ckpt.checkpoint_offsets.tolist() == [64, 128]

    ns_ckpt = SimpleNamespace(checkpoint_offsets=torch.tensor([64, 0, 128], dtype=torch.int32), query_lens=[64, 0, 128])
    _, _, _, sliced_ns, _ = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc,
        chunk_meta=SimpleNamespace(keep_meta=torch.tensor([True, False, True]), cu_seqlens_kern=(0, 64, 192)),
        checkpoint=ns_ckpt,
    )
    assert sliced_ns is not ns_ckpt
    assert sliced_ns.checkpoint_offsets.tolist() == [64, 128]
    assert sliced_ns.query_lens == [64, 128]


def test_glm5next_checkpoint_prefill_handles_keep_meta_and_host_tuple() -> None:
    import vllm_ascend.models.glm5next.kda as glm5next_kda_mod

    ckpt = _DummyCheckpointMetadata(
        checkpoint_offsets=torch.tensor([64, 0], dtype=torch.int32),
        state_indices=torch.tensor([2, 3], dtype=torch.int64),
        offsets=[64, 0],
        query_lens=(64, 0),
    )
    chunk_meta = SimpleNamespace(keep_meta=torch.tensor([True, False]), cu_seqlens_kern=(0, 64))
    fake_out = torch.ones((1, 64, 2, 16), dtype=torch.float32)
    fake_final = torch.ones((1, 2, 16, 16), dtype=torch.float32)

    glm_layer: Any = glm5next_kda_mod.Glm5NextLinearAttention.__new__(glm5next_kda_mod.Glm5NextLinearAttention)
    torch.nn.Module.__init__(glm_layer)
    glm_layer.prefix = "model.layers.0.self_attn"
    glm_layer.kda_lower_bound = -5.0
    glm_layer.local_num_heads, glm_layer.head_dim = 2, 16
    glm_layer.local_projection_size = 32
    glm_layer.kv_cache = (torch.zeros((4, 3, 96)), torch.zeros((4, 2, 16, 16)))
    glm_layer._conv_state_dim_first = False
    glm_layer._merged_conv_weight = torch.zeros((96, 4))
    glm_layer.A_log, glm_layer.dt_bias = torch.zeros((2,)), None
    glm_layer._ascend_internal_checkpoint_enabled = True
    glm_layer._checkpoint_exporter = MagicMock()
    glm_layer._triton_checkpoint_spec = ((4, 2, 16, 16), torch.float32)

    meta: Any = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=64,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=64,
        non_spec_state_indices_tensor=torch.tensor([0, 1], dtype=torch.int32),
    )
    meta.prefill_query_start_loc = torch.tensor([0, 64, 64], dtype=torch.int32)
    meta.prefill_state_indices = torch.tensor([2, 3], dtype=torch.int64)
    meta.prefill_has_initial_state = torch.tensor([True, False], dtype=torch.bool)
    meta.non_spec_prefill_metadata = SimpleNamespace(
        causal_conv1d=SimpleNamespace(query_start_loc=None, cache_indices=None, initial_state_mode=None),
        chunk=chunk_meta,
    )
    meta.checkpoint = ckpt
    fwd_ctx = SimpleNamespace(attn_metadata={glm_layer.prefix: meta})
    core_out = torch.zeros((1, 64, 2, 16), dtype=torch.float32)
    with (
        patch.object(glm5next_kda_mod, "get_forward_context", return_value=fwd_ctx),
        patch.object(glm5next_kda_mod, "wait_for_kv_layer_from_connector"),
        patch.object(glm5next_kda_mod, "record_attention_compute_start"),
        patch.object(glm5next_kda_mod, "maybe_save_kv_layer_to_connector"),
        patch.object(glm5next_kda_mod, "causal_conv1d", side_effect=lambda x, *args, **kwargs: x),
        patch.object(glm5next_kda_mod, "gather_initial_states", return_value=torch.zeros((1, 2, 16, 16))),
        patch.object(glm5next_kda_mod, "scatter_states"),
        patch.object(
            glm5next_kda_mod,
            "run_kda_checkpoint_prefill",
            return_value=(fake_out, fake_final),
        ) as mock_glm_run,
    ):
        glm_layer._forward_core(
            torch.ones((64, 96)),
            torch.zeros((1, 64, 2, 16)),
            torch.zeros((1, 64, 2)),
            core_out,
        )
    glm_kwargs = mock_glm_run.call_args.kwargs
    assert isinstance(glm_kwargs["cu_seqlens"], torch.Tensor)
    assert glm_kwargs["cu_seqlens"].tolist() == [0, 64]
    assert glm_kwargs["prebuilt_meta"] is None
    assert glm_kwargs["checkpoint"].checkpoint_offsets.tolist() == [64]


def test_kimi_kda_checkpoint_prefill_handles_keep_meta_and_host_tuple() -> None:
    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

    ckpt = _DummyCheckpointMetadata(
        checkpoint_offsets=torch.tensor([64, 0], dtype=torch.int32),
        state_indices=torch.tensor([2, 3], dtype=torch.int64),
        offsets=[64, 0],
        query_lens=(64, 0),
    )
    chunk_meta = SimpleNamespace(keep_meta=torch.tensor([True, False]), cu_seqlens_kern=(0, 64))
    fake_out = torch.ones((1, 64, 2, 16), dtype=torch.float32)
    fake_final = torch.ones((1, 2, 16, 16), dtype=torch.float32)

    kimi_layer: Any = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    kimi_layer._checkpoint_exporter = MagicMock()
    kimi_layer._triton_checkpoint_spec = ((4, 2, 16, 16), torch.float32)
    kimi_layer._ascend_kda_state_copy = SimpleNamespace(
        gather=MagicMock(return_value=torch.zeros((1, 2, 16, 16))),
        scatter=MagicMock(),
    )
    kimi_layer._kda_state_copy_ready = True
    kimi_layer.A_log, kimi_layer.dt_bias, kimi_layer.gate_lower_bound = torch.zeros((2,)), None, -5.0
    with patch("vllm_ascend.ops.kimi_kda.run_kda_checkpoint_prefill", return_value=(fake_out, fake_final)) as mock_kimi:
        kimi_layer._run_prefill_with_checkpoint(
            q=fake_out,
            k=fake_out,
            v=fake_out,
            raw_gate=fake_out,
            beta=torch.zeros((1, 64, 2)),
            raw_qkv=torch.zeros((64, 96)),
            conv_state=torch.zeros((4, 3, 96)),
            recurrent_state=torch.zeros((4, 2, 16, 16)),
            state_indices=torch.tensor([2, 3], dtype=torch.int64),
            has_initial_state=torch.tensor([True, False], dtype=torch.bool),
            prefill_query_start_loc=torch.tensor([0, 64, 64], dtype=torch.int32),
            prebuilt_metadata=chunk_meta,
            checkpoint=ckpt,
        )
    kimi_kwargs = mock_kimi.call_args.kwargs
    assert isinstance(kimi_kwargs["cu_seqlens"], torch.Tensor)
    assert kimi_kwargs["cu_seqlens"].tolist() == [0, 64]
    assert kimi_kwargs["prebuilt_meta"] is None
    assert kimi_kwargs["checkpoint"].checkpoint_offsets.tolist() == [64]


def test_chunk_kda_with_fused_gate_raw_beta_defaults_to_softplus_unless_bounded() -> None:
    q = torch.ones((1, 16, 2, 16), dtype=torch.float32)
    raw_beta = torch.zeros((1, 16, 2), dtype=torch.float32)
    a_log = torch.zeros((2,), dtype=torch.float32)

    with (
        patch.object(kda_triton, "apply_kda_gate", return_value=torch.zeros_like(q)) as mock_gate,
        patch.object(kda_triton, "chunk_kda", return_value=(q, None)),
    ):
        kda_triton.chunk_kda_with_fused_gate(q=q, k=q, v=q, raw_g=q, raw_beta=raw_beta, A_log=a_log)
        assert mock_gate.call_args.kwargs["safe_gate"] is False

        kda_triton.chunk_kda_with_fused_gate(q=q, k=q, v=q, raw_g=q, raw_beta=raw_beta, A_log=a_log, lower_bound=-4.0)
        assert mock_gate.call_args.kwargs["safe_gate"] is True
        assert mock_gate.call_args.kwargs["lower_bound"] == -4.0

        kda_triton.chunk_kda_with_fused_gate(q=q, k=q, v=q, raw_g=q, raw_beta=raw_beta, A_log=a_log, safe_gate=True)
        assert mock_gate.call_args.kwargs["safe_gate"] is True
        assert mock_gate.call_args.kwargs["lower_bound"] == kda_triton.DEFAULT_KDA_LOWER_BOUND


@dataclass(frozen=True)
class _FakeDevice:
    type: str
    index: int = 0


class _TrackedTensor(torch.Tensor):
    _fake_device: Any = None
    _forbid_tolist: bool = False
    _tolist_counter: list[int] | None = None
    _to_result: torch.Tensor | None = None

    @property
    def device(self) -> Any:  # type: ignore[override]
        if self._fake_device is not None:
            return self._fake_device
        return super().device

    def contiguous(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        res = super().contiguous(*args, **kwargs)
        if self._fake_device is not None and isinstance(res, torch.Tensor) and res is not self:
            wrapped = res.as_subclass(_TrackedTensor)
            wrapped._fake_device = self._fake_device
            return wrapped
        return res

    def to(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if self._to_result is not None:
            return self._to_result
        return super().to(*args, **kwargs)

    def tolist(self) -> Any:
        if self._forbid_tolist:
            raise AssertionError("unexpected D2H .tolist() sync")
        if self._tolist_counter is not None:
            self._tolist_counter[0] += 1
        return super().tolist()

    def item(self) -> Any:
        if self._forbid_tolist:
            raise AssertionError("unexpected D2H .item() sync")
        return super().item()


def _as_fake_npu(tensor: torch.Tensor, fake_device: Any | None = None) -> _TrackedTensor:
    wrapped = tensor.as_subclass(_TrackedTensor)
    wrapped._fake_device = fake_device if fake_device is not None else _FakeDevice(type="npu")
    return wrapped


def _make_tracked_tensor(
    data: Any,
    *,
    dtype: torch.dtype,
    fake_device: Any = None,
    forbid_tolist: bool = False,
    tolist_counter: list[int] | None = None,
    to_result: torch.Tensor | None = None,
) -> _TrackedTensor:
    tensor = torch.tensor(data, dtype=dtype).as_subclass(_TrackedTensor)
    tensor._fake_device = fake_device
    tensor._forbid_tolist = forbid_tolist
    tensor._tolist_counter = tolist_counter
    tensor._to_result = to_result
    return tensor


def test_slice_checkpoint_metadata_skips_tolist_without_host_lists() -> None:
    ckpt = SimpleNamespace(
        checkpoint_offsets=torch.tensor([64, 0, 128], dtype=torch.int32),
        state_indices=torch.tensor([11, 22, 33], dtype=torch.int64),
        offsets=None,
        query_lens=None,
    )
    keep_meta = _make_tracked_tensor([True, False, True], dtype=torch.bool, forbid_tolist=True)
    sliced = kda_checkpoint._slice_checkpoint_metadata(ckpt, keep_meta, 3)

    assert sliced is not ckpt
    assert torch.equal(sliced.checkpoint_offsets, torch.tensor([64, 128], dtype=torch.int32))
    assert torch.equal(sliced.state_indices, torch.tensor([11, 33], dtype=torch.int64))


def test_prepare_kda_checkpoint_prefill_inputs_avoids_device_tolist_and_evaluates_lazily() -> None:
    state_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    has_initial_state = torch.tensor([True, False, True], dtype=torch.bool)
    query_start_loc = torch.tensor([0, 64, 64, 192], dtype=torch.int32)
    base_ckpt = _DummyCheckpointMetadata(
        checkpoint_offsets=torch.tensor([64, 0, 128], dtype=torch.int32),
        state_indices=torch.tensor([11, 22, 33], dtype=torch.int64),
        offsets=[64, 0, 128],
        query_lens=(64, 0, 128),
    )

    fake_npu_device = _FakeDevice(type="npu")

    # 1) Pre-moved device bool mask with host cu_seqlens pair avoids tensor.tolist() completely.
    keep_meta = _make_tracked_tensor(
        [True, False, True],
        dtype=torch.bool,
        fake_device=fake_npu_device,
        forbid_tolist=True,
        to_result=_make_tracked_tensor([True, False, True], dtype=torch.bool, forbid_tolist=True),
    )
    chunk_meta = SimpleNamespace(
        keep_meta=keep_meta,
        cu_seqlens_host=(0, 64, 64, 192),
        cu_seqlens_kern=(0, 64, 192),
    )
    _, _, _, sliced_from_host, _ = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc,
        chunk_meta=chunk_meta,
        checkpoint=base_ckpt,
    )

    assert sliced_from_host.offsets == [64, 128]
    assert sliced_from_host.query_lens == (64, 128)

    # 2) CPU keep_meta preserved across .to(device) avoids calling .tolist() on the moved device tensor.
    state_indices_npu = _make_tracked_tensor([1, 2, 3], dtype=torch.int64, fake_device=fake_npu_device)
    query_start_loc_npu = _make_tracked_tensor([0, 64, 64, 192], dtype=torch.int32, fake_device=fake_npu_device)
    moved_keep = _make_tracked_tensor(
        [True, False, True],
        dtype=torch.bool,
        fake_device=fake_npu_device,
        forbid_tolist=True,
    )
    cpu_keep = _make_tracked_tensor([True, False, True], dtype=torch.bool, to_result=moved_keep)
    _, _, _, sliced_from_cpu_keep, _ = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices_npu,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc_npu,
        chunk_meta=SimpleNamespace(keep_meta=cpu_keep, cu_seqlens_kern=None),
        checkpoint=base_ckpt,
    )

    assert sliced_from_cpu_keep.offsets == [64, 128]
    assert sliced_from_cpu_keep.query_lens == (64, 128)

    # 3) Fallback without host cu_seqlens resolves device keep_meta.tolist() lazily at most once for both fields.
    tolist_counter = [0]
    fallback_keep = _make_tracked_tensor(
        [True, False, True],
        dtype=torch.bool,
        fake_device=fake_npu_device,
        tolist_counter=tolist_counter,
        to_result=_make_tracked_tensor([True, False, True], dtype=torch.bool, tolist_counter=tolist_counter),
    )
    _, _, _, sliced_lazy, _ = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        prefill_query_start_loc=query_start_loc,
        chunk_meta=SimpleNamespace(keep_meta=fallback_keep, cu_seqlens_kern=(0, 64, 192)),
        checkpoint=base_ckpt,
    )

    assert tolist_counter[0] == 1
    assert sliced_lazy.offsets == [64, 128]
    assert sliced_lazy.query_lens == (64, 128)


def test_kda_checkpoint_requires_explicit_opt_in_by_default() -> None:
    assert (
        kda_checkpoint.supports_kda_prefill_checkpoint(
            _valid_vllm_config(kda_internal_prefill_checkpoints=None),
            head_dim=128,
            conv_size=4,
            local_num_heads=8,
        )
        is False
    )
    assert (
        kda_checkpoint.supports_kda_prefill_checkpoint(
            _valid_vllm_config(kda_internal_prefill_checkpoints=False),
            head_dim=128,
            conv_size=4,
            local_num_heads=8,
        )
        is False
    )
    for falsy_raw in ("false", " FALSE ", "0", 0, 1, 2):
        assert (
            kda_checkpoint.supports_kda_prefill_checkpoint(
                _valid_vllm_config(kda_internal_prefill_checkpoints=falsy_raw),
                head_dim=128,
                conv_size=4,
                local_num_heads=8,
            )
            is False
        )
    for truthy_raw in ("true", " TRUE "):
        assert (
            kda_checkpoint.supports_kda_prefill_checkpoint(
                _valid_vllm_config(kda_internal_prefill_checkpoints=truthy_raw),
                head_dim=128,
                conv_size=4,
                local_num_heads=8,
            )
            is True
        )
    with patch(
        "vllm_ascend.ascend_config.get_ascend_config",
        return_value=SimpleNamespace(kda_internal_prefill_checkpoints=False),
    ):
        assert (
            kda_checkpoint.supports_kda_prefill_checkpoint(
                _valid_vllm_config(kda_internal_prefill_checkpoints=True),
                head_dim=128,
                conv_size=4,
                local_num_heads=8,
            )
            is False
        )
    with patch(
        "vllm_ascend.ascend_config.get_ascend_config",
        return_value=SimpleNamespace(kda_internal_prefill_checkpoints=True),
    ):
        assert (
            kda_checkpoint.supports_kda_prefill_checkpoint(
                _valid_vllm_config(kda_internal_prefill_checkpoints=None),
                head_dim=128,
                conv_size=4,
                local_num_heads=8,
            )
            is True
        )


def test_disabled_kda_checkpoint_clears_base_spec_and_allows_attn_builder_init() -> None:
    """Verify disabled KDA layers clear inherited non-zero MambaSpec fields so GDN builder init succeeds."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    from vllm_ascend.attention.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

    assert "num_prefill_checkpoint_blocks" in MambaSpec.__dataclass_fields__, (
        "KDA checkpoint tests expect MambaSpec with num_prefill_checkpoint_blocks"
    )

    configs = [
        SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_mode="none", block_size=256),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, use_ubatching=False),
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512),
            kv_transfer_config=None,
            speculative_config=None,
            num_hash_lookahead_tokens=0,
            additional_config={"kda_internal_prefill_checkpoints": True},
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_mode="align", block_size=256),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, use_ubatching=False),
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512),
            kv_transfer_config=object(),
            speculative_config=None,
            num_hash_lookahead_tokens=0,
            additional_config={"kda_internal_prefill_checkpoints": True},
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_mode="align", block_size=256),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, use_ubatching=False),
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=512),
            kv_transfer_config=None,
            speculative_config=None,
            num_hash_lookahead_tokens=0,
            additional_config={"kda_internal_prefill_checkpoints": False},
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
    ]

    for cfg in configs:
        layer: Any = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
        torch.nn.Module.__init__(layer)
        layer.head_dim = 128
        layer.conv_size = 4
        layer.local_num_heads = 4
        layer.kda_prefill_backend = "triton"
        layer.get_state_dtype = lambda: (torch.bfloat16, torch.bfloat16)

        with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
            enabled = kda_checkpoint.initialize_kda_checkpoints(layer, cfg)
        assert enabled is False
        assert layer._ascend_internal_checkpoint_enabled is False

        upstream_nonzero_spec = MambaSpec(
            block_size=256,
            shapes=((8, 3), (4, 128, 128)),
            dtypes=(torch.bfloat16, torch.bfloat16),
            mamba_cache_mode=cfg.cache_config.mamba_cache_mode,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=64,
        )
        base_zero_spec = replace(
            upstream_nonzero_spec,
            num_prefill_checkpoint_blocks=0,
            prefill_checkpoint_alignment=None,
        )
        with (
            patch.object(MambaBase, "get_kv_cache_spec", return_value=base_zero_spec),
            patch("vllm.models.kimi_k3.nvidia.kda.kda_prefill_checkpoint_alignment", return_value=64),
        ):
            resolved_spec = layer.get_kv_cache_spec(cfg)

        assert resolved_spec.num_prefill_checkpoint_blocks == 0
        assert resolved_spec.prefill_checkpoint_alignment is None

        with patch.object(KimiK3DeltaAttention, "get_kv_cache_spec", return_value=upstream_nonzero_spec):
            resolved_from_k3 = layer.get_kv_cache_spec(cfg)
        assert resolved_from_k3.num_prefill_checkpoint_blocks == 0
        assert resolved_from_k3.prefill_checkpoint_alignment is None

        def _stub_gdn_parent_init(
            builder_self: Any,
            spec: Any,
            names: list[str],
            vllm_cfg: Any,
            dev: torch.device,
        ) -> None:
            builder_self.vllm_config = vllm_cfg
            builder_self.decode_cudagraph_max_bs = 0
            builder_self._USE_COMMON_KERNEL_METADATA = False

        cfg.compilation_config.static_forward_context = {"layer.0": layer}
        with patch.object(GDNAttentionMetadataBuilder, "__init__", _stub_gdn_parent_init):
            with pytest.raises(ValueError, match="Ascend-owned checkpoint-capable layers"):
                AscendGDNAttentionMetadataBuilder(upstream_nonzero_spec, ["layer.0"], cfg, torch.device("cpu"))
            builder = AscendGDNAttentionMetadataBuilder(resolved_spec, ["layer.0"], cfg, torch.device("cpu"))
            builder_from_k3 = AscendGDNAttentionMetadataBuilder(resolved_from_k3, ["layer.0"], cfg, torch.device("cpu"))
        assert builder._ascend_gdn_checkpoints is False
        assert builder_from_k3._ascend_gdn_checkpoints is False


def test_chunk_delta_h_and_chunk_kda_input_guards_and_contiguity() -> None:
    fake_npu = _FakeDevice(type="npu")
    other_npu = _FakeDevice(type="npu", index=1)
    k_cpu = torch.zeros((1, 64, 2, 64), dtype=torch.float32)
    w_cpu = torch.zeros((1, 64, 2, 64), dtype=torch.float32)
    u_cpu = torch.zeros((1, 64, 2, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="must be on an NPU device"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(k=k_cpu, w=w_cpu, u=u_cpu)

    k_npu = _as_fake_npu(k_cpu, fake_npu)
    w_npu = _as_fake_npu(w_cpu, fake_npu)
    u_npu = _as_fake_npu(u_cpu, fake_npu)

    with pytest.raises(ValueError, match="same device"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(k=k_npu, w=_as_fake_npu(w_cpu, other_npu), u=u_npu)

    with pytest.raises(ValueError, match="cu_seqlens must be on"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k_npu,
            w=w_npu,
            u=u_npu,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
        )

    with pytest.raises(ValueError, match="cu_seqlens must have int32 or int64 dtype"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k_npu,
            w=w_npu,
            u=u_npu,
            cu_seqlens=_as_fake_npu(torch.tensor([0.0, 64.0], dtype=torch.float32), fake_npu),
        )

    valid_cu = _as_fake_npu(torch.tensor([0, 64], dtype=torch.int32), fake_npu)
    idx_stub = torch.zeros((1, 2), dtype=torch.int32)
    off_stub = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="checkpoint_state must have float16, bfloat16, or float32 dtype"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k_npu,
            w=w_npu,
            u=u_npu,
            cu_seqlens=valid_cu,
            chunk_indices=idx_stub,
            chunk_offsets=off_stub,
            checkpoint_state=_as_fake_npu(torch.zeros((1, 2, 64, 64), dtype=torch.int32), fake_npu),
            checkpoint_offsets=_as_fake_npu(torch.tensor([64], dtype=torch.int32), fake_npu),
        )

    with pytest.raises(ValueError, match="checkpoint_offsets must have int32 or int64 dtype"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k_npu,
            w=w_npu,
            u=u_npu,
            cu_seqlens=valid_cu,
            chunk_indices=idx_stub,
            chunk_offsets=off_stub,
            checkpoint_state=_as_fake_npu(torch.zeros((1, 2, 64, 64), dtype=torch.float32), fake_npu),
            checkpoint_offsets=_as_fake_npu(torch.tensor([64.0], dtype=torch.float32), fake_npu),
        )

    # Non-contiguous int64 cu_seqlens & checkpoint_offsets (N=2) are normalized before kernel dispatch.
    k_npu_2 = _as_fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32), fake_npu)
    w_npu_2 = _as_fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32), fake_npu)
    u_npu_2 = _as_fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32), fake_npu)
    non_contig_cu = _as_fake_npu(torch.tensor([0, 99, 64, 99, 128], dtype=torch.int64)[::2], fake_npu)
    non_contig_ckpt_off = _as_fake_npu(torch.tensor([64, 99, 0], dtype=torch.int64)[::2], fake_npu)
    assert not non_contig_cu.is_contiguous()
    assert not non_contig_ckpt_off.is_contiguous()

    fake_kernel = MagicMock()
    with patch.object(chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda", fake_kernel):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k_npu_2,
            w=w_npu_2,
            u=u_npu_2,
            cu_seqlens=non_contig_cu,
            chunk_indices=idx_stub,
            chunk_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
            checkpoint_state=_as_fake_npu(torch.zeros((2, 2, 64, 64), dtype=torch.bfloat16), fake_npu),
            checkpoint_offsets=non_contig_ckpt_off,
        )
    kernel_kwargs = fake_kernel.__getitem__.return_value.call_args.kwargs
    assert kernel_kwargs["cu_seqlens"].is_contiguous()
    assert kernel_kwargs["checkpoint_offsets"].is_contiguous()

    # chunk_kda also validates dtypes and normalizes contiguity before chunk_kda_fwd.
    small = _as_fake_npu(torch.ones((1, 64, 2, 16), dtype=torch.bfloat16), fake_npu)
    beta_small = _as_fake_npu(torch.ones((1, 64, 2), dtype=torch.bfloat16), fake_npu)
    with pytest.raises(ValueError, match="cu_seqlens must have int32 or int64 dtype"):
        kda_triton.chunk_kda(
            small,
            small,
            small,
            small,
            beta_small,
            cu_seqlens=torch.tensor([0.0, 64.0], dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="checkpoint_offsets must have int32 or int64 dtype"):
        kda_triton.chunk_kda(
            small,
            small,
            small,
            small,
            beta_small,
            checkpoint_state=torch.zeros((1, 2, 16, 16), dtype=torch.bfloat16),
            checkpoint_offsets=torch.tensor([False], dtype=torch.bool),
        )
    with patch.object(kda_triton, "chunk_kda_fwd", return_value=(small, None)) as mock_fwd:
        kda_triton.chunk_kda(
            small,
            small,
            small,
            small,
            beta_small,
            cu_seqlens=non_contig_cu,
            checkpoint_state=_as_fake_npu(torch.zeros((1, 2, 16, 16), dtype=torch.float32), fake_npu),
            checkpoint_offsets=non_contig_ckpt_off,
        )
    fwd_kwargs = mock_fwd.call_args.kwargs
    assert fwd_kwargs["cu_seqlens"].is_contiguous()
    assert fwd_kwargs["checkpoint_offsets"].is_contiguous()
