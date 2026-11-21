# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.ops import kda_checkpoint
from vllm_ascend.ops.triton.kda import chunk_delta_h
from vllm_ascend.ops.triton.kda import kda as kda_triton


def _valid_vllm_config(*, mamba_cache_mode: str = "align", pcp_size: int = 1, use_ubatching: bool = False):
    return SimpleNamespace(
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
    k = torch.zeros((1, 128, 2, 64), dtype=torch.float32)
    w = torch.zeros((1, 128, 2, 64), dtype=torch.float32)
    u = torch.zeros((1, 128, 2, 64), dtype=torch.float32)
    cu = torch.tensor([0, 128], dtype=torch.int32)

    with pytest.raises(ValueError, match="provided together"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k,
            w=w,
            u=u,
            cu_seqlens=cu,
            chunk_indices=torch.zeros((2, 2), dtype=torch.int32),
            chunk_offsets=torch.tensor([0, 2], dtype=torch.int32),
            checkpoint_state=torch.zeros((1, 2, 64, 64)),
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
            checkpoint_state=torch.zeros((2, 2, 64, 64)),
            checkpoint_offsets=torch.tensor([64], dtype=torch.int32),
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
