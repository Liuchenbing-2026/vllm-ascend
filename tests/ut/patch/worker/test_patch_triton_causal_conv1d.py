# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from unittest.mock import MagicMock

import torch
import vllm.model_executor.layers.mamba.ops.causal_conv1d as _cc1d

from vllm_ascend.ops.triton.mamba import causal_conv1d as npu_cc1d
from vllm_ascend.patch.worker import patch_triton


class _KernelRecorder:
    """Minimal Triton kernel stand-in that records grid and launch arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[object, ...], dict[str, object]]] = []

    def __getitem__(self, grid: tuple[int, ...]):
        def _launch(*args: object, **kwargs: object) -> None:
            self.calls.append((grid, args, kwargs))

        return _launch


def test_causal_conv1d_update_npu_is_exported():
    assert "causal_conv1d_update_npu" in npu_cc1d.__all__
    assert callable(npu_cc1d.causal_conv1d_update_npu)


def test_patch_triton_rebinds_causal_conv1d_update(monkeypatch):
    stale_update = MagicMock()
    monkeypatch.setattr(_cc1d, "causal_conv1d_update", stale_update)

    importlib.reload(patch_triton)

    assert _cc1d.causal_conv1d_update is npu_cc1d.causal_conv1d_update_npu
    stale_update.assert_not_called()


def test_causal_conv1d_update_npu_2d_decode_without_indices(monkeypatch):
    recorder = _KernelRecorder()
    monkeypatch.setattr(npu_cc1d, "_causal_conv1d_update_kernel", recorder)

    x = torch.randn(3, 8, dtype=torch.bfloat16)
    conv_state = torch.randn(4, 8, 3, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)

    out = npu_cc1d.causal_conv1d_update_npu(x, conv_state, weight, activation="silu")

    assert len(recorder.calls) == 1
    grid, args, kwargs = recorder.calls[0]
    assert grid == (3, 1)
    torch.testing.assert_close(args[4], torch.arange(3, dtype=torch.int32))
    assert args[8] == 3
    assert args[9] == 8
    assert args[10] == 1
    assert kwargs["HAS_NULL_BLOCK"] is False
    assert kwargs["IS_VARLEN"] is False
    assert kwargs["IS_SPEC_DECODING"] is False
    assert kwargs["SILU_ACTIVATION"] is True
    assert kwargs["launch_pdl"] is False
    assert out.shape == (3, 8)
    assert out.dtype == torch.bfloat16


def test_causal_conv1d_update_npu_varlen_spec_dim_last_layout(monkeypatch):
    recorder = _KernelRecorder()
    monkeypatch.setattr(npu_cc1d, "_causal_conv1d_update_kernel", recorder)

    x = torch.randn(5, 16, dtype=torch.bfloat16)
    conv_state = torch.randn(8, 5, 16, dtype=torch.bfloat16)
    weight = torch.randn(4, 16, dtype=torch.bfloat16)
    conv_state_indices = torch.tensor([2, 0], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    out = torch.empty_like(x)

    ret = npu_cc1d.causal_conv1d_update_npu(
        x,
        conv_state,
        weight,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=3,
        null_block_id=0,
        out=out,
    )

    assert ret is out
    assert len(recorder.calls) == 1
    grid, args, kwargs = recorder.calls[0]
    assert grid == (2, 1)
    assert args[8] == 2
    assert args[9] == 16
    assert args[10] == 3
    assert args[11] == 5
    assert args[12] == 8
    assert args[13:16] == (0, 1, 16)
    assert args[16:18] == (1, 16)
    assert args[18:21] == (80, 1, 16)
    assert args[22:25] == (0, 1, 16)
    assert args[25] == 0
    assert kwargs["IS_VARLEN"] is True
    assert kwargs["IS_SPEC_DECODING"] is True
    assert kwargs["HAS_NULL_BLOCK"] is True
    assert kwargs["NP2_STATELEN"] == 8
    assert kwargs["launch_pdl"] is False


def test_causal_conv1d_update_npu_pad_slot_id_and_empty_batch(monkeypatch):
    recorder = _KernelRecorder()
    monkeypatch.setattr(npu_cc1d, "_causal_conv1d_update_kernel", recorder)

    x = torch.randn(2, 8, dtype=torch.bfloat16)
    conv_state = torch.randn(4, 8, 3, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    indices = torch.tensor([1, -1], dtype=torch.int32)

    npu_cc1d.causal_conv1d_update_npu(
        x,
        conv_state,
        weight,
        conv_state_indices=indices,
        null_block_id=0,
        pad_slot_id=-1,
    )

    assert len(recorder.calls) == 1
    _, args, kwargs = recorder.calls[0]
    assert args[25] == -1
    assert kwargs["HAS_NULL_BLOCK"] is True
    assert kwargs["launch_pdl"] is False

    recorder.calls.clear()
    empty_x = torch.empty(0, 8, dtype=torch.bfloat16)
    empty_out = npu_cc1d.causal_conv1d_update_npu(empty_x, conv_state, weight)
    assert len(recorder.calls) == 0
    assert empty_out.shape == (0, 8)
    assert empty_out.dtype == torch.bfloat16
