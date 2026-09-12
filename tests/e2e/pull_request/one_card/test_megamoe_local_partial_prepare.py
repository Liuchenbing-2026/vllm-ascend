# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend.ops.triton.megamoe_prepare import prepare_cann_megamoe_local_partial


def _inputs(tokens, probability_dtype, mask_dtype):
    hidden = torch.randn((tokens, 2048), dtype=torch.bfloat16, device="npu")
    ids = torch.randint(0, 256, (tokens, 8), dtype=torch.int32, device="npu")
    weights = torch.rand((tokens, 8), dtype=probability_dtype, device="npu")
    mask = None
    if mask_dtype is not None:
        mask = (torch.arange(tokens, device="npu") % 3 != 0).to(mask_dtype)
    return hidden, ids, weights, mask


def _assert_matches_reference(inputs, actual, preapply_active_mask=False):
    expected = _append_cann_megamoe_dummy_tokens(*inputs, 256, 0, 1)
    assert actual[4] == expected[4]
    for index in range(4):
        reference = expected[index].float() if index == 2 else expected[index]
        if preapply_active_mask and index == 3:
            assert actual[index] is None
            continue
        if preapply_active_mask and index == 1 and inputs[3] is not None:
            reference = reference.clone()
            reference[: actual[4]].masked_fill_(~inputs[3].bool().unsqueeze(-1), 256)
        assert actual[index].dtype == reference.dtype
        assert torch.equal(actual[index], reference)


@pytest.mark.parametrize("tokens", [1, 7, 32, 513, 8192])
@pytest.mark.parametrize("preapply_active_mask", [False, True])
@pytest.mark.parametrize(
    "probability_dtype,mask_dtype",
    [(torch.bfloat16, None), (torch.bfloat16, torch.bool), (torch.bfloat16, torch.int8), (torch.float32, None)],
)
def test_local_partial_preparation_exact(tokens, probability_dtype, mask_dtype, preapply_active_mask=False):
    inputs = _inputs(tokens, probability_dtype, mask_dtype)
    snapshots = tuple(None if x is None else x.clone() for x in inputs)
    actual = prepare_cann_megamoe_local_partial(*inputs, 256, 0, 1, preapply_active_mask=preapply_active_mask)
    _assert_matches_reference(inputs, actual, preapply_active_mask)
    for source, snapshot in zip(inputs, snapshots):
        if source is not None:
            assert torch.equal(source, snapshot)


@pytest.mark.parametrize("preapply_active_mask", [False, True])
def test_local_partial_preparation_queued_outputs_own_storage(preapply_active_mask=False):
    inputs = [_inputs(513, torch.bfloat16, torch.int8) for _ in range(8)]
    outputs = [
        prepare_cann_megamoe_local_partial(*values, 256, 0, 1, preapply_active_mask=preapply_active_mask)
        for values in inputs
    ]
    torch.npu.synchronize()
    for values, actual in zip(inputs, outputs):
        _assert_matches_reference(values, actual, preapply_active_mask)
    for index in range(3 if preapply_active_mask else 4):
        assert len({values[index].data_ptr() for values in outputs}) == len(outputs)
    # CANN's active-mask handling is allowed to modify its routing output.
    outputs[0][1].fill_(-1)
    for values, actual in zip(inputs[1:], outputs[1:]):
        _assert_matches_reference(values, actual, preapply_active_mask)
