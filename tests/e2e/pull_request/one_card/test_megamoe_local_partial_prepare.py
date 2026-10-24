# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

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
    hidden, ids, weights, mask = inputs
    tokens = hidden.shape[0]
    expected_ids = ids.clone()
    if preapply_active_mask and mask is not None:
        expected_ids.masked_fill_(~mask.bool().unsqueeze(-1), 256)
    assert actual[4] == tokens
    assert actual[0].data_ptr() == hidden.data_ptr(), "hidden states should not be copied"
    assert actual[0].shape == (tokens, 2048)
    assert actual[1].shape == (tokens, 8)
    assert actual[1].dtype == torch.int32
    assert actual[1].data_ptr() != ids.data_ptr(), "CANN must not mutate the caller's routing IDs"
    assert torch.equal(actual[1], expected_ids)
    assert actual[2].dtype == torch.float32
    assert torch.equal(actual[2], weights.float())
    if preapply_active_mask:
        assert actual[3] is None
    else:
        expected_mask = torch.ones(tokens, dtype=torch.int8, device="npu") if mask is None else mask.to(torch.int8)
        assert actual[3].dtype == torch.int8
        assert torch.equal(actual[3], expected_mask)


@pytest.mark.parametrize("tokens", [1, 7, 32, 513, 8192])
@pytest.mark.parametrize("preapply_active_mask", [False, True])
@pytest.mark.parametrize(
    "probability_dtype,mask_dtype",
    [(torch.bfloat16, None), (torch.bfloat16, torch.bool), (torch.bfloat16, torch.int8), (torch.float32, None)],
)
def test_local_partial_preparation_exact(tokens, probability_dtype, mask_dtype, preapply_active_mask):
    inputs = _inputs(tokens, probability_dtype, mask_dtype)
    snapshots = tuple(None if x is None else x.clone() for x in inputs)
    actual = prepare_cann_megamoe_local_partial(*inputs, 256, 0, 1, preapply_active_mask=preapply_active_mask)
    _assert_matches_reference(inputs, actual, preapply_active_mask)
    for source, snapshot in zip(inputs, snapshots):
        if source is not None:
            assert torch.equal(source, snapshot)


def test_local_partial_preparation_bool_mask_preapply_has_cann_contract():
    inputs = _inputs(513, torch.bfloat16, torch.bool)
    original_ids = inputs[1].clone()
    actual = prepare_cann_megamoe_local_partial(*inputs, 256, 0, 1, preapply_active_mask=True)
    _assert_matches_reference(inputs, actual, preapply_active_mask=True)
    assert actual[3] is None, "preapplied masks must not reach CANN as DT_BOOL"
    assert torch.equal(inputs[1], original_ids)


@pytest.mark.parametrize("preapply_active_mask", [False, True])
def test_local_partial_preparation_queued_outputs_own_routing_storage(preapply_active_mask):
    inputs = [_inputs(513, torch.bfloat16, torch.int8) for _ in range(8)]
    snapshots = [values[1].clone() for values in inputs]
    outputs = [
        prepare_cann_megamoe_local_partial(*values, 256, 0, 1, preapply_active_mask=preapply_active_mask)
        for values in inputs
    ]
    torch.npu.synchronize()
    for values, snapshot, actual in zip(inputs, snapshots, outputs):
        _assert_matches_reference(values, actual, preapply_active_mask)
        assert torch.equal(values[1], snapshot)
    assert len({values[1].data_ptr() for values in outputs}) == len(outputs)
    # CANN's active-mask handling is allowed to modify its routing output.
    outputs[0][1].fill_(-1)
    for values, actual in zip(inputs[1:], outputs[1:]):
        _assert_matches_reference(values, actual, preapply_active_mask)
