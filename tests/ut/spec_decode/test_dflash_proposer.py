#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for the DFlash proposer query buffers.

ACL graph dispatch can select a padded execution size that is larger than the
logical query limit ``max_batch_size * (1 + num_speculative_tokens)``. The
persistent query buffers must therefore be sized for the configured graph
padding range, otherwise slicing them with the padded token count silently
returns a shorter tensor than the draft model input.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

_MAX_NUM_TOKENS = 256
_NUM_SPECULATIVE_TOKENS = 3
_HIDDEN_SIZE = 8


@pytest.mark.parametrize(
    ("max_num_seqs", "max_capture_size", "num_input_tokens"),
    [
        pytest.param(33, 136, 136, id="padded-query"),
        pytest.param(32, 128, 128, id="capture-aligned-query"),
        pytest.param(33, 132, 132, id="equal-capacity"),
        pytest.param(33, 128, 132, id="query-exceeds-capture"),
        pytest.param(33, None, 132, id="unset-capture-size"),
        pytest.param(33, 0, 132, id="disabled-capture"),
    ],
)
def test_query_buffers_cover_execution_shape(max_num_seqs, max_capture_size, num_input_tokens):
    """Graph padding must not silently truncate query positions or slots."""
    device = torch.device("cpu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_capture_size),
    )

    def init_base(self, vllm_config, device, runner=None):
        self.max_batch_size = max_num_seqs
        self.num_speculative_tokens = _NUM_SPECULATIVE_TOKENS
        self.max_num_tokens = _MAX_NUM_TOKENS
        self.hidden_size = _HIDDEN_SIZE
        self.dtype = torch.float32
        self.device = device

    with patch.object(AscendEagleProposer, "__init__", init_base):
        proposer = AscendDflashProposer(config, device)

    logical_query_tokens = max_num_seqs * (1 + _NUM_SPECULATIVE_TOKENS)
    padded_query_tokens = max(logical_query_tokens, max_capture_size or 0)

    # Graph padding must not silently truncate the persistent query buffers.
    for buffer in (proposer.positions, proposer._slot_mapping_buffer):
        assert buffer[:num_input_tokens].shape == (num_input_tokens,)

    # Padding changes storage capacity, not the logical query or context limits.
    assert proposer.max_query_tokens == logical_query_tokens
    assert proposer.max_padded_query_tokens == padded_query_tokens
    assert proposer.max_positions == _MAX_NUM_TOKENS + padded_query_tokens
    assert proposer.arange_dflash.shape == (proposer.max_positions + 1,)
    assert proposer._context_positions_buffer.shape == (_MAX_NUM_TOKENS,)
    assert proposer._context_slot_mapping_buffer.shape == (_MAX_NUM_TOKENS,)
    assert proposer._dflash_hidden_states.shape == (_MAX_NUM_TOKENS, _HIDDEN_SIZE)

    for buffer in (proposer.positions, proposer._slot_mapping_buffer):
        assert buffer.shape == (padded_query_tokens,)
        assert buffer.dtype == torch.int32
        assert buffer.device == device
