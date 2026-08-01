# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from collections.abc import Iterator
from contextlib import contextmanager

from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_ascend.ascend_forward_context import override_mrv2_forward_inputs


@contextmanager
def draft_forward_inputs(input_buffers: InputBuffers) -> Iterator[None]:
    """Announce a draft forward to the Ascend forward-context hook.

    Every speculator's draft forward runs inside the announcement
    `NPUModelRunner.execute_model` makes for the target, which names the
    target's id buffer. Re-announcing here is what keeps a hash-routed MoE gate
    or an id-consuming expert selector from silently reading the target's tokens
    during a draft forward, so it has to wrap every speculator's model call, not
    just the autoregressive ones.

    `input_buffers` is the drafter's own buffer set; the hook slices the id
    buffer down to the token count of the forward it is building for, which is
    exactly what the speculators pass to the model.
    """
    with override_mrv2_forward_inputs(input_buffers.input_ids, is_draft_model=True):
        yield
