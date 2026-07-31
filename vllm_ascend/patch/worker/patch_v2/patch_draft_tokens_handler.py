# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

_original_set_draft_tokens = DraftTokensHandler.set_draft_tokens


def set_draft_tokens(self, input_batch, draft_tokens):
    """Slice drafts to the dynamic speculative length before scheduling.

    With dynamic speculative decoding the Ascend model runner stores the
    per-step K on this handler (``dynamic_num_spec_tokens``). The placeholder
    width recorded here decides how many spec tokens the scheduler lets the
    next step verify, so slicing is what makes the dynamic K take effect.
    """
    num_spec_tokens = getattr(self, "dynamic_num_spec_tokens", None)
    if num_spec_tokens is not None and num_spec_tokens < draft_tokens.shape[1]:
        draft_tokens = draft_tokens[:, :num_spec_tokens]
    return _original_set_draft_tokens(self, input_batch, draft_tokens)


DraftTokensHandler.set_draft_tokens = set_draft_tokens
