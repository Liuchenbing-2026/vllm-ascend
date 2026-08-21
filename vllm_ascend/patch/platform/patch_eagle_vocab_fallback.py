#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""EAGLEConfig vocab_size fallback for Qwen3.5/3.8-style configs.

The Qwen3.8 family config (``model_type=qwen3_5``) stores the vocabulary size
inside ``text_config.vocab_size`` and has no top-level ``vocab_size``. The
dspark-backport vLLM's ``EAGLEConfig.__init__`` reads ``self.model.vocab_size``
directly, which crashes MTP/EAGLE spec-config construction for these models.
This patch materializes ``vocab_size`` from ``text_config`` before the original
constructor runs.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.transformers_utils.configs.eagle import EAGLEConfig

_ORIGINAL_EAGLE_INIT = EAGLEConfig.__init__


def _materialize_vocab_size(model: Any) -> None:
    if model is None or hasattr(model, "vocab_size"):
        return
    text_config = getattr(model, "text_config", None)
    if isinstance(text_config, dict):
        vocab_size = text_config.get("vocab_size")
    else:
        vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is not None:
        model.vocab_size = vocab_size


@wraps(_ORIGINAL_EAGLE_INIT)
def _eagle_init_with_vocab_fallback(
    self: EAGLEConfig,
    *args: Any,
    **kwargs: Any,
) -> None:
    model = args[0] if args else kwargs.get("model")
    _materialize_vocab_size(model)
    _ORIGINAL_EAGLE_INIT(self, *args, **kwargs)


EAGLEConfig.__init__ = _eagle_init_with_vocab_fallback
