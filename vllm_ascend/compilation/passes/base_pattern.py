#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

_registered_patterns: set[str] = set()


class BasePattern:
    """Base class for FX graph fusion patterns.

    Provides a standard interface for pattern matching and replacement
    registration with both torch inductor and torchair backends.
    """

    def __init__(self, vllm_config, eps=1e-6):
        self.vllm_config = vllm_config
        self.eps = eps

    def get_inputs(self):
        raise NotImplementedError

    def get_pattern(self):
        raise NotImplementedError

    def get_replacement(self):
        raise NotImplementedError

    def get_extra_stream_scope_check(self):
        return lambda match: True

    def register(self, pm_pass):
        import torch._inductor.pattern_matcher as pm
        import torchair

        pattern_id = f"{self.__class__.__name__}_{self.eps}"
        if pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        pm.register_replacement(
            pattern_fn, replacement_fn, example_inputs,
            pm.fwd_only, pm_pass)

        torchair.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )

        _registered_patterns.add(pattern_id)
