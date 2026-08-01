# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""CPU-only tests for the target-forward announcements the v2 runner makes."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import get_mrv2_forward_inputs
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager, ModelWithContext
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class TestExecuteModelAnnouncement(TestBase):
    """Upstream opens the forward context deep inside execute_model.

    The platform hook that fills in the Ascend fields sees only batch-shaped
    arguments, so the id buffer has to be announced around the whole call.
    Without the override the hook publishes the ContextVar default, and
    `input_ids=None` is not a value an id-consuming expert selector can use.
    """

    @staticmethod
    def _runner(target_ids):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.input_buffers = SimpleNamespace(input_ids=target_ids)
        return runner

    def test_announces_the_target_ids_around_the_base_call(self):
        target_ids = torch.arange(16, dtype=torch.int32)
        runner = self._runner(target_ids)
        observed = {}

        def _record(self, scheduler_output, **kwargs):
            observed["announcement"] = get_mrv2_forward_inputs()
            observed["scheduler_output"] = scheduler_output
            observed["kwargs"] = kwargs
            return "model-runner-output"

        with patch.object(GPUModelRunner, "execute_model", _record):
            output = runner.execute_model("scheduler-output", dummy_run=True, is_profile=True)
            announced_after = get_mrv2_forward_inputs()

        announced_ids, is_draft_model = observed["announcement"]
        self.assertIs(announced_ids, target_ids)
        # The target, not a drafter: an attention impl keys on this flag.
        self.assertFalse(is_draft_model)
        self.assertEqual(output, "model-runner-output")
        # The announcement must not outlive the call it was made for.
        self.assertEqual(announced_after, (None, False))

    def test_forwards_every_flag_to_the_base_by_keyword(self):
        # _dummy_run, profile_run and warmup all funnel through here, and each
        # is distinguished only by these flags; binding them positionally would
        # silently transpose them if upstream reorders the signature.
        runner = self._runner(torch.arange(16, dtype=torch.int32))
        observed = {}

        def _record(self, scheduler_output, **kwargs):
            observed["scheduler_output"] = scheduler_output
            observed["kwargs"] = kwargs
            return None

        with patch.object(GPUModelRunner, "execute_model", _record):
            runner.execute_model(
                "scheduler-output",
                intermediate_tensors="tensors",
                dummy_run=True,
                skip_attn_for_dummy_run=True,
                is_profile=True,
            )

        self.assertEqual(observed["scheduler_output"], "scheduler-output")
        self.assertEqual(
            observed["kwargs"],
            {
                "intermediate_tensors": "tensors",
                "dummy_run": True,
                "skip_attn_for_dummy_run": True,
                "is_profile": True,
            },
        )


class TestAclGraphCaptureAnnouncement(TestBase):
    """Capture builds its own forward contexts, one per graph size.

    They reach the same platform hook as a real step, so the hook needs the same
    announcement it gets from NPUModelRunner.execute_model -- which is not in
    scope here, because capture runs from initialize_kv_cache, not a forward.
    """

    def _capture(self, input_ids):
        manager = ModelAclGraphManager.__new__(ModelAclGraphManager)
        observed = {}

        def _record(self, model, *args, **kwargs):
            observed["announcement"] = get_mrv2_forward_inputs()
            observed["model"] = model
            return "captured"

        with patch.object(ModelCudaGraphManager, "capture", _record):
            observed["result"] = manager.capture(
                nn.Identity(),
                "model-state",
                SimpleNamespace(input_ids=input_ids),
                None,
                "block-tables",
                [],
                "kv-cache-config",
            )
            observed["announced_after"] = get_mrv2_forward_inputs()
        return observed

    def test_announces_the_input_ids_around_the_capture(self):
        input_ids = torch.arange(16, dtype=torch.int32)

        observed = self._capture(input_ids)

        announced_ids, is_draft_model = observed["announcement"]
        self.assertIs(announced_ids, input_ids)
        self.assertFalse(is_draft_model)
        self.assertEqual(observed["result"], "captured")
        self.assertEqual(observed["announced_after"], (None, False))

    def test_wraps_the_model_for_the_capture(self):
        observed = self._capture(torch.arange(16, dtype=torch.int32))

        # The wrapper is what marks the forward context as capturing; the base
        # manager must never see the bare model.
        self.assertIsInstance(observed["model"], ModelWithContext)
        self.assertIsInstance(observed["model"].get_original_model(), nn.Identity)
