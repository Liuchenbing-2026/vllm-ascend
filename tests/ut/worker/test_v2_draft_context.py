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
"""CPU-only tests for the draft-forward announcement each v2 speculator makes."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import get_mrv2_forward_inputs, override_mrv2_forward_inputs
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class TestSpeculatorDraftAnnouncement(TestBase):
    """Every v2 speculator must announce its own ids around the draft forward.

    A speculator that does not re-announce runs its draft forward under the
    announcement `NPUModelRunner.execute_model` made for the target, so the
    forward context publishes the target's token ids and `is_draft_model=False`
    to an id-consuming expert selector during a draft forward. DFlash and DSpark
    do not derive from AscendAutoRegressiveSpeculator, so each needs its own
    `_run_model` override; dropping any of them makes these tests fail.
    """

    def _assert_announces_draft_ids(self, speculator_cls, base_cls, base_result):
        draft_ids = torch.arange(100, 108, dtype=torch.int32)
        target_ids = torch.arange(8, dtype=torch.int32)

        speculator = speculator_cls.__new__(speculator_cls)
        speculator.input_buffers = SimpleNamespace(input_ids=draft_ids)
        # AscendAutoRegressiveSpeculator._run_model calls this after the model;
        # it needs the real attention metadata, which this test does not build.
        speculator._ascend_update_seq_lens = lambda attn_metadata: None

        announced_during_forward = {}

        def _record_announcement(self, *args, **kwargs):
            announced_during_forward["value"] = get_mrv2_forward_inputs()
            return base_result

        with (
            patch.object(base_cls, "_run_model", _record_announcement),
            # The target's announcement wraps the whole of execute_model, so a
            # draft forward always runs nested inside it.
            override_mrv2_forward_inputs(target_ids),
        ):
            speculator._run_model(4, None, None, None)
            announced_after_forward = get_mrv2_forward_inputs()

        announced_ids, is_draft_model = announced_during_forward["value"]
        self.assertIs(announced_ids, draft_ids)
        self.assertTrue(is_draft_model)

        # The target's announcement is restored once the draft forward ends.
        restored_ids, restored_is_draft = announced_after_forward
        self.assertIs(restored_ids, target_ids)
        self.assertFalse(restored_is_draft)

    def test_dflash_announces_its_own_ids(self):
        self._assert_announces_draft_ids(
            AscendDFlashSpeculator,
            DFlashSpeculator,
            base_result=torch.zeros(4, 2),
        )

    def test_dspark_announces_its_own_ids(self):
        # DSpark derives from DFlash upstream but not on the Ascend side, so it
        # does not inherit AscendDFlashSpeculator's override.
        self._assert_announces_draft_ids(
            AscendDSparkSpeculator,
            DFlashSpeculator,
            base_result=torch.zeros(4, 2),
        )

    def test_autoregressive_announces_its_own_ids(self):
        self._assert_announces_draft_ids(
            AscendAutoRegressiveSpeculator,
            AutoRegressiveSpeculator,
            base_result=(torch.zeros(4, 2), torch.zeros(4, 2)),
        )
