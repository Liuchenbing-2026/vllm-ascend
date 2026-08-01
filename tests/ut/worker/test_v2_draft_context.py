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

    A speculator reaches its draft forward with no announcement in scope: the
    only one `NPUModelRunner` makes covers `execute_model`, and upstream drives
    `propose` from `sample_tokens`. So a speculator that does not announce for
    itself publishes the `(None, False)` default, and `input_ids=None` is not a
    value an id-consuming expert selector can use. DFlash and DSpark do not
    derive from AscendAutoRegressiveSpeculator, so each needs its own
    `_run_model` override; dropping any of them makes these tests fail.
    """

    def _build_speculator(self, speculator_cls, draft_ids):
        speculator = speculator_cls.__new__(speculator_cls)
        speculator.input_buffers = SimpleNamespace(input_ids=draft_ids)
        # AscendAutoRegressiveSpeculator._run_model calls this after the model;
        # it needs the real attention metadata, which this test does not build.
        speculator._ascend_update_seq_lens = lambda attn_metadata: None
        return speculator

    def _assert_announces_draft_ids(self, speculator_cls, base_cls, base_result):
        draft_ids = torch.arange(100, 108, dtype=torch.int32)
        speculator = self._build_speculator(speculator_cls, draft_ids)

        announced_during_forward = {}

        def _record_announcement(self, *args, **kwargs):
            announced_during_forward["value"] = get_mrv2_forward_inputs()
            return base_result

        with patch.object(base_cls, "_run_model", _record_announcement):
            speculator._run_model(4, None, None, None)
            announced_after_forward = get_mrv2_forward_inputs()

        announced_ids, is_draft_model = announced_during_forward["value"]
        self.assertIs(announced_ids, draft_ids)
        self.assertTrue(is_draft_model)

        # Nothing is in scope around a draft forward, so the announcement must
        # unwind to the default rather than leak into the next forward.
        self.assertEqual(announced_after_forward, (None, False))

    def _assert_overrides_a_stale_announcement(self, speculator_cls, base_cls, base_result):
        """Worst case only: a target announcement that is somehow still live.

        This is not the production call stack -- `execute_model` has returned by
        the time a speculator runs -- but the drafter's ids have to win over any
        outer announcement, and the outer one has to be restored on exit.
        """
        draft_ids = torch.arange(100, 108, dtype=torch.int32)
        target_ids = torch.arange(8, dtype=torch.int32)
        speculator = self._build_speculator(speculator_cls, draft_ids)

        announced_during_forward = {}

        def _record_announcement(self, *args, **kwargs):
            announced_during_forward["value"] = get_mrv2_forward_inputs()
            return base_result

        with (
            patch.object(base_cls, "_run_model", _record_announcement),
            override_mrv2_forward_inputs(target_ids),
        ):
            speculator._run_model(4, None, None, None)
            announced_after_forward = get_mrv2_forward_inputs()

        announced_ids, is_draft_model = announced_during_forward["value"]
        self.assertIs(announced_ids, draft_ids)
        self.assertTrue(is_draft_model)

        restored_ids, restored_is_draft = announced_after_forward
        self.assertIs(restored_ids, target_ids)
        self.assertFalse(restored_is_draft)

    def test_dflash_overrides_a_live_outer_announcement(self):
        self._assert_overrides_a_stale_announcement(
            AscendDFlashSpeculator,
            DFlashSpeculator,
            base_result=torch.zeros(4, 2),
        )

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
