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
"""CPU-only tests for AscendBlockTables' zero-KV-cache-group guards.

An attention-free model (encoder-only pooling) owns no KV cache group, which
makes the first grid dimension of both Triton launches below zero. CANN rejects
that with EE1003 where CUDA no-ops it, so neither launch may be reached.
"""

from unittest.mock import patch

import torch
from vllm.v1.worker.gpu.block_table import BlockTables

from tests.ut.base import TestBase
from vllm_ascend.worker.v2.block_table import AscendBlockTables

_BLOCK_TABLE_MODULE = "vllm_ascend.worker.v2.block_table"


class _RefusingKernel:
    """Stands in for a Triton kernel that must not be launched."""

    def __init__(self):
        self.launched_grids = []

    def __getitem__(self, grid):
        self.launched_grids.append(grid)
        raise AssertionError(f"kernel launched with grid {grid}")


def _make_block_tables(num_kv_cache_groups, max_num_batched_tokens=8, max_num_reqs=4, max_num_blocks=2):
    """Build an AscendBlockTables without touching a device.

    __init__ allocates NPU tensors and pointer tables, none of which the guarded
    branches read.
    """
    block_tables = AscendBlockTables.__new__(AscendBlockTables)
    block_tables.num_kv_cache_groups = num_kv_cache_groups
    block_tables.input_block_tables = [
        torch.zeros(max_num_reqs, max_num_blocks, dtype=torch.int32) for _ in range(num_kv_cache_groups)
    ]
    block_tables.slot_mappings = torch.zeros(num_kv_cache_groups, max_num_batched_tokens, dtype=torch.int32)
    block_tables.cp_rank = 0
    block_tables.cp_size = 1
    block_tables.cp_interleave = 1
    block_tables.block_table_ptrs = None
    block_tables.block_table_strides = None
    block_tables.block_sizes_tensor = None
    return block_tables


class TestGatherBlockTablesWithoutKVCacheGroups(TestBase):
    def test_attention_free_model_skips_the_gather_launch(self):
        block_tables = _make_block_tables(0)

        with patch.object(BlockTables, "gather_block_tables", side_effect=AssertionError("kernel launched")):
            gathered = block_tables.gather_block_tables(torch.arange(2, dtype=torch.int32), num_reqs_padded=2)

        self.assertEqual(gathered, ())

    def test_one_group_still_reaches_the_gather_launch(self):
        block_tables = _make_block_tables(1)
        sentinel = (torch.zeros(2, 2, dtype=torch.int32),)

        with patch.object(BlockTables, "gather_block_tables", return_value=sentinel) as gather:
            gathered = block_tables.gather_block_tables(torch.arange(2, dtype=torch.int32), num_reqs_padded=2)

        self.assertIs(gathered, sentinel)
        gather.assert_called_once()


class TestComputeSlotMappingsWithoutKVCacheGroups(TestBase):
    @staticmethod
    def _inputs(num_reqs=2):
        return (
            torch.arange(num_reqs, dtype=torch.int32),
            torch.zeros(num_reqs + 1, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        )

    def test_attention_free_model_skips_the_slot_mapping_launch(self):
        block_tables = _make_block_tables(0)
        idx_mapping, query_start_loc, positions = self._inputs()

        with patch(f"{_BLOCK_TABLE_MODULE}._compute_slot_mappings_kernel", _RefusingKernel()):
            slot_mappings = block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                num_tokens_padded=4,
            )

        # The empty group dimension is preserved: callers still index per group.
        self.assertEqual(tuple(slot_mappings.shape), (0, 4))

    def test_one_group_still_reaches_the_slot_mapping_launch(self):
        block_tables = _make_block_tables(1)
        idx_mapping, query_start_loc, positions = self._inputs()
        refusing_kernel = _RefusingKernel()

        with (
            patch(f"{_BLOCK_TABLE_MODULE}._compute_slot_mappings_kernel", refusing_kernel),
            self.assertRaises(AssertionError),
        ):
            block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                num_tokens_padded=4,
            )

        self.assertEqual(refusing_kernel.launched_grids, [(1, 3)])
