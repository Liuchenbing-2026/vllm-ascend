# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.attention_v1 import FIAParamProvider
from vllm_ascend.attention.utils import fia_physical_block_table, fia_physical_kv_views


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("offset", [0, 17])
def test_physical_views_preserve_selected_pages_and_storage(factor, offset):
    blocks, page, width = 5, 8, 4
    storage = torch.arange(offset + blocks * factor * page * width + page * width)
    key = storage.as_strided((blocks, page, width), (factor * page * width, width, 1), offset)
    value = storage.as_strided(key.shape, key.stride(), offset + page * width)
    physical_key, physical_value, stride = fia_physical_kv_views(key, value)
    ids = torch.tensor([4, 1, 3, 0])
    assert stride == factor
    for source, physical in [(key, physical_key), (value, physical_value)]:
        assert physical.is_contiguous()
        assert physical.data_ptr() == source.data_ptr()
        assert physical.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        torch.testing.assert_close(physical[ids * stride], source[ids])


def test_non_ndh_inner_layout_is_not_reinterpreted():
    key = torch.zeros(4, 8, 16)[..., ::2]
    out_key, out_value, stride = fia_physical_kv_views(key, key)
    assert out_key is key and out_value is key and stride == 1


def test_block_addresses_are_cached_per_metadata_and_keep_sentinels():
    first = SimpleNamespace(block_tables=torch.tensor([[2, -1, 0]], dtype=torch.int32))
    converted = fia_physical_block_table(first, 2)
    torch.testing.assert_close(converted, torch.tensor([[4, -1, 0]], dtype=torch.int32))
    assert fia_physical_block_table(first, 2) is converted
    assert fia_physical_block_table(first, 1) is first.block_tables
    second = SimpleNamespace(block_tables=torch.tensor([[3, 1, 0]], dtype=torch.int32))
    assert fia_physical_block_table(second, 2) is not converted
    torch.testing.assert_close(fia_physical_block_table(second, 2), second.block_tables * 2)


def test_graph_provider_resolves_current_step_physical_addresses():
    provider = FIAParamProvider("layer", None, block_stride=2)
    for table in ([[3, 1]], [[2, 0]]):
        metadata = SimpleNamespace(block_tables=torch.tensor(table), actual_seq_lengths_q=[2], seq_lens_list=[133])
        params = provider.resolve({"layer": metadata})
        torch.testing.assert_close(params["block_table"], metadata.block_tables * 2)
        assert params["actual_seq_lengths_kv"] == [133]
