# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import array
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.patch.worker.patch_v2.patch_uva import (
    UvaBufferWrapper,
    _copy_chunks_to_uva,
    _int64_storage,
    _zeros_on_npu,
)


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_chunk_copy_updates_fallback_device_buffer(dtype):
    buf = UvaBufferWrapper.__new__(UvaBufferWrapper)
    buf._cpu = torch.zeros(4, dtype=dtype)
    buf._np = buf._cpu.numpy()
    buf._uva = torch.full((4,), -1, dtype=dtype)
    buf._use_real_uva = False
    buf._modified_indices = set()
    pool = SimpleNamespace(_next_buf=lambda n: buf)

    first = _copy_chunks_to_uva(pool, [np.array([4, 5]), array.array("i", [6])], 3)
    assert first.tolist() == [4, 5, 6]
    assert buf._uva.tolist() == [4, 5, 6, -1]
    assert not buf._modified_indices
    second = _copy_chunks_to_uva(pool, [np.array([], dtype=np.int32), np.array([8, 9])], 2)
    assert second.tolist() == [8, 9]
    assert buf._uva.tolist() == [8, 9, 6, -1]


def test_uint64_npu_zeros_view_int64_storage(monkeypatch):
    viewed = {}

    class _FakeStorage:
        def view(self, dtype):
            viewed["dtype"] = dtype
            return "uint64-view"

    def _zeros(*args, **kwargs):
        viewed["zeros"] = (args, kwargs)
        return _FakeStorage()

    monkeypatch.setattr(torch, "zeros", _zeros)
    cpu = torch.empty(4, dtype=torch.uint64)

    assert _zeros_on_npu(cpu) == "uint64-view"
    assert viewed["zeros"] == ((cpu.shape,), {"dtype": torch.int64, "device": "npu"})
    assert viewed["dtype"] == torch.uint64


def test_supported_dtype_uses_zeros_like(monkeypatch):
    monkeypatch.setattr(torch, "zeros_like", lambda cpu, device: ("zeros_like", device, cpu.dtype))
    cpu = torch.empty(2, dtype=torch.int32)

    assert _zeros_on_npu(cpu) == ("zeros_like", "npu", torch.int32)


def test_uint64_int64_storage_keeps_pointer_bits():
    high_bit = torch.tensor([0, 2**63 + 3], dtype=torch.uint64)
    storage = torch.zeros(2, dtype=torch.int64)
    viewed = storage.view(torch.uint64)

    _int64_storage(viewed).copy_(_int64_storage(high_bit))

    assert viewed.tolist() == high_bit.tolist()
    assert _int64_storage(torch.zeros(2, dtype=torch.int32)).dtype == torch.int32
