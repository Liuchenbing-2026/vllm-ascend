# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real NPU storage sharing, layout rejection and repeated consumer exit."""

import gc
import multiprocessing
import traceback

import pytest
import torch
import torch_npu
from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def _consume(entries, expected, connection):
    try:
        torch.npu.set_device(0)
        for entry, reference in zip(entries, expected):
            tensor = entry.rebuild(0)
            torch.testing.assert_close(tensor.cpu(), reference, atol=0, rtol=0)
            tensor.add_(1)
            torch.npu.synchronize()
            del tensor
        gc.collect()
        connection.send(None)
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        connection.close()


def test_npu_weight_cache_survives_consumer_restarts():
    torch.npu.set_device(0)
    tensors = [torch.arange(128, device="npu", dtype=dtype).reshape(8, 16) for dtype in (torch.float32, torch.bfloat16)]
    tensors.append(torch.arange(128, device="npu").reshape(8, 16)[1:5, 2:9])
    context = multiprocessing.get_context("spawn")
    for _ in range(2):
        expected = [tensor.cpu() for tensor in tensors]
        entries = [TensorEntry.from_tensor(tensor, "param") for tensor in tensors]
        assert all(entry.cpu_tensor is None and entry.ipc_args is not None for entry in entries)
        parent, child = context.Pipe()
        process = context.Process(target=_consume, args=(entries, expected, child))
        process.start()
        child.close()
        try:
            assert parent.poll(180), "consumer did not finish"
            error = parent.recv()
            assert error is None, error
            process.join(30)
            assert process.exitcode == 0
            for tensor, reference in zip(tensors, expected):
                torch.testing.assert_close(tensor.cpu(), reference + 1, atol=0, rtol=0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(30)
            parent.close()


def test_npu_weight_cache_rejects_cpu_tensor():
    with pytest.raises(ValueError, match="Unsupported weight cache device"):
        NPUPlatform.reduce_weight_cache_tensor(torch.zeros(2))


def test_npu_weight_cache_rejects_nz():
    torch.npu.set_device(0)
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    try:
        torch.npu.config.allow_internal_format = True
        tensor = torch_npu.npu_format_cast(
            torch.ones(32, 32, device="npu", dtype=torch.bfloat16), ACL_FORMAT_FRACTAL_NZ
        )
        assert torch_npu.get_npu_format(tensor) == ACL_FORMAT_FRACTAL_NZ
        with pytest.raises(ValueError, match="requires ND tensors"):
            NPUPlatform.reduce_weight_cache_tensor(tensor)
    finally:
        torch.npu.config.allow_internal_format = previous == b"enable"
