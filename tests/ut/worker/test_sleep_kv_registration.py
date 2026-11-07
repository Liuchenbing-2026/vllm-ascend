# SPDX-License-Identifier: Apache-2.0
"""Level-1 sleep keeps the KV cache, so it must not unregister it (#21165)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.worker.worker import NPUWorker


def _worker() -> NPUWorker:
    with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
        worker = NPUWorker()
    worker.model_runner = MagicMock()
    worker.model_runner.model.named_buffers.return_value = []
    worker.model_config = SimpleNamespace(sleep_preserve_parameter_names=[])
    worker.sleep_wakeup_manager = MagicMock()
    return worker


@pytest.mark.parametrize("level", [1, 2])
@patch("vllm_ascend.worker.worker.NPUWorker.synchronize_device")
@patch("vllm_ascend.worker.worker.torch.npu.mem_get_info", return_value=(100, 200))
@patch("vllm_ascend.worker.worker.CaMemAllocator")
@patch("vllm_ascend.worker.worker.global_te")
@patch("vllm_ascend.worker.worker.get_ascend_config")
def test_sleep_only_unregisters_when_kv_cache_is_offloaded(
    mock_get_config, mock_global_te, mock_allocator_class, mock_mem_get_info, mock_synchronize_device, level
):
    mock_get_config.return_value = SimpleNamespace(
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False)
    )

    _worker().sleep(level=level)

    if level == 1:
        mock_global_te.unregister_buffer.assert_not_called()
    else:
        mock_global_te.unregister_buffer.assert_called_once_with()
