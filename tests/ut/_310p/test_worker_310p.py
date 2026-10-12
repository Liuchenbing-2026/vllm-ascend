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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend._310p.worker_310p import NPUWorker310


def _make_profile_context(free_memory: int) -> MagicMock:
    """Fake ``memory_profiling`` context whose post-profile free memory is fixed."""
    profile_result = MagicMock()
    profile_result.non_torch_increase = 1000
    profile_result.torch_peak_increase = 9000
    profile_result.weights_memory = 500
    profile_result.non_kv_cache_memory = 500
    profile_result.after_profile = SimpleNamespace(free_memory=free_memory)
    context = MagicMock()
    context.__enter__ = MagicMock(return_value=profile_result)
    context.__exit__ = MagicMock(return_value=False)
    return context


class TestNPUWorker310MemoryProfiling(TestBase):
    def _make_worker(self) -> NPUWorker310:
        worker = NPUWorker310.__new__(NPUWorker310)
        worker.init_snapshot = SimpleNamespace(free_memory=8500, total_memory=10000)
        worker.requested_memory = 8000
        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = 500
        worker._scale_kv_cache_memory_for_multi_group = MagicMock(side_effect=lambda value: value)
        return worker

    @patch("vllm_ascend._310p.worker_310p.is_rc_device")
    @patch("vllm_ascend._310p.worker_310p.torch.npu.memory_reserved")
    @patch("vllm_ascend._310p.worker_310p.torch.npu.mem_get_info")
    @patch("vllm_ascend._310p.worker_310p.memory_profiling")
    def test_accepts_unchanged_free_memory(
        self, mock_memory_profiling, mock_mem_get_info, mock_memory_reserved, mock_is_rc_device
    ):
        """An unchanged free-memory snapshot (e.g. a HAMi vNPU quota) is not an error."""
        mock_memory_profiling.return_value = _make_profile_context(8500)
        mock_mem_get_info.return_value = (0, 0)
        mock_memory_reserved.return_value = 0
        mock_is_rc_device.return_value = False

        worker = self._make_worker()

        # non_torch_memory_cleared_by_empty_cache = 0 - 1000 = -1000
        # available = (8000 - 500 - (-1000)) // 2 = 4250
        self.assertEqual(worker.determine_available_memory(), 4250)

    @patch("vllm_ascend._310p.worker_310p.is_rc_device")
    @patch("vllm_ascend._310p.worker_310p.torch.npu.memory_reserved")
    @patch("vllm_ascend._310p.worker_310p.torch.npu.mem_get_info")
    @patch("vllm_ascend._310p.worker_310p.memory_profiling")
    def test_rejects_increased_free_memory(
        self, mock_memory_profiling, mock_mem_get_info, mock_memory_reserved, mock_is_rc_device
    ):
        """More free memory after profiling means another process released memory."""
        mock_memory_profiling.return_value = _make_profile_context(9000)
        mock_mem_get_info.return_value = (0, 0)
        mock_memory_reserved.return_value = 0
        mock_is_rc_device.return_value = False

        worker = self._make_worker()

        with self.assertRaises(AssertionError) as cm:
            worker.determine_available_memory()

        self.assertIn("Error in memory profiling", str(cm.exception))
