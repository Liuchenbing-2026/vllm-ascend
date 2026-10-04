# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.compilation.acl_graph import ACLGraphEntry, ACLGraphWrapper


@pytest.fixture
def replay():
    config = SimpleNamespace(
        compilation_config=MagicMock(),
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
    )
    ascend = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(enable_super_kernel=False, enable_replay_event_sync=True)
    )
    descriptor = BatchDescriptor(num_tokens=8, uniform=True)
    context = SimpleNamespace(batch_descriptor=descriptor, cudagraph_runtime_mode=CUDAGraphMode.FULL, attn_metadata={})
    calls = MagicMock()
    stream, event, graph = calls.stream, calls.event, calls.graph
    with (
        patch("vllm_ascend.compilation.acl_graph.get_ascend_config", return_value=ascend),
        patch("vllm_ascend.compilation.acl_graph.current_platform"),
        patch("vllm_ascend.compilation.acl_graph.get_forward_context", return_value=context),
        patch("vllm_ascend.compilation.acl_graph.use_updatable_graph", return_value=True),
        patch("vllm_ascend.compilation.acl_graph._EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
        patch("vllm_ascend.compilation.acl_graph.torch.npu.current_stream", return_value=stream),
        patch("vllm_ascend.compilation.acl_graph.torch.npu.Event", return_value=event),
    ):
        wrapper = ACLGraphWrapper(MagicMock(), config, CUDAGraphMode.FULL, update_stream=MagicMock())
        wrapper.is_debugging_mode = False
        wrapper.concrete_aclgraph_entries[descriptor] = ACLGraphEntry(descriptor, graph, "result")
        yield wrapper, context, calls


def test_waits_previous_completion_before_parameter_update(replay):
    wrapper, _, calls = replay
    assert wrapper() == "result"
    names = [c[0] for c in calls.mock_calls]
    assert names.index("stream.synchronize") < names.index("graph.replay")
    assert names.index("graph.update") < names.index("event.record")
    calls.reset_mock()
    wrapper()
    names = [c[0] for c in calls.mock_calls]
    assert names.index("event.synchronize") < names.index("graph.replay") < names.index("graph.update")
    assert names.index("graph.update") < names.index("event.record")
    calls.stream.synchronize.assert_not_called()


def test_shape_change_falls_back_to_full_stream_wait(replay):
    wrapper, context, calls = replay
    wrapper()
    calls.reset_mock()
    descriptor = BatchDescriptor(num_tokens=16, uniform=True)
    wrapper.concrete_aclgraph_entries[descriptor] = ACLGraphEntry(descriptor, calls.graph, "result")
    context.batch_descriptor = descriptor
    wrapper()
    calls.stream.synchronize.assert_called_once()
    calls.event.synchronize.assert_not_called()


def test_eager_interleaving_invalidates_previous_event(replay):
    wrapper, context, calls = replay
    wrapper()
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    wrapper()
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    calls.reset_mock()
    wrapper()
    calls.stream.synchronize.assert_called_once()
    calls.event.synchronize.assert_not_called()


@pytest.mark.parametrize("disabled_by", ["flag", "speculative", "tp", "pp", "enpu"])
def test_excluded_paths_do_not_record_completion_event(replay, disabled_by):
    wrapper, _, calls = replay
    if disabled_by == "flag":
        wrapper.enable_replay_event_sync = False
    elif disabled_by == "speculative":
        wrapper.vllm_config.speculative_config = object()
    elif disabled_by in ("tp", "pp"):
        field = "tensor_parallel_size" if disabled_by == "tp" else "pipeline_parallel_size"
        setattr(wrapper.vllm_config.parallel_config, field, 2)
    else:
        wrapper.enable_enpu = True
    wrapper()
    calls.event.record.assert_not_called()
    calls.event.synchronize.assert_not_called()
    if disabled_by != "enpu":
        calls.stream.synchronize.assert_called_once()


def test_failed_update_cannot_reuse_stale_event(replay):
    wrapper, _, calls = replay
    wrapper()
    calls.graph.update.side_effect = RuntimeError("failed update")
    with pytest.raises(RuntimeError, match="failed update"):
        wrapper()
    assert wrapper._last_replay_descriptor is None
