# SPDX-License-Identifier: Apache-2.0
"""vLLM release/main output construction shared by copied schedulers."""

from __future__ import annotations

from typing import Any
from unittest.mock import NonCallableMock

from vllm.v1.core.sched.output import CachedRequestData

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.34.0"):
    from vllm.v1.core.sched.output import IterationDetails

_DEFAULT_ASYNC_SCHEDULING_DEPTH = 2
_MIN_IN_FLIGHT_OUTPUT_TOKENS = 1


def require_non_bool_int(value: Any, name: str = "value") -> int:
    """Convert ``value`` to ``int`` while rejecting ``bool`` instances."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    return int(value)


def _read_config_attr(obj: Any, attr: str) -> Any:
    """Read ``attr`` from a config object while treating unconfigured mocks as unset."""
    value = None if obj is None else getattr(obj, attr, None)
    return None if isinstance(value, NonCallableMock) else value


def _resolve_pp_size(parallel_cfg: Any) -> int:
    """Resolve ``pipeline_parallel_size`` with ``bool`` rejection."""
    pp_raw = _read_config_attr(parallel_cfg, "pipeline_parallel_size")
    return 1 if pp_raw is None else max(1, require_non_bool_int(pp_raw, "pipeline_parallel_size"))


def resolve_max_concurrent_batches(vllm_config: Any) -> int:
    """Resolve ``max_concurrent_batches`` across release, main, and test configs.

    When async scheduling is enabled on vLLM builds with configurable
    ``async_scheduling_depth``, computes ``pp_size + async_scheduling_depth - 1``.
    When ``async_scheduling`` is disabled, ignores ``async_scheduling_depth`` and
    returns ``pp_size``. On older vLLM builds without
    ``SchedulerConfig.async_scheduling_depth``, falls back smoothly to
    ``pp_size + 1`` (or explicit ``vllm_config.max_concurrent_batches``).
    """
    if vllm_config is None:
        return 1

    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    async_enabled = _read_config_attr(sched_cfg, "async_scheduling")
    if async_enabled is True:
        pp_size = _resolve_pp_size(parallel_cfg)
        depth = _read_config_attr(sched_cfg, "async_scheduling_depth")
        if depth is not None:
            resolved_depth = max(
                _DEFAULT_ASYNC_SCHEDULING_DEPTH,
                require_non_bool_int(depth, "async_scheduling_depth"),
            )
            return pp_size + resolved_depth - 1
        base_depth = pp_size + _DEFAULT_ASYNC_SCHEDULING_DEPTH - 1
        configured = _read_config_attr(vllm_config, "max_concurrent_batches")
        if configured is not None:
            return max(base_depth, require_non_bool_int(configured, "max_concurrent_batches"))
        return base_depth

    if async_enabled is False or (sched_cfg is not None and not isinstance(sched_cfg, NonCallableMock)):
        return _resolve_pp_size(parallel_cfg)

    configured = _read_config_attr(vllm_config, "max_concurrent_batches")
    if configured is not None:
        return max(1, require_non_bool_int(configured, "max_concurrent_batches"))
    return _resolve_pp_size(parallel_cfg)


def min_in_flight_output_tokens(scheduler: Any, request: Any) -> int:
    """Return minimum guaranteed output tokens from in-flight async steps.

    Delegates to upstream ``Scheduler._min_in_flight_output_tokens`` when
    present, and falls back safely on legacy vLLM releases that lack
    ``Request.num_output_placeholder_steps``.
    """
    upstream_fn = getattr(scheduler, "_min_in_flight_output_tokens", None)
    if callable(upstream_fn) and not (
        isinstance(scheduler, NonCallableMock) and isinstance(upstream_fn, NonCallableMock)
    ):
        result = upstream_fn(request)
        if not isinstance(result, NonCallableMock):
            return require_non_bool_int(result, "_min_in_flight_output_tokens()")
    placeholder_steps = _read_config_attr(request, "num_output_placeholder_steps")
    if placeholder_steps is not None:
        steps = require_non_bool_int(placeholder_steps, "num_output_placeholder_steps")
        raw = _read_config_attr(scheduler, "num_sampled_tokens_per_step")
        per_step = (
            _MIN_IN_FLIGHT_OUTPUT_TOKENS if raw is None else require_non_bool_int(raw, "num_sampled_tokens_per_step")
        )
        return max(_MIN_IN_FLIGHT_OUTPUT_TOKENS, steps * per_step)
    return _MIN_IN_FLIGHT_OUTPUT_TOKENS


def make_cached_request_data(
    scheduler: Any,
    running_reqs: list,
    num_scheduled_tokens: dict[str, int],
    spec_decode_tokens: dict[str, list[int]],
    req_to_new_blocks: dict,
    iteration_details: IterationDetails | None = None,
) -> CachedRequestData:
    """Construct cached-request data using the upstream version's signature."""
    if vllm_version_is("0.34.0"):
        # v0.34.0 has no iteration-details argument yet: that landed on main
        # after the release branch was cut.
        return scheduler._make_cached_request_data(
            running_reqs, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks
        )

    return scheduler._make_cached_request_data(
        running_reqs, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks, iteration_details=iteration_details
    )
