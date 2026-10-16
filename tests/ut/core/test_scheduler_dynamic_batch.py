# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.core import scheduler_dynamic_batch as sdb
from vllm_ascend.core.scheduler_dynamic_batch import SchedulerDynamicBatch

# Keyword set that vllm/v1/engine/core.py passes when it builds the scheduler.
# Kept in sync with ``vllm/v1/engine/core.py`` in the pinned vLLM version.
ENGINE_SCHEDULER_KWARGS = dict(
    vllm_config=None,
    kv_cache_config=None,
    structured_output_manager=None,
    include_finished_set=False,
    log_stats=False,
    block_size=16,
    hash_block_size=16,
)


def test_dynamic_batch_signature_binds_engine_kwargs():
    """EngineCore binds ``hash_block_size``; the signature must accept it.

    Regression test for the v0.23.0 failure where
    ``SchedulerDynamicBatch.__init__() got an unexpected keyword argument
    'hash_block_size'`` aborted EngineCore at construction time.
    """
    signature = inspect.signature(SchedulerDynamicBatch.__init__)
    # Raises TypeError if any engine keyword (notably hash_block_size) is missing.
    signature.bind(None, **ENGINE_SCHEDULER_KWARGS)

    assert "hash_block_size" in signature.parameters


def test_dynamic_batch_forwards_hash_block_size_to_super():
    """The accepted ``hash_block_size`` must reach ``Scheduler.__init__``."""
    captured = {}

    def fake_super_init(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Minimal state consumed by the remainder of SchedulerDynamicBatch.__init__.
        self.scheduler_config = SimpleNamespace(
            max_num_batched_tokens=2048,
            SLO_limits_for_dynamic_batch=-1,
        )

    with (
        patch.object(sdb.Scheduler, "__init__", fake_super_init),
        patch.object(sdb, "BudgetRefiner", MagicMock()),
    ):
        SchedulerDynamicBatch(
            vllm_config=MagicMock(),
            kv_cache_config=MagicMock(),
            structured_output_manager=MagicMock(),
            block_size=32,
            hash_block_size=64,
            include_finished_set=False,
            log_stats=False,
        )

    assert captured["kwargs"]["hash_block_size"] == 64
    # Positional ``block_size`` keeps its old behaviour.
    assert captured["args"][3] == 32
