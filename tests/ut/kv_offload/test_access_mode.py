# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Behavioral unit tests for KV offload access_mode support."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.core import kv_cache_manager, kv_cache_utils
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload import base, factory
from vllm.v1.request import RequestStatus

from tests.ut.kv_offload.utils import create_request, create_vllm_config
from vllm_ascend.distributed.kv_transfer import register_connector
from vllm_ascend.distributed.kv_transfer.kv_pool import access_mode
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import npu, offloading_connector
from vllm_ascend.utils import vllm_version_is

am, kcm, kcu, kv_base, kv_fac = access_mode, kv_cache_manager, kv_cache_utils, base, factory
npu_mod, oc_mod = npu, offloading_connector


def _cfgs(extra: dict[str, Any]) -> tuple[Any, KVCacheConfig]:
    if "AscendOffloadingConnector" not in KVConnectorFactory._registry:
        register_connector()
    for n, c in (("CPUOffloadingSpec", "NPUOffloadingSpec"), ("TieringOffloadingSpec", "NPUTieringOffloadingSpec")):
        if n not in kv_fac.OffloadingSpecFactory._registry:
            kv_fac.OffloadingSpecFactory.register_spec(n, npu_mod.__name__, c)
    kc = KVTransferConfig(
        kv_connector="AscendOffloadingConnector",
        kv_role="kv_both",
        engine_id=f"u-{uuid4().hex}",
        kv_connector_extra_config=extra,
    )
    sp = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    return create_vllm_config(4, 256, 16, kv_transfer_config=kc), KVCacheConfig(16, [], [KVCacheGroupSpec(["l0"], sp)])


class _DummyTier(npu_mod.SecondaryTierManager):
    medium, locality = kv_base.Medium.STORAGE, kv_base.Locality.LOCAL

    def __init__(self, spec: Any, view: memoryview, tier_type: str, mode: am.OffloadAccessMode, hit: set[Any]) -> None:
        super().__init__(spec, view, tier_type)
        self._mode, self._hit, self.access_mode, self.lookups, self.stores = mode, hit, mode, [], []

    def lookup(self, key: kv_base.OffloadKey, req_context: kv_base.ReqContext) -> kv_base.LookupResult:
        self.lookups.append(key)
        return kv_base.LookupResult.HIT if key in self._hit else kv_base.LookupResult.MISS

    submit_store = lambda self, job: self.stores.append(list(job.keys))
    submit_load = on_request_finished = reset_cache = shutdown = lambda *_: None
    get_finished_jobs = drain_jobs = lambda *_: ()
    on_new_request = lambda *_: kv_base.RequestOffloadingContext(kv_base.OffloadPolicy.CHUNK_LEVEL)


def test_access_mode_specs_and_tiering(monkeypatch: pytest.MonkeyPatch) -> None:
    assert am.parse_access_mode(None) is am.OffloadAccessMode.READWRITE
    for bad in ("", "ro", True, 1):
        with pytest.raises(ValueError, match="access_mode"):
            am.parse_access_mode({"access_mode": bad})
    monkeypatch.delattr(npu_mod.SecondaryTierManager, "access_mode", raising=False)
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", spy := MagicMock(wraps=npu_mod.SharedOffloadRegion))
    mk = lambda c: kv_fac.OffloadingSpecFactory.create_spec(build_offloading_config(*_cfgs(c))).get_manager()
    sp = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 65536}
    with pytest.raises(ValueError, match="access_mode"):
        mk({**sp, "secondary_tiers": [{"type": "fs", "access_mode": "ro"}]})
    spy.assert_not_called()
    hk, sk = kv_base.make_offload_key(b"1".ljust(32, b"0"), 0), kv_base.make_offload_key(b"2".ljust(32, b"0"), 0)
    kw = {} if vllm_version_is("0.34.0") else {"trigger": kv_base.StoreTrigger.ON_COMPUTE}
    for mode in (am.OffloadAccessMode.READ, am.OffloadAccessMode.WRITE, am.OffloadAccessMode.READWRITE):
        tiers: list[_DummyTier] = []
        mk_t = lambda c, v, s, m=mode, ts=tiers: ts.append(_DummyTier(s, v, "d", m, {hk})) or ts[-1]
        monkeypatch.setattr(npu_mod.tiering_spec.SecondaryTierFactory, "create_secondary_tier", mk_t)
        mgr = mk({**sp, "tier_poll_interval_s": 0.0, "secondary_tiers": [{"type": "fs", "access_mode": mode.value}]})
        ctx = kv_base.ReqContext(req_id=f"r-{mode.value}")
        mgr.on_new_request(ctx)
        assert mgr._should_store_to_tier(tiers[0], 1) is mode.can_write and mgr.prepare_store([sk], ctx, **kw)
        mgr.complete_store([sk], ctx, success=True, **kw)
        assert (mgr.lookup(hk, ctx) in (kv_base.LookupResult.HIT_PENDING, kv_base.LookupResult.RETRY)) is mode.can_read
        assert tiers[0].stores == ([[sk]] if mode.can_write else [])
        assert tiers[0].lookups == ([hk] if mode.can_read else [])
        mgr.shutdown()


def test_ascend_offloading_connector_access_mode() -> None:
    kw = {} if vllm_version_is("0.34.0") else {"trigger": kv_base.StoreTrigger.ON_COMPUTE}
    for mode in ("readwrite", "read", "write"):
        vc, kc = _cfgs({"cpu_bytes_to_use": 131072, "access_mode": mode})
        conn = oc_mod.AscendOffloadingConnector(vc, KVConnectorRole.SCHEDULER, kc)
        assert (sched := conn.connector_scheduler) is not None
        up = mode == "readwrite" or am.has_upstream_access_mode()
        assert type(sched) is (OffloadingConnectorScheduler if up else oc_mod.AscendOffloadingConnectorScheduler)
        probe, wreq = create_request(10, 32, 16), create_request(11, 16, 16)
        skey = [kv_base.make_offload_key(bytes(probe.block_hashes[0]), 0)]
        sched.manager.prepare_store(skey, kv_base.ReqContext("seed"), **kw)
        sched.manager.complete_store(skey, kv_base.ReqContext("seed"), success=True, **kw)
        conn.on_new_request(probe)
        assert conn.get_num_new_matched_tokens(probe, 0)[0] == (0 if mode == "write" else 16)
        wblk = kcm.KVCacheBlocks(([kcu.KVCacheBlock(3)],))
        conn.on_new_request(wreq)
        conn.get_num_new_matched_tokens(wreq, 0)
        conn.update_state_after_alloc(wreq, wblk, 0)
        out = SchedulerOutput.make_empty()
        out.scheduled_new_reqs = [NewRequestData.from_request(wreq, wblk.get_block_ids(), wreq.prompt_token_ids)]
        out.num_scheduled_tokens = {wreq.request_id: wreq.num_tokens}
        assert (len(conn.build_connector_meta(out).store_jobs) > 0) is (mode != "read")
        if mode == "read":
            pre, fin = SchedulerOutput.make_empty(), SchedulerOutput.make_empty()
            pre.preempted_req_ids, fin.finished_req_ids = {wreq.request_id}, {wreq.request_id}
            assert conn.build_connector_meta(pre).store_jobs == {}
            wreq.status = RequestStatus.FINISHED_ABORTED
            conn.build_connector_meta(fin)
            assert wreq.request_id not in sched._req_status
        conn.shutdown()
