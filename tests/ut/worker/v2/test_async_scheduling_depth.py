# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as _NS
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.async_utils import AsyncOutput

from vllm_ascend.core.scheduler_output import min_in_flight_output_tokens, resolve_max_concurrent_batches
from vllm_ascend.kv_trace.kv_trace_plan import AccessPlan, PlanSnapshotter
from vllm_ascend.kv_trace.kv_trace_runtime import KVTraceRuntime
from vllm_ascend.observability.runtime_guard.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.observability.runtime_guard.hooks import AscendAsyncOutput, maybe_wrap_v2_async_output
from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.observability.runtime_guard.state import MAX_DEFERRED_REAP_WAVES, RequestGuardStore


def _apply_av_stubs(monkeypatch: pytest.MonkeyPatch, *, generic: bool = False) -> tuple[Any, Any, Any]:
    import vllm_ascend.patch.worker.patch_adaptive_verification as patch_av

    def _slot(*shape: int, **_kw: Any) -> Any:
        c = torch.zeros(shape)
        return _NS(cpu=c, gpu=(g := c.clone()), np=c.numpy(), copy_to_cpu=partial(c.copy_, g))

    def _ev(*_a: Any, **_kw: Any) -> MagicMock:
        return MagicMock()

    av: Any = ModuleType("adaptive_verification")
    av.CpuGpuBuffer, av.stream, av.gpu_sync_allowed = _slot, (lambda *_: nullcontext()), nullcontext
    cu = _NS(Event=_ev, current_stream=lambda d: MagicMock(device=d))
    av.torch = _NS(float32=torch.float32, Event=_ev, accelerator=cu, cuda=cu)

    class _Legacy:
        def __init__(self, rs: Any, _qsl: Any, _nb: int, _mtl: int) -> None:
            self.req_states, self.num_speculative_steps, self._copy_stream = rs, rs.num_speculative_steps, MagicMock()
            self._confidence_probs = torch.ones((rs.max_num_reqs, rs.num_speculative_steps))
            self._stale_confidences = [_slot(rs.max_num_reqs, rs.num_speculative_steps) for _ in range(2)]
            self._copy_events = [_ev(blocking=True) for _ in range(2)]
            self._pending_resets: list[int] = []
            self._stale_idx = 0

        def add_request(self, idx: int) -> None:
            self._stale_confidences[self._stale_idx].np[idx].fill(1.0)
            self._pending_resets.append(idx)

    src = av.torch.accelerator if generic else av.torch.cuda
    _Legacy.record_confidences = lambda self, _p, _b: src.current_stream(self.req_states.device)  # type: ignore[attr-defined]
    av.AdaptiveVerificationManager = _Legacy
    av.maybe_create_adaptive_verification_manager = lambda en, _a, *a, **_kw: _Legacy(*a) if en else None
    mr: Any = ModuleType("vllm.v1.worker.gpu.model_runner")
    mr.maybe_create_adaptive_verification_manager = av.maybe_create_adaptive_verification_manager
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.spec_decode.adaptive_verification", av)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", mr)
    monkeypatch.setattr(patch_av, "adaptive_verification", av)
    patch_av._apply_confidence_ring_patch()
    return patch_av, av, mr


def test_resolve_max_concurrent_batches_and_in_flight_bounds() -> None:
    assert (resolve_max_concurrent_batches(None), resolve_max_concurrent_batches(Mock())) == (1, 1)
    assert resolve_max_concurrent_batches(_NS(max_concurrent_batches=5)) == 5
    for pp, on, d, exp in [(1, True, 2, 2), (1, True, 4, 4), (2, True, 6, 7), (2, False, 6, 2), (2, True, None, 3)]:
        sc = _NS(async_scheduling=on) if d is None else _NS(async_scheduling=on, async_scheduling_depth=d)
        pc = _NS(pipeline_parallel_size=pp)
        assert resolve_max_concurrent_batches(_NS(scheduler_config=sc, parallel_config=pc)) == exp
    assert min_in_flight_output_tokens(_NS(_min_in_flight_output_tokens=lambda r: r.n * 2), _NS(n=3)) == 6
    assert min_in_flight_output_tokens(_NS(num_sampled_tokens_per_step=1), _NS(num_output_placeholder_steps=3)) == 3
    assert min_in_flight_output_tokens(_NS(num_sampled_tokens_per_step=0), _NS(num_output_placeholder_steps=4)) == 1
    assert min_in_flight_output_tokens(Mock(), Mock()) == 1
    with pytest.raises(TypeError):
        resolve_max_concurrent_batches(_NS(max_concurrent_batches=True))
    with pytest.raises(TypeError):
        min_in_flight_output_tokens(_NS(num_sampled_tokens_per_step=1), _NS(num_output_placeholder_steps=False))


@pytest.mark.parametrize("kind", ["recompute", "dyntra_lb", "balance"])
def test_custom_schedulers_respect_async_depth_and_preempt_reset(monkeypatch, kind: str) -> None:
    from vllm import SamplingParams
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler
    from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeSchedulerConfig
    from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

    info_d = dict(architecture="OPTForCausalLM", attn_type="decoder", is_text_generation_model=True, supports_pp=True)
    info = type("_I", (), {**info_d, "supported_video_pruning_methods": (), "__getattr__": lambda *_: None})()
    AS, BS = AsyncScheduler, BalanceScheduler
    bal = AS if issubclass(AS, BS) else type("_AB", (BS, AS), {})
    cls_map = {"recompute": AsyncRecomputeScheduler, "dyntra_lb": AsyncDyntraLBScheduler, "balance": bal}
    asc = _NS(finegrained_tp_config=_NS(oproj_tensor_parallel_size=0))
    monkeypatch.setattr("vllm_ascend.core.recompute_scheduler.get_ascend_config", lambda: asc, raising=False)
    kw = {"async_scheduling_depth": 4} if "async_scheduling_depth" in SchedulerConfig.__dataclass_fields__ else {}
    w = str(Path(__file__).resolve().parents[2] / "_fake_weight")
    with patch.object(ModelRegistry, "inspect_model_cls", return_value=(info, "OPTForCausalLM")):
        sc = SchedulerConfig(max_model_len=128, is_encoder_decoder=False, async_scheduling=True, **kw)
        mc = ModelConfig(model=w, skip_tokenizer_init=True, dtype="float16", seed=42)
        cc, dc = CacheConfig(block_size=16), DeviceConfig("cpu")
        vllm_cfg = VllmConfig(scheduler_config=sc, model_config=mc, cache_config=cc, device_config=dc)
    if kind == "recompute":
        rc = RecomputeSchedulerConfig.initialize_from_config(_NS(scheduler_config=sc, model_config=mc))
        assert rc.scheduler_cls.endswith("AsyncRecomputeScheduler") and (not kw or rc.async_scheduling_depth == 4)
    vllm_cfg.cache_config.num_gpu_blocks = 128
    spec = FullAttentionSpec(16, num_kv_heads=1, head_size=16, dtype=torch.float16)
    kv_cfg = KVCacheConfig(128, [], [KVCacheGroupSpec(["l0"], spec)])
    som = StructuredOutputManager(vllm_cfg)
    sched = cls_map[kind](vllm_config=vllm_cfg, kv_cache_config=kv_cfg, structured_output_manager=som, block_size=16)
    if kind == "balance":
        sched._balance_enabled, sched.balance_queue = True, [torch.tensor([0], dtype=torch.int)]
    init_none_hash(sha256)
    hb = get_request_block_hasher(16, sha256)
    req = Request("req-d4", list(range(8)), SamplingParams(max_tokens=5), None, block_hasher=hb)
    has_native = hasattr(req, "num_output_placeholder_steps")
    sched.add_request(req)
    assert "req-d4" in sched.schedule().num_scheduled_tokens
    req.num_computed_tokens, req.num_output_placeholders, req.num_output_placeholder_steps = 15, 6, 3
    assert "req-d4" not in sched.schedule().num_scheduled_tokens
    sched.running.clear()
    sched._preempt_request(req, 0.0, drop_stale_output=True)
    assert req.num_output_placeholders == 0 and ((not has_native) or req.num_output_placeholder_steps == 0)
    assert "req-d4" in sched.schedule().num_scheduled_tokens


@pytest.mark.parametrize("generic", [False, True])
def test_adaptive_verification_ring_patch_guards_and_row_reuse(monkeypatch, generic: bool) -> None:
    from vllm_ascend.patch.worker.patch_adaptive_verification import _resize_confidence_ring

    if generic:
        monkeypatch.setattr(inspect, "getsource", MagicMock(side_effect=OSError("no source")))
    patch_av, av, mr = _apply_av_stubs(monkeypatch, generic=generic)
    rs = _NS(device=torch.device("cpu"), num_speculative_steps=1, max_num_reqs=2)
    sc, pc = _NS(async_scheduling=True, async_scheduling_depth=4), _NS(pipeline_parallel_size=2)
    vc = _NS(scheduler_config=sc, parallel_config=pc)
    mgr = mr.maybe_create_adaptive_verification_manager(True, [], rs, torch.zeros(3), 1, 16, vllm_config=vc)
    assert (len(mgr._stale_confidences), len(mgr._copy_events), len(mgr._pending_resets)) == (5, 5, 5)
    batch = _NS(num_reqs=2, idx_mapping=torch.arange(2, dtype=torch.int64))
    for step in range(1, 12):
        mgr.record_confidences(torch.full((2, 1), step / 10.0), batch)
        if step == 6:
            assert mgr._stale_confidences[mgr._stale_idx].np[:, 0].tolist() == pytest.approx([0.2, 0.2])
            mgr.add_request(1)
        elif step > 6:
            assert mgr._stale_confidences[mgr._stale_idx].np[1, 0] == pytest.approx(0.7 if step == 11 else 1.0)
    with pytest.raises(TypeError):
        _resize_confidence_ring(_NS(), True)
    wrapped_init = av.AdaptiveVerificationManager.__init__
    patch_av._apply_confidence_ring_patch()
    assert av.AdaptiveVerificationManager.__init__ is wrapped_init
    monkeypatch.setattr(inspect, "signature", lambda _t: (_ for _ in ()).throw(ValueError("bad")))
    _, bav, _ = _apply_av_stubs(monkeypatch)
    assert not hasattr(bav.AdaptiveVerificationManager, "_vllm_ascend_confidence_ring_patched")


def test_kv_trace_depth_pool_out_of_order_and_shrink(monkeypatch) -> None:
    import vllm_ascend.kv_trace.kv_trace_plan as kv_plan_mod

    orig_empty = torch.empty
    monkeypatch.setattr(torch, "npu", _NS(Event=MagicMock, current_stream=lambda: "s0"), raising=False)
    monkeypatch.setattr(kv_plan_mod.torch, "empty", lambda *a, **kw: orig_empty(*a, **{**kw, "pin_memory": False}))
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

    class _NpuTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, data: torch.Tensor) -> _NpuTensor:
            return torch.Tensor._make_subclass(cls, data)

        device = property(lambda _self: torch.device("meta"))  # type: ignore[assignment]

    with pytest.raises(TypeError):
        PlanSnapshotter(max_in_flight=True)  # type: ignore[arg-type]
    sc, pc = _NS(async_scheduling=True, async_scheduling_depth=4), _NS(pipeline_parallel_size=1, tensor_parallel_size=1)
    ac = {"kv_trace": {"enabled": True, "action": "raise"}}
    ns = dict(kv_transfer_config=None, model_config=_NS(), speculative_config=None)
    cfg = _NS(use_v2_model_runner=True, additional_config=ac, scheduler_config=sc, parallel_config=pc, **ns)
    rt = KVTraceRuntime(cfg)
    kc, bt = _NS(num_blocks=16, kv_cache_groups=[]), _NS(redirect_writes_to_null_block=False)
    runner = _NS(vllm_config=cfg, kv_cache_config=kc, is_pooling_model=False, sample_tokens=int, block_tables=bt)
    rt.configure(runner)
    m0 = {0: _NS(slot_mapping=_NpuTensor(torch.tensor([9])), num_actual_tokens=1)}
    [rt._snapshotter.freeze(AccessPlan("forward", ["r"], {"r": (None, 1)}, m0, False, {})) for _ in range(5)]
    assert list(rt._snapshotter._free_slots) == [0, 1, 2, 3]

    def _enq(step: int, out: Any) -> Any:
        rt._phase, rt._scheduled = "forward", {f"r-{step}": (None, 1)}
        rt._metadata = {0: _NS(slot_mapping=_NpuTensor(torch.tensor([step])), num_actual_tokens=1)}
        rt._snapshot_audit(_NS(request_ids=[f"r-{step}"], runner=runner))
        return rt._guard_output(out)

    out0, _ = _enq(0, _NS(get_output=lambda: "out-0")), _enq(1, _NS(raw="out-1"))
    p0 = rt._pending_audits[0]
    assert _enq(2, _NS(get_output=lambda: "out-2")).get_output() == "out-2"
    out3, out4 = _enq(3, _NS(get_output=lambda: "out-3")), _enq(4, _NS(get_output=lambda: "out-4"))
    assert not p0.is_done and p0.slot_id == 0 and rt._pending_audits[-1].slot_id in {1, 2, 3}
    assert (out0.get_output(), out3.get_output(), out4.get_output()) == ("out-0", "out-3", "out-4")
    sc.async_scheduling = False
    rt.configure(runner)
    assert (rt._max_pending_audits, len(rt._snapshotter._slots), len(rt._pending_audits)) == (1, 1, 0)


def test_runtime_guard_wave_bounded_gates_and_reap_window(monkeypatch, request) -> None:
    inner = type("_Inner", (AsyncOutput,), {"__init__": lambda s, o: setattr(s, "_o", o), "get_output": lambda s: s._o})
    monkeypatch.setattr("vllm_ascend.observability.runtime_guard.hooks.is_async_output_rank", lambda: True)
    qsl = "vllm_ascend.observability.runtime_guard.detector.logits_finite.query_start_loc_np"
    monkeypatch.setattr(qsl, lambda *_a, **_kw: np.array([0, 1], dtype=np.int64))
    det = LogitsFiniteDetector(runtime_config=None, runner=None)
    det._enabled, det._item_sync = True, False
    gates = [_NS(item=lambda: True), _NS(item=lambda: False), RuntimeError("d2h failed")]
    with patch.object(torch.Tensor, "to", side_effect=gates):
        for w in (1, 2, 3):
            lg = torch.tensor([[1.0 if w == 1 else float("nan"), 1.0]])
            det.check_all(logits=lg, logits_indices=None, input_batch=_NS(req_ids=[f"r{w}"]), wave=w)
    assert det.check_deferred(up_to_wave=1) == []
    assert [(a.req_id, a.wave) for a in det.check_deferred(up_to_wave=2)] == [("r2", 2)]
    assert [(a.req_id, a.wave) for a in det.check_deferred()] == [("r3", 3)]
    seen: list[dict[str, Any]] = []
    guard = _NS(wave_tracker=_NS(current_wave=lambda: 3), check_after_sample=lambda **kw: seen.append(kw))
    wrapped = maybe_wrap_v2_async_output(inner(_NS(sampled_token_ids=[[7]], req_ids=["r"])), _NS(runtime_guard=guard))
    assert isinstance(wrapped, AscendAsyncOutput) and wrapped.get_output() is not None and seen[-1]["up_to_wave"] == 3
    RequestGuardStore.reset_for_tests()
    request.addfinalizer(RequestGuardStore.reset_for_tests)
    proc = object.__new__(RuntimeGuardProcessor)
    for d, pp, exp in [(2, 1, MAX_DEFERRED_REAP_WAVES), (8, 2, 11)]:
        sc, pc = _NS(async_scheduling=True, async_scheduling_depth=d), _NS(pipeline_parallel_size=pp)
        proc._configure_store_deferred_reap_waves(_NS(vllm_config=_NS(scheduler_config=sc, parallel_config=pc)))
        assert RequestGuardStore.get().max_deferred_waves == exp
