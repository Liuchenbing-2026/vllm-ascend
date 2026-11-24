# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests for HiSparse cross-TP shared indexer offloading."""

from __future__ import annotations

import contextlib
import dataclasses
import mmap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.config as ocfg_mod
import vllm.v1.kv_cache_interface as kv_cache_interface
import vllm.v1.kv_offload.config as oc
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_offload.base import CanonicalKVCaches

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu as npu_mod
import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector as oc_mod
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec

ALIGN = npu_mod.NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT


def _make_indexer(
    cache_sparse_li_c8: bool = False,
    dcp_sharded: bool = True,
    max_tp_shards: int | None = 1,
) -> AscendSFAIndexerCacheSpec:
    if cache_sparse_li_c8:
        return AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
            dcp_sharded=dcp_sharded,
            max_tp_shards=max_tp_shards,
        )
    return AscendSFAIndexerCacheSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        dcp_sharded=dcp_sharded,
        max_tp_shards=max_tp_shards,
    )


def _make_group(
    specs: dict[str, Any],
    *,
    style: str,
    host_resident: bool = False,
    role: Any = kv_cache_interface.KVCacheGroupRole.DEFAULT,
) -> Any:
    if style == "legacy":
        spec = (
            next(iter(specs.values()))
            if len(specs) == 1
            else kv_cache_interface.UniformTypeKVCacheSpecs.from_specs(specs)
        )
        return kv_cache_interface.KVCacheGroupSpec(list(specs), spec, host_resident=host_resident, role=role)
    first = next(iter(specs.values()))
    group_cls = getattr(kv_cache_interface, "KVCacheGroup", None)
    if style == "kv_cache_group" and group_cls is not None:
        return group_cls(
            kv_cache_specs=specs,
            manager_config=None,
            block_size=first.block_size,
            dcp_sharded=getattr(first, "dcp_sharded", False),
            host_resident=host_resident,
            role=role,
        )
    return SimpleNamespace(
        layer_names=list(specs),
        kv_cache_specs=specs,
        block_size=first.block_size,
        host_resident=host_resident,
        role=role,
        page_size_bytes=sum(spec.page_size_bytes for spec in specs.values()),
        prefix_cacheable=not host_resident,
        enable_kv_transfer=True,
        is_eagle_group=False,
    )


def _make_hisparse_config(
    *,
    style: str = "native",
    cache_sparse_li_c8: bool = False,
    dcp_sharded: bool = True,
    max_tp_shards: int | None = 1,
    num_blocks: int = 8,
    host_num_blocks: int | None = 8,
    layers: tuple[str, ...] = ("l0.idx",),
) -> Any:
    native_style = "kv_cache_group" if hasattr(kv_cache_interface, "KVCacheGroup") else "legacy"
    effective_style = native_style if style == "native" else style
    mla = AscendMLAAttentionSpec(
        block_size=128 if cache_sparse_li_c8 else 16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    res_cls = getattr(kv_cache_interface, "HiSparsePageSpec", getattr(kv_cache_interface, "HiSparseResidentSpec", None))
    resident_spec = (
        res_cls(block_size=mla.block_size, page_size=mla.page_size_bytes)
        if res_cls is not None
        else SimpleNamespace(
            block_size=mla.block_size,
            page_size_bytes=mla.page_size_bytes,
            prefix_cacheable=False,
            dcp_sharded=False,
        )
    )
    groups = [
        _make_group({"l0.attn": mla}, style=effective_style, host_resident=True),
        _make_group(
            dict.fromkeys(layers, _make_indexer(cache_sparse_li_c8, dcp_sharded, max_tp_shards)),
            style=effective_style,
            role=kv_cache_interface.KVCacheGroupRole.HISPARSE_INDEXER,
        ),
        _make_group({"hs.res": resident_spec}, style=effective_style),
    ]
    if effective_style == native_style:
        return kv_cache_interface.KVCacheConfig(
            num_blocks,
            [],
            groups,
            hisparse_host_num_blocks=host_num_blocks,
            kv_tp_replicas=1,
        )
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        hisparse_host_num_blocks=host_num_blocks,
        kv_tp_replicas=1,
        prefix_cacheable_group_ids=(),
    )


def _make_offload_config(
    extra: dict[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
    tp_size: int = 1,
    replicated: bool = False,
) -> oc.OffloadingConfig:
    parallel_kwargs: dict[str, Any] = dict(
        rank=rank,
        world_size=world_size,
        tp_size=tp_size,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        data_parallel_index=0,
        is_parallelism_agnostic=True,
        data_parallel_size=1,
        data_parallel_rank_local=None,
    )
    if any(field.name == "local_world_size" for field in dataclasses.fields(oc.OffloadingParallelConfig)):
        parallel_kwargs["local_world_size"] = world_size
    return oc.OffloadingConfig(
        groups=(oc.OffloadingGroupConfig(16, ("l0.idx",), 1),),
        worker_kv_bytes_per_block=ALIGN,
        enable_kv_cache_events=False,
        extra_config=extra,
        engine_id="e-t91",
        model=oc.OffloadingModelConfig("ds", "bfloat16"),
        cache=oc.OffloadingCacheConfig(16, 1),
        parallel=oc.OffloadingParallelConfig(**parallel_kwargs),
        replicated_layout=replicated,
    )


def _make_vllm_config(*, tp: int = 4, **overrides: Any) -> SimpleNamespace:
    use_mla = overrides.pop("use_mla", True)
    world_size = overrides.pop("ws", tp)
    nnodes = overrides.pop("nnodes", 1)
    parallel = dict(
        rank=0,
        world_size=world_size,
        local_world_size=world_size // max(1, nnodes),
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_index=0,
        data_parallel_size=1,
        data_parallel_rank_local=0,
        distributed_executor_backend="mp",
        nnodes_within_dp=nnodes,
    )
    parallel.update(overrides)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="e",
            kv_connector="OffloadingConnector",
            kv_connector_extra_config={"cpu_bytes_to_use": 32 * ALIGN},
        ),
        parallel_config=SimpleNamespace(**parallel),
        model_config=SimpleNamespace(
            use_mla=use_mla,
            model="ds",
            dtype=torch.bfloat16,
            max_model_len=4096,
            get_total_num_kv_heads=lambda: 1,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="auto",
            kv_cache_layout="LBHNC",
            enable_prefix_caching=True,
            prefix_cache_retention_interval=0,
            prefix_match_unit=None,
        ),
        kv_events_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        use_v2_model_runner=False,
        num_hash_lookahead_tokens=0,
    )


@pytest.mark.parametrize(
    ("tp", "style", "c8"),
    [(1, "legacy", False), (4, "legacy", True), (4, "kv_cache_group", False)],
)
def test_prepare_hisparse_config_and_parallel_boundaries(tp: int, style: str, c8: bool) -> None:
    cfg = _make_hisparse_config(style=style, cache_sparse_li_c8=c8, layers=("l0.idx", "l1.idx"))
    prepared = oc_mod.prepare_hisparse_offloading_kv_cache_config(_make_vllm_config(tp=tp), cfg)
    assert prepared.kv_tp_replicas == tp and cfg.kv_tp_replicas == 1
    assert (prepared is not cfg) if tp > 1 else (prepared is cfg)

    dcp_cfg = _make_hisparse_config(style="kv_cache_group", dcp_sharded=True)
    assert (
        oc_mod.prepare_hisparse_offloading_kv_cache_config(
            _make_vllm_config(tp=4, decode_context_parallel_size=2), dcp_cfg
        ).kv_tp_replicas
        == 1
    )
    assert (
        oc_mod.prepare_hisparse_offloading_kv_cache_config(
            _make_vllm_config(tp=8), _make_hisparse_config(style="kv_cache_group", max_tp_shards=2)
        ).kv_tp_replicas
        == 4
    )
    assert (
        oc_mod.prepare_hisparse_offloading_kv_cache_config(
            _make_vllm_config(tp=8), _make_hisparse_config(style="kv_cache_group", max_tp_shards=None)
        ).kv_tp_replicas
        == 1
    )

    non_hs = _make_hisparse_config(style="kv_cache_group", host_num_blocks=None)
    zero_blocks = _make_hisparse_config(style="kv_cache_group", num_blocks=0)
    assert oc_mod.prepare_hisparse_offloading_kv_cache_config(_make_vllm_config(tp=8), non_hs) is non_hs
    assert oc_mod.prepare_hisparse_offloading_kv_cache_config(_make_vllm_config(tp=8), zero_blocks) is zero_blocks

    cases: tuple[tuple[dict[str, Any], bool], ...] = (
        ({}, True),
        ({"tp": 1, "ws": 1}, False),
        ({"decode_context_parallel_size": 2}, False),
        ({"pipeline_parallel_size": 2}, False),
        ({"prefill_context_parallel_size": 2}, False),
        ({"nnodes": 2}, False),
        ({"distributed_executor_backend": "ray"}, False),
        ({"ws": 8}, False),
        ({"use_mla": False}, False),
        ({"max_tp": 2}, False),
    )
    for overrides, expected_replicated in cases:
        kw = dict(overrides)
        max_tp = kw.pop("max_tp", 1)
        vc = _make_vllm_config(**kw)
        prepared_cfg = oc_mod.prepare_hisparse_offloading_kv_cache_config(
            vc,
            _make_hisparse_config(max_tp_shards=max_tp),
        )
        ocfg = ocfg_mod.build_offloading_config(vc, prepared_cfg)
        assert ocfg.replicated_layout is expected_replicated
        assert npu_mod.NPUOffloadingSpec(ocfg)._uses_shared_region() is expected_replicated


@pytest.mark.parametrize("cls", [npu_mod.NPUOffloadingSpec, npu_mod.NPUTieringOffloadingSpec])
def test_single_copy_dedup_shared_region_and_fallback(monkeypatch: pytest.MonkeyPatch, cls: type[Any]) -> None:
    rep = cls(_make_offload_config({"cpu_bytes_to_use": 32 * ALIGN}, world_size=4, tp_size=4, replicated=True))
    per = cls(_make_offload_config({"cpu_bytes_to_use": 32 * ALIGN}, world_size=4, tp_size=4, replicated=False))
    assert rep.num_chunks == 32 and per.num_chunks == 8
    assert rep._uses_shared_region() and rep.replicated_layout and not per.replicated_layout
    if cls is not npu_mod.NPUOffloadingSpec:
        return

    reg_kwargs: dict[str, Any] = {}
    worker_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(
        npu_mod,
        "SharedOffloadRegion",
        lambda **kw: reg_kwargs.update(kw) or SimpleNamespace(**kw),
    )
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", lambda **kw: worker_kwargs.update(kw) or "worker")
    assert npu_mod._parse_pin_shared_mmap({"pin_shared_mmap": "true"})
    assert not npu_mod._parse_pin_shared_mmap({"pin_shared_mmap": "no"})

    spec = npu_mod.NPUOffloadingSpec(
        _make_offload_config(
            {"cpu_bytes_to_use": 4 * ALIGN, "pin_shared_mmap": "true"},
            rank=3,
            world_size=4,
            tp_size=4,
            replicated=True,
        )
    )
    assert spec.create_worker(object()) == "worker"  # type: ignore[arg-type]
    assert reg_kwargs["rank"] == 0 and worker_kwargs["pin_shared_mmap"] is True
    assert worker_kwargs["worker_slot_only"] is False

    reg_kwargs.clear()
    worker_kwargs.clear()
    zero_spec = npu_mod.NPUOffloadingSpec(
        _make_offload_config({"cpu_bytes_to_use": 128}, rank=1, world_size=4, tp_size=4, replicated=True)
    )
    zero_spec.create_worker(object())  # type: ignore[arg-type]
    assert not reg_kwargs and "mmap_region" not in worker_kwargs


def test_cross_tp_single_writer_and_shared_mmap_read_consistency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.kv_offload.factory.OffloadingSpecFactory.create_spec",
        staticmethod(npu_mod.NPUOffloadingSpec),
    )

    def fake_init(
        self: Any,
        vc: Any,
        role: KVConnectorRole,
        kc: kv_cache_interface.KVCacheConfig,
    ) -> None:
        self._vllm_config, self._kv_cache_config, self._role = vc, kc, role
        spec = npu_mod.NPUOffloadingSpec(ocfg_mod.build_offloading_config(vc, kc))
        self.connector_scheduler = SimpleNamespace(spec=spec) if role == KVConnectorRole.SCHEDULER else None
        self.connector_worker = (
            oc_mod.AscendOffloadingConnectorWorker(spec, vc, kc) if role == KVConnectorRole.WORKER else None
        )

    monkeypatch.setattr(OffloadingConnector, "__init__", fake_init)
    kc = _make_hisparse_config()
    sched = oc_mod.AscendOffloadingConnector(_make_vllm_config(rank=0, tp=4), KVConnectorRole.SCHEDULER, kc)
    workers = [
        oc_mod.AscendOffloadingConnector(_make_vllm_config(rank=r, tp=4), KVConnectorRole.WORKER, kc).connector_worker
        for r in range(4)
    ]
    assert [w._is_store_writer for w in workers] == [True, False, False, False]  # type: ignore[union-attr]

    chunk_bytes = sched.connector_scheduler.spec.kv_bytes_per_chunk  # type: ignore[union-attr]
    path = tmp_path / "shared.mmap"
    path.write_bytes(b"\x00" * (chunk_bytes * 2))
    views: list[torch.Tensor] = []
    with contextlib.ExitStack() as stack:
        for _ in range(4):
            file_obj = stack.enter_context(open(path, "r+b"))
            mm = stack.enter_context(
                mmap.mmap(
                    file_obj.fileno(),
                    chunk_bytes * 2,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            )
            views.append(torch.frombuffer(memoryview(mm), dtype=torch.int8).view(2, chunk_bytes))
        payload = torch.arange(chunk_bytes, dtype=torch.int64).to(torch.int8)
        views[0][1].copy_(payload)
        assert all(torch.equal(views[r][1], payload) for r in range(1, 4))
        views.clear()


@pytest.mark.parametrize(
    ("style", "c8"),
    [("legacy", False), ("legacy", True), ("kv_cache_group", False), ("kv_cache_group", True)],
)
def test_ascend_worker_registers_hisparse_indexer_caches(
    monkeypatch: pytest.MonkeyPatch,
    style: str,
    c8: bool,
) -> None:
    layer, stride = "l0.idx", (1 if c8 else 4)
    kc = _make_hisparse_config(style=style, cache_sparse_li_c8=c8, num_blocks=3, layers=(layer,))
    idx = _make_indexer(cache_sparse_li_c8=c8)
    worker = oc_mod.AscendOffloadingConnectorWorker.__new__(oc_mod.AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kc
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={layer: SimpleNamespace(hisparse_block_stride=stride)}
        )
    )
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append  # type: ignore[method-assign]
    monkeypatch.setattr(oc_mod, "derive_canonical_mappings", lambda *a: {})
    if c8:
        key_cache = torch.empty((3, 128, 1, 128), dtype=torch.int8)
        scale_cache = torch.empty((3, 128, 1, 1), dtype=torch.float16)
        worker.register_kv_caches({layer: (key_cache, scale_cache)})
        assert [t.tensor.data_ptr() for t in captured[0].tensors] == [key_cache.data_ptr(), scale_cache.data_ptr()]
    else:
        worker.register_kv_caches({layer: (torch.empty((3 * stride, 16, 1, 128), dtype=torch.bfloat16),)})
        assert [t.tensor.shape for t in captured[0].tensors] == [(3, idx.page_size_bytes)]
    assert sum(ref.page_size_bytes for ref in captured[0].group_data_refs[0]) == idx.unpadded_page_size_bytes
