# SPDX-License-Identifier: Apache-2.0
"""CPU byte oracle for scheduler-to-worker retained KV address mapping.

The real connector computes request metadata, block buckets and READ ranges.
Only the transport is replaced: it copies between bounded CPU byte tensors.
This does not validate NPU DMA, model arithmetic or peer crash recovery.
"""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.bidirectional import BidirectionalKVConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import MooncakeKVConnectorStats


class BoundedCPURead:
    def __init__(self, local_regions, remote_regions):
        self.local_regions = local_regions
        self.remote_regions = remote_regions
        self.total_bytes = 0

    @staticmethod
    def region(regions, address, length):
        for base, data in regions:
            offset = address - base
            if offset >= 0 and offset + length <= data.numel():
                return data[offset : offset + length]
        raise AssertionError("READ range exceeds its registered CPU storage")

    def batch_transfer_sync_read(self, session, local_addresses, remote_addresses, lengths):
        assert session == "127.0.0.1:9000"
        assert len(local_addresses) == len(remote_addresses) == len(lengths)
        for local, remote, length in zip(local_addresses, remote_addresses, lengths):
            destination = self.region(self.local_regions, local, length)
            source = self.region(self.remote_regions, remote, length)
            destination.copy_(source)
            self.total_bytes += length
        return 0


def make_scheduler(block_size, spec):
    scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    scheduler.bidirectional = BidirectionalKVConfig(True, 480, 1)
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)
    scheduler.block_size = block_size
    scheduler._reqs_need_recv = {}
    scheduler._reqs_recv_info = {}
    scheduler._reqs_in_batch = set()
    scheduler._recving_thread = SimpleNamespace(add_request=lambda *args: None)
    scheduler.group_unique_specs = [[spec]]
    return scheduler


def make_storage(layer_count, physical_blocks, heads, block_size, head_dim, interleaved):
    # Both layouts have full, contiguous per-page data; interleaved K/V views
    # have a page stride twice their valid byte length, matching the NPU case.
    shapes = (
        (physical_blocks, 2, heads, block_size, head_dim)
        if interleaved
        else (2, physical_blocks, heads, block_size, head_dim)
    )
    storage = []
    pairs = []
    for layer in range(layer_count):
        backing = torch.arange(math.prod(shapes)).remainder(251).reshape(shapes).to(torch.bfloat16) + layer
        storage.append(backing)
        pairs.append((backing[:, 0], backing[:, 1]) if interleaved else (backing[0], backing[1]))
    return storage, pairs


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize(
    "block_size,prompt_tokens,retained_tokens,local_hits",
    [(16, 81, 48, 0), (16, 81, 48, 16), (128, 2335, 2176, 0), (128, 2335, 2176, 1024)],
)
def test_retained_scheduler_metadata_to_read_bytes(block_size, prompt_tokens, retained_tokens, local_hits, interleaved):
    physical_blocks, heads, head_dim, layer_count = 32, 2, 8, 2
    source_storage, source = make_storage(layer_count, physical_blocks, heads, block_size, head_dim, interleaved)
    local_storage, local = make_storage(layer_count, physical_blocks, heads, block_size, head_dim, interleaved)
    for backing in local_storage:
        backing.fill_(-1)
    expected_storage = [backing.clone() for backing in local_storage]
    expected = [
        (backing[:, 0], backing[:, 1]) if interleaved else (backing[0], backing[1]) for backing in expected_storage
    ]
    source_blocks = list(range(physical_blocks - 1, 0, -1))[: retained_tokens // block_size]
    local_blocks = [(index * 7 + 3) % physical_blocks for index in range(math.ceil(prompt_tokens / block_size))]
    hit_blocks = local_hits // block_size
    for layer in range(layer_count):
        for kind in range(2):
            for logical, remote_block in enumerate(source_blocks):
                expected[layer][kind][local_blocks[logical]].copy_(source[layer][kind][remote_block])
                if logical < hit_blocks:
                    local[layer][kind][local_blocks[logical]].copy_(source[layer][kind][remote_block])

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=head_dim, head_size_v=head_dim, dtype=torch.bfloat16
    )
    scheduler = make_scheduler(block_size, spec)
    request = SimpleNamespace(
        request_id="next-turn",
        num_prompt_tokens=prompt_tokens,
        prompt_token_ids=list(range(prompt_tokens)),
        kv_transfer_params={
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_retained": True,
            "remote_num_tokens": retained_tokens,
            "remote_num_prompt_tokens": retained_tokens,
            "remote_block_ids": (source_blocks,),
            "remote_engine_id": "decoder",
            "remote_request_id": "previous-turn",
            "remote_host": "127.0.0.1",
            "remote_port": 6001,
        },
    )
    matched, asynchronous = scheduler.get_num_new_matched_tokens(request, local_hits)
    assert (matched, asynchronous) == (retained_tokens - local_hits, True)
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: (local_blocks[hit_blocks:],),
        get_block_ids=lambda: (local_blocks,),
    )
    scheduler.update_state_after_alloc(request, blocks, matched)
    metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert metadata.requests[request.request_id].num_external_tokens == matched

    # Deliberately reverse remote layer registration. Layer pairs, rather than
    # registration order or coincident physical block IDs, must select data.
    order = list(reversed(range(layer_count)))
    names = [f"layer.{layer}" for layer in range(layer_count)]
    block_len = heads * block_size * head_dim * 2
    local_bases = [[pair[0].data_ptr(), pair[1].data_ptr()] for pair in local]
    remote_bases = [[source[layer][0].data_ptr(), source[layer][1].data_ptr()] for layer in order]
    strides = [[pair[0].stride(0) * 2, pair[1].stride(0) * 2] for pair in local]
    remote_strides = [[source[layer][0].stride(0) * 2, source[layer][1].stride(0) * 2] for layer in order]
    remote = MooncakePPTransferMetadata(
        block_size=block_size,
        num_blocks=physical_blocks,
        layer_names=[names[layer] for layer in order],
        layer_block_sizes=[block_size] * layer_count,
        group_indices=[0] * layer_count,
        block_strides=remote_strides,
        block_lens=[[block_len, block_len] for _ in names],
        block_shapes=[[(heads, block_size, head_dim)] * 2 for _ in names],
        block_size_scales=[[1, 1] for _ in names],
        metadata_by_pcp_rank={
            0: MooncakePCPTransferMetadata(
                metadata_by_tp_rank={
                    0: MooncakeTPTransferMetadata(9000, list(range(layer_count)), remote_bases, "127.0.0.1", 6000)
                }
            )
        },
    )
    worker = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    worker.tp_rank, worker.tp_size, worker.pcp_size, worker.dcp_rank, worker.dcp_size = 0, 1, 1, 0, 1
    worker.layer_names = names
    worker.group_indices = [0] * layer_count
    worker.spec_indices = [0] * layer_count
    worker.kv_cache_specs = [spec]
    worker.layer_block_sizes = [block_size] * layer_count
    worker.block_size_scales = [[1, 1] for _ in names]
    worker.kv_caches_base_addr = local_bases
    worker.block_strides = strides
    worker.block_lens = [[block_len, block_len] for _ in names]
    worker.block_shapes = [[(heads, block_size, head_dim)] * 2 for _ in names]
    worker.local_metadata = SimpleNamespace(local_ip="127.0.0.1")
    worker.xfer_stats = MooncakeKVConnectorStats()
    worker.xfer_stats_lock = threading.Lock()
    worker.engine = BoundedCPURead(
        [(backing.data_ptr(), backing.view(torch.uint8).reshape(-1)) for backing in local_storage],
        [(backing.data_ptr(), backing.view(torch.uint8).reshape(-1)) for backing in source_storage],
    )
    layer_pairs = [(layer, order.index(layer)) for layer in range(layer_count)]
    buckets, _ = worker._build_transfer_block_buckets(
        remote, layer_pairs, {pair: [[0]] for pair in layer_pairs}, 1, 1, metadata.requests, {}
    )
    for (pcp_rank, tp_rank), entries in buckets.items():
        worker._execute_tp_transfer_bucket(0, pcp_rank, tp_rank, 1, 1, 1, remote, entries)

    assert worker.engine.total_bytes == (retained_tokens - local_hits) * heads * head_dim * 2 * 2 * layer_count
    # Whole backing tensors also cover untouched cached pages, the new suffix,
    # unrelated physical pages, and sibling K/V storage between page strides.
    for actual, independent in zip(local_storage, expected_storage):
        assert torch.equal(actual.view(torch.uint8), independent.view(torch.uint8))
