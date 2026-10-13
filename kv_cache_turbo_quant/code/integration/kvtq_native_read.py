"""Native reader for long histories; preserves the original byte/LUT semantics."""

import torch
import torch_npu  # noqa: F401

NATIVE_EXTENSION = (
    "/ws/queued_read_20261014/torch-extensions/tq_queued_read/tq_queued_read.so"
)


class NativePagedReader:
    def __init__(self):
        if not hasattr(torch.ops.turboquant, "tq_paged_read_queued"):
            torch.ops.load_library(NATIVE_EXTENSION)

    def __call__(self, cache, table, lengths, lookup, metadata=None, segment="decode"):
        total = sum(lengths)
        if total == 0:
            return torch.empty(
                (0, cache.shape[2], 128), dtype=torch.bfloat16, device=cache.device
            )
        if (
            cache.shape[-1] != 66
            or cache.dtype != torch.uint8
            or not cache.is_contiguous()
        ):
            raise ValueError(
                "native reader requires the original contiguous 66-byte store4 cache"
            )
        fingerprint = (str(cache.device), cache.shape[1], tuple(lengths))
        plans = (
            getattr(metadata, "_kvtq_native_jobs", None)
            if metadata is not None
            else None
        )
        if plans is not None and segment in plans and plans[segment][0] == fingerprint:
            jobs = plans[segment][1]
        else:
            descriptors = []
            begin = 0
            for sequence, length in enumerate(lengths):
                for position in range(0, length, cache.shape[1]):
                    descriptors.append(
                        [
                            sequence,
                            position // cache.shape[1],
                            begin + position,
                            min(cache.shape[1], length - position),
                        ]
                    )
                begin += length
            jobs = torch.tensor(descriptors, dtype=torch.int32, device=cache.device)
            if metadata is not None:
                if plans is None:
                    plans = {}
                    metadata._kvtq_native_jobs = plans
                plans[segment] = (fingerprint, jobs)
        # Physical pages are deliberately read from the current table on every
        # call. A reused metadata object with changed page IDs cannot use stale KV.
        return torch.ops.turboquant.tq_paged_read_queued(
            cache, table.contiguous(), jobs, lookup, total
        )
