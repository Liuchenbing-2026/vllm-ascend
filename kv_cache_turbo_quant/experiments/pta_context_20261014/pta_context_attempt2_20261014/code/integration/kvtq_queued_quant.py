"""Queue-aware submission of the original, unchanged Ascend C quantizer."""

import torch
import torch_npu  # noqa: F401

NATIVE_EXTENSION = "/ws/pta_context_attempt2_20261014/torch-extensions/tq_pta_quant/tq_pta_quant.so"


class QueuedQuantizer:
    def __init__(self):
        if not hasattr(torch.ops.turboquant, "kv_cache_turbo_quant_pta"):
            torch.ops.load_library(NATIVE_EXTENSION)

    def __call__(self, vectors, rotation, qjl, bits):
        return torch.ops.turboquant.kv_cache_turbo_quant_pta(vectors, rotation, qjl, bits)
