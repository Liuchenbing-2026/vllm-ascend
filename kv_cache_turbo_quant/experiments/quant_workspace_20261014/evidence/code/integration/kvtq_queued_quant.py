"""Queue-aware submission of the original, unchanged Ascend C quantizer."""

import torch
import torch_npu  # noqa: F401

NATIVE_EXTENSION = "/ws/quant_ws_20261014/torch-extensions/tq_quant_ws/tq_quant_ws.so"


class QueuedQuantizer:
    def __init__(self):
        if not hasattr(torch.ops.turboquant, "kv_cache_turbo_quant_workspace"):
            torch.ops.load_library(NATIVE_EXTENSION)

    def __call__(self, vectors, rotation, qjl, bits):
        return torch.ops.turboquant.kv_cache_turbo_quant_workspace(vectors, rotation, qjl, bits)
