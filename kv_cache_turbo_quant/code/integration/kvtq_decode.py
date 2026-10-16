"""Keep one Python dispatch for the unchanged rotation/attention/output sequence."""
import torch


class NativeDecode:
    def __init__(self, extension_path):
        if not hasattr(torch.ops.turboquant, "tq_decode"):
            torch.ops.load_library(extension_path)
        self._operator = torch.ops.turboquant.tq_decode

    def __call__(self, query, key, value, rotation, rotation_t, q_lengths,
                 kv_lengths, heads, kv_heads, block_size, scale, output):
        self._operator(query, key, value, rotation, rotation_t, q_lengths,
                       kv_lengths, heads, kv_heads, block_size, scale, output)
