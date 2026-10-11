"""Fixed original read, wrapped for the skill verifier's strict integer check.

The original output is viewed as int16, with no value conversion. All tensor
inputs are integers so the verifier requires exact equality of every bit.
"""
import math

import torch

from kvtq_store_reference import CENTROIDS, _dequant_dense


class Model(torch.nn.Module):
    def forward(self, cache_u8, block_table, cumulative, centroid_bits,
                total_tokens, max_seq_len, seq_lens_list):
        return _dequant_dense(None, cache_u8.view(torch.bfloat16),
                              block_table, seq_lens_list).view(torch.int16)


def get_init_inputs():
    return []


def get_input_groups():
    # Diverse tails, ragged batches, reordered/reused physical pages, head
    # counts and non-contiguous table views. Every packed byte code occurs.
    cases = [([1], 1, 128, False), ([15, 16, 17, 127, 128, 129], 2, 128, False),
             ([0, 33, 0, 257], 3, 128, True), ([20]*16, 2, 128, False),
             ([2000 + i % 3 for i in range(60)], 2, 128, False),
             ([0, 0], 2, 128, False), ([31, 257, 511], 4, 64, True)]
    generator = torch.Generator().manual_seed(20261011)
    inputs = []
    for lengths, heads, block_size, strided in cases:
        max_len = max(lengths)
        blocks = max(1, math.ceil(max_len / block_size))
        pages = max(2, blocks*len(lengths) + 3)
        cache = torch.randint(0, 256, (pages, block_size, heads, 66),
                              dtype=torch.uint8, generator=generator)
        codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
        cache.reshape(-1, 66)[:4, :64] = codes.view(4, 64)
        norm = (torch.rand((pages, block_size, heads), generator=generator)*32).to(torch.bfloat16)
        norm.reshape(-1)[:4] = torch.tensor([0.0, 1.0, 2.0, 32.0], dtype=torch.bfloat16)
        cache[..., 64:] = norm.unsqueeze(-1).view(torch.uint8)
        table = torch.randint(0, pages, (len(lengths), blocks), generator=generator, dtype=torch.int32)
        if strided:
            wide = torch.empty((len(lengths), blocks*2), dtype=torch.int32)
            wide[:, ::2] = table
            table = wide.to('npu')[:, ::2]
        else:
            table = table.to('npu')
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        cent = torch.tensor(CENTROIDS[4], dtype=torch.bfloat16).view(torch.int16)
        inputs.append([cache.to('npu'), table,
                       torch.tensor(cumulative, dtype=torch.int32, device='npu'),
                       cent.to('npu'), sum(lengths), max_len, lengths])
    return inputs
