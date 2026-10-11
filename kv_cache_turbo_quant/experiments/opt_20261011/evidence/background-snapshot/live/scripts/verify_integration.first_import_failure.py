#!/usr/bin/env python3
"""Compare the unchanged store path with the candidate through real NPU APIs."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch_npu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(name, expected, actual, records):
    torch.npu.synchronize()
    assert expected.shape == actual.shape, name
    mismatch = (expected.view(torch.int16) != actual.view(torch.int16)).sum().item()
    records.append({'name': name, 'shape': list(actual.shape), 'mismatched_bf16_bits': mismatch})
    assert mismatch == 0, records[-1]
    print(json.dumps(records[-1]), flush=True)


def main():
    root = Path('/ws/opt_20261011')
    sys.path.insert(0, str(root / 'candidate/integration'))
    original = load('original_store', '/root/kvtq_integration/kvtq_store.py')
    candidate = load('candidate_store', root / 'candidate/integration/kvtq_store.py')
    bits = int(os.environ.get('VLLM_ASCEND_KVTQ_BITS', '4'))
    records = []
    if bits != 4:
        # These formats must continue using the original staged implementation.
        width = {2: 32, 3: 48}[bits] + 2
        cache = torch.randint(0, 256, (4, 128, 2, width), dtype=torch.uint8)
        norms = torch.rand(4, 128, 2, 1).to(torch.bfloat16)
        cache[..., -2:] = norms.view(torch.uint8)
        cache = cache.to('npu').view(torch.bfloat16)
        table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device='npu')
        equal(f'unchanged_{bits}bit_fallback', original._dequant_dense(None, cache, table, [129, 31]),
              candidate._dequant_dense(None, cache, table, [129, 31]), records)
    else:
        sys.path.insert(0, str(root / 'verify'))
        from paged_read_torch import get_input_groups
        groups = get_input_groups()
        for number, (cache, table, cumulative, centroid, total, maximum, lengths) in enumerate(groups, 1):
            metadata = SimpleNamespace()
            equal(f'full_read_dispatch_{number}',
                  original._dequant_dense(None, cache.view(torch.bfloat16), table, lengths),
                  candidate._dequant_dense(None, cache.view(torch.bfloat16), table, lengths, metadata), records)
        for number in [3, 4]:
            cache, table, cumulative, centroid, total, maximum, lengths = groups[number]
            padded = torch.zeros(len(lengths), 32, dtype=torch.int32, device='npu')
            padded[:, :table.shape[1]] = table
            equal(f'padded_block_table_c{len(lengths)}',
                  original._dequant_dense(None, cache.view(torch.bfloat16), padded, lengths),
                  candidate._dequant_dense(None, cache.view(torch.bfloat16), padded, lengths,
                                         SimpleNamespace()), records)
        # Deterministic coverage of all 256 packed byte codes, with selected pages.
        cache = torch.randint(0, 256, (2, 128, 2, 66), dtype=torch.uint8)
        cache[0, :4, 0, :64] = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(4, 64)
        norm = torch.ones(2, 128, 2, 1, dtype=torch.bfloat16)
        norm[0, :4, 0, 0] = torch.tensor([0.25, 1.0, 65504.0, 2**-20], dtype=torch.bfloat16)
        cache[..., 64:] = norm.view(torch.uint8)
        cache = cache.to('npu').view(torch.bfloat16)
        table = torch.tensor([[0, 1]], dtype=torch.int32, device='npu')
        metadata = SimpleNamespace()
        equal('all_256_codes', original._dequant_dense(None, cache, table, [4]),
              candidate._dequant_dense(None, cache, table, [4], metadata), records)
        plan = metadata._kvtq_read_plans['decode'][1]
        table[0, 0] = 1
        equal('same_lengths_new_physical_page', original._dequant_dense(None, cache, table, [4]),
              candidate._dequant_dense(None, cache, table, [4], metadata), records)
        assert metadata._kvtq_read_plans['decode'][1] is plan
        equal('same_metadata_changed_lengths', original._dequant_dense(None, cache, table, [129]),
              candidate._dequant_dense(None, cache, table, [129], metadata), records)
        assert metadata._kvtq_read_plans['decode'][1] is not plan
        for number in [3, 4]:
            cache, table, cumulative, cent, total, maximum, lengths = groups[number]
            batch = len(lengths)
            generator = torch.Generator().manual_seed(20261011)
            query = torch.randn(batch, 8, 128, generator=generator).to(torch.bfloat16).to('npu')
            attention = SimpleNamespace(key_cache=cache.view(torch.bfloat16),
                value_cache=cache.view(torch.bfloat16), num_kv_heads=2, num_heads=8,
                scale=1 / math.sqrt(128))
            metadata = SimpleNamespace(seq_lens_list=lengths, block_tables=table)
            equal(f'FIA_decode_c{batch}_host_lengths',
                  original._tq_decode(attention, query, metadata, torch.empty_like(query)),
                  candidate._tq_decode(attention, query, metadata, torch.empty_like(query)), records)
        # One decode plus two prefill chunks, each with an existing KV history.
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        cache = groups[1][0]
        table = groups[1][1][:3].contiguous()
        query = torch.randn(9, 8, 128).to(torch.bfloat16).to('npu')
        attention = SimpleNamespace(key_cache=cache.view(torch.bfloat16),
            value_cache=cache.view(torch.bfloat16), num_kv_heads=2, num_heads=8,
            scale=1 / math.sqrt(128))
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to('npu')
        metadata = SimpleNamespace(seq_lens_list=[7, 20, 33], block_tables=table,
            attn_state=AscendAttentionState.ChunkedPrefill, num_decode_tokens=1,
            num_decodes=1, num_prefills=2, actual_seq_lengths_q=[1, 4, 9], attn_mask=mask)
        equal('mixed_decode_prefill_with_history_sparse3',
              original._tq_forward_fia(attention, query, query, query, metadata, torch.empty_like(query)),
              candidate._tq_forward_fia(attention, query, query, query, metadata, torch.empty_like(query)), records)
    output = root / f'verify/integration_bits{bits}.json'
    output.write_text(json.dumps({'mse_bits': bits, 'passed': len(records), 'records': records,
        'scope': 'strict BF16 bit equality at real read and FIA integration APIs'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
