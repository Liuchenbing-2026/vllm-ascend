"""Compare full byte state to a CPU pack/scatter oracle and original quantizer."""
import importlib.util
import json
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


def main():
    root = Path('/ws/diagnosis_20261013')
    sys.path.insert(0, str(root / 'candidate_write/integration'))
    from kvtq_write import PackedCacheWriter

    torch.ops.load_library('/ws/torch-extensions/turboquant_torch/turboquant_torch.so')
    torch.manual_seed(20261013)
    records = []
    writer = PackedCacheWriter()
    for heads in [1, 2, 4]:
        for tokens in [1, 16, 17, 63, 64, 129, 4096, 8000]:
            capacity = max(1024, tokens * 2)
            cache = torch.randint(0, 256, (capacity, heads, 66), dtype=torch.uint8)
            idx = torch.randint(0, 256, (tokens, heads, 64), dtype=torch.uint8)
            norm_bytes = torch.randint(0, 256, (tokens, heads, 2), dtype=torch.uint8)
            norms = norm_bytes.view(torch.bfloat16).squeeze(-1)
            for order in ['adjacent', 'reverse', 'random']:
                slots = (torch.arange(tokens) if order == 'adjacent' else
                         torch.arange(tokens - 1, -1, -1) if order == 'reverse' else
                         torch.randperm(capacity)[:tokens])
                expected = cache.clone().index_copy_(0, slots, torch.cat([idx, norm_bytes], dim=-1))
                actual = cache.npu().view(torch.bfloat16)
                index_npu, norm_npu, slots_npu = idx.npu(), norms.npu(), slots.npu()
                for _ in range(10):
                    writer(index_npu, norm_npu, slots_npu, actual)
                torch.npu.synchronize()
                mismatched = (expected != actual.view(torch.uint8).cpu()).sum().item()
                record = {'heads': heads, 'tokens': tokens, 'order': order,
                          'repeats': 10, 'mismatched_bytes': mismatched}
                records.append(record)
                print(json.dumps(record), flush=True)
                assert mismatched == 0, record
    original = load('previous_store', '/ws/c128_20261012/candidate/integration/kvtq_store.py')
    candidate = load('candidate_store', root / 'candidate_write/integration/kvtq_store.py')
    for tokens in [1, 17, 60, 128, 256, 4096]:
        x = torch.randn(tokens, 2, 128).to(torch.bfloat16).npu()
        initial = torch.randint(0, 256, (128, 128, 2, 66), dtype=torch.uint8)
        slots = torch.randperm(16384)[:tokens].npu()
        old_cache, new_cache = initial.npu().view(torch.bfloat16), initial.npu().view(torch.bfloat16)
        original._write_cache(SimpleNamespace(), x, old_cache, slots)
        candidate._write_cache(SimpleNamespace(), x, new_cache, slots)
        torch.npu.synchronize()
        mismatch = (old_cache.view(torch.uint8) != new_cache.view(torch.uint8)).sum().item()
        record = {'kind': 'real_quantizer_full_cache_state', 'tokens': tokens,
                  'heads': 2, 'mismatched_bytes': mismatch}
        records.append(record)
        print(json.dumps(record), flush=True)
        assert mismatch == 0, record
    (root / 'verify-write.json').write_text(json.dumps({'passed': len(records), 'records': records,
        'scope': 'all cache bytes and untouched slots, arbitrary BF16 bit patterns, tail tiles and physical-slot reuse'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
