"""Bitwise cache-state gates and isolated writer timing; no model speed claims."""
import json
import os
from pathlib import Path
import time

import torch
import torch_npu


def write(kind, cache, slots, packed):
    if kind == 'u8_index_copy':
        cache.index_copy_(0, slots, packed)
    elif kind == 'bf16_index_copy':
        cache.view(torch.bfloat16).index_copy_(0, slots, packed.view(torch.bfloat16))
    elif kind == 'bf16_scatter_nd':
        torch_npu.npu_scatter_nd_update_(cache.view(torch.bfloat16), slots[:, None], packed.view(torch.bfloat16))
    elif kind == 'u8_scatter_nd':
        torch_npu.npu_scatter_nd_update_(cache, slots[:, None], packed)


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20261013)
    root = Path('/ws/diagnosis_20261013')
    kinds = ['u8_index_copy', 'bf16_index_copy', 'bf16_scatter_nd', 'u8_scatter_nd']
    records = []
    for heads in [1, 2, 4]:
        for tokens in [1, 16, 17, 63, 64, 129, 4096, 8000]:
            capacity = max(1024, tokens * 2)
            initial = torch.randint(0, 256, (capacity, heads, 66), dtype=torch.uint8)
            packed = torch.randint(0, 256, (tokens, heads, 66), dtype=torch.uint8)
            # Packed indices may be NaNs when viewed as BF16. This gate requires
            # preservation of every original byte, including every BF16 pattern.
            slots = torch.randperm(capacity)[:tokens].to(torch.int64)
            expected = initial.clone().index_copy_(0, slots, packed)
            packed_npu, slots_npu = packed.npu(), slots.npu()
            for kind in kinds:
                cache = initial.npu()
                record = {'kind': kind, 'tokens': tokens, 'heads': heads}
                try:
                    for _ in range(3):
                        write(kind, cache, slots_npu, packed_npu)
                    torch.npu.synchronize()
                    assert torch.equal(cache.cpu(), expected), record
                    record['passed'] = True
                except Exception as error:
                    record.update(passed=False, error=repr(error))
                records.append(record)
                print(json.dumps(record), flush=True)
    supported = [kind for kind in kinds if all(r['passed'] for r in records if r['kind'] == kind)]
    timings = []
    for tokens in [16, 60, 128, 256, 4096, 8000]:
        cache = torch.zeros(2033536, 2, 66, dtype=torch.uint8, device='npu')
        packed = torch.randint(0, 256, (tokens, 2, 66), dtype=torch.uint8, device='npu')
        slots = torch.arange(tokens, dtype=torch.int64, device='npu')
        for kind in supported:
            for _ in range(5):
                write(kind, cache, slots, packed)
            torch.npu.synchronize()
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            for _ in range(100):
                write(kind, cache, slots, packed)
            end.record()
            end.synchronize()
            row = {'kind': kind, 'tokens': tokens, 'iterations': 100,
                   'device_event_mean_us': begin.elapsed_time(end) * 10,
                   'host_and_device_mean_us': (time.perf_counter() - started) * 10000}
            timings.append(row)
            print(json.dumps(row), flush=True)
    output = {'records': records, 'supported': supported, 'timings': timings,
              'scope': 'byte equality including untouched cache slots; writer microbenchmark excludes quantization and does not measure model throughput'}
    (root / 'probe-write.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
