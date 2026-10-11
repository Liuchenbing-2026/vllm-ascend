#!/usr/bin/env python3
"""Screen unchanged 4-bit reads against the frozen original integer oracle."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

import torch
import torch_npu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(os.environ.get('TASK_SCREEN_ROOT', '/ws/opt_20261012'))
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    (root / 'background-pipeline.pid').write_text(str(os.getpid()) + '\n')
    sys.path.insert(0, '/ws/opt_20261011/verify')
    from paged_read_torch import Model, get_input_groups
    groups = get_input_groups()
    reference = Model()
    expected = [reference(*args) for args in groups]
    torch.npu.synchronize()
    paths = sorted((root / 'candidates').glob('*.py'), key=lambda p: (p.stem != 'baseline_v1', p.stem))
    results = []
    for number, path in enumerate(paths):
        record = {'candidate': path.stem, 'precision': [], 'timings': []}
        (root / 'status.json').write_text(json.dumps({'stage': 'micro_screening', 'candidate': path.stem, 'pid': os.getpid()}) + '\n')
        try:
            candidate = load(f'candidate_{number}', path).ModelNew()
            for case, (args, golden) in enumerate(zip(groups, expected, strict=True), 1):
                actual = candidate(*args)
                torch.npu.synchronize()
                mismatch = (actual != golden).sum().item()
                record['precision'].append({'case': case, 'shape': list(actual.shape), 'mismatch_bits': mismatch})
                assert actual.shape == golden.shape and actual.dtype == golden.dtype and mismatch == 0, record['precision'][-1]
            for case in [4, 5]:
                args = groups[case - 1]
                for _ in range(5):
                    candidate(*args)
                torch.npu.synchronize()
                samples = []
                for _ in range(9):
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        candidate(*args)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 10)
                record['timings'].append({'case': case, 'input_lengths': args[-1], 'heads': args[0].shape[2],
                    'event_elapsed_ms_per_call': samples, 'median_ms': statistics.median(samples),
                    'scope': 'NPU event interval across 10 Python dispatches; screening only, not kernel-only profiling or whole-model throughput'})
            record['status'] = 'pass'
        except Exception as error:
            record.update(status='fail', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            (root / 'logs' / f'{path.stem}.failure.log').write_text(record['traceback'])
            with (root / 'issues.jsonl').open('a') as file:
                file.write(json.dumps({'discovered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'candidate': path.stem, 'phenomenon_and_impact': str(error), 'raw_evidence': f'logs/{path.stem}.failure.log',
                    'confirmed_cause': None, 'handling': 'retain failed candidate; exclude from selection',
                    'branch': 'perf/tq-read-throughput-20261012', 'base': 'eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865',
                    'verification': record['precision'], 'status': 'unresolved', 'remaining': 'inspect retained traceback'}) + '\n')
            # A device failure invalidates subsequent measurements in this process.
            if any(marker in str(error) for marker in ['507035', 'ACL_ERROR', 'NPU error']):
                results.append(record)
                (root / 'micro-results.json').write_text(json.dumps(results, indent=2) + '\n')
                raise
        results.append(record)
        (root / 'micro-results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    (root / 'status.json').write_text(json.dumps({'stage': 'micro_finished', 'passed': sum(r['status'] == 'pass' for r in results), 'total': len(results)}) + '\n')
    (root / 'micro.exit').write_text('0\n')


if __name__ == '__main__':
    main()
