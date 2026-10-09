#!/usr/bin/env python3
"""Validate paired manual KV budgets and record original startup evidence."""
import argparse
import hashlib
import json
import pathlib
import re


def read_log(path):
    payload = path.read_bytes()
    text = payload.decode()
    workers = {}
    evidence = []
    capacity = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r'Worker_TP(\d+).*Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB for KV Cache', line)
        if match:
            rank, free, budget = match.groups()
            value = {'initial_free_gib': float(free), 'kv_budget_gib': float(budget)}
            if rank in workers and workers[rank] != value:
                raise ValueError(f'Inconsistent worker budget: {path}:{number}')
            workers[rank] = value
            evidence.append({'line': number, 'text': line})
        match = re.search(r'GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x', line)
        if match:
            capacity = {'tokens': int(match[1].replace(',', '')),
                        'max_request_tokens': int(match[2].replace(',', '')),
                        'max_resident_concurrency': float(match[3])}
            evidence.append({'line': number, 'text': line})
    assert set(workers) == {'0', '1'}, path
    assert capacity and capacity['max_request_tokens'] == 132096, path
    assert all(value['kv_budget_gib'] == 24 for value in workers.values()), path
    return {'log_path': str(path), 'log_snapshot_sha256': hashlib.sha256(payload).hexdigest(),
            'workers': workers, 'capacity': capacity, 'evidence': evidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16-log', required=True, type=pathlib.Path)
    parser.add_argument('--store4-log', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    bf16, store4 = read_log(args.bf16_log), read_log(args.store4_log)
    assert bf16['workers'] == store4['workers'], 'Initial available memory or budget differs'
    report = {'bf16': bf16, 'store4': store4,
              'store_vs_bf16_capacity_ratio': store4['capacity']['tokens'] / bf16['capacity']['tokens'],
              'capacity_is_not_throughput': True}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['store_vs_bf16_capacity_ratio', 'capacity_is_not_throughput']}))


if __name__ == '__main__':
    main()
