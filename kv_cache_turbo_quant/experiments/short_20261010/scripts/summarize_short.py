#!/usr/bin/env python3
"""Validate exact work and payload pairing; summarize explicit formal cohorts."""
import argparse
import hashlib
import json
import pathlib
import re
import statistics


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered)-1)*quantile
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def summarize(root, run_id):
    rows = []
    for scenario, input_len, output_len, concurrency in [(1, 20, 20, 16), (2, 2000, 200, 60)]:
        row = {'scenario': scenario, 'input_len': input_len, 'output_len': output_len,
               'concurrency': concurrency, 'repetitions': 3}
        payload_reference = None
        for mode in ['bf16', 'store4']:
            cohorts = []
            requests = []
            for repeat in range(1, 4):
                path = root/'results'/run_id/f's{scenario}_{mode}_in{input_len}_out{output_len}_c{concurrency}_r{repeat}.json'
                raw = path.read_bytes()
                result = json.loads(raw)
                assert (result['mode'], result['input_len'], result['output_len'], result['concurrency'], result['num_prompts']) == (mode, input_len, output_len, concurrency, concurrency)
                assert result['seed'] == 20261010
                assert len(result['requests']) == len(result['payload_sha256']) == concurrency
                assert [r['id'] for r in result['requests']] == list(range(concurrency))
                if payload_reference is None:
                    payload_reference = result['payload_sha256']
                assert payload_reference == result['payload_sha256'], path
                successful = [r for r in result['requests'] if r['status'] == 'success']
                assert result['successful'] == len(successful)
                assert result['failed'] == concurrency-len(successful)
                assert result['complete_cohort'] == (len(successful) == concurrency)
                for request in successful:
                    assert request['usage']['prompt_tokens'] == input_len
                    assert request['usage']['completion_tokens'] == output_len
                    assert request['usage']['total_tokens'] == input_len+output_len
                    assert request['http_status'] == 200
                    assert request['finish_reason'] == 'length'
                if result['complete_cohort']:
                    assert abs(result['output_throughput_tps']-concurrency*output_len/result['duration_s']) < 1e-9
                else:
                    assert result['output_throughput_tps'] is None
                result['source_path'] = str(path.relative_to(root))
                result['source_sha256'] = hashlib.sha256(raw).hexdigest()
                cohorts.append(result)
                requests.extend(successful)
            complete = all(c['complete_cohort'] for c in cohorts)
            duration = sum(c['duration_s'] for c in cohorts)
            row[mode] = {
                'complete': complete, 'successful': sum(c['successful'] for c in cohorts),
                'failed': sum(c['failed'] for c in cohorts), 'cohort_count': len(cohorts),
                'sum_cohort_duration_s': duration,
                'output_throughput_tps': 3*concurrency*output_len/duration if complete else None,
                'total_throughput_tps': 3*concurrency*(input_len+output_len)/duration if complete else None,
                'mean_ttft_s': statistics.mean(r['ttft_s'] for r in requests) if complete else None,
                'mean_tpot_ms': 1000*statistics.mean(r['tpot_s'] for r in requests) if complete else None,
                'mean_latency_s': statistics.mean(r['latency_s'] for r in requests) if complete else None,
                'p99_ttft_s': percentile([r['ttft_s'] for r in requests], .99) if complete else None,
                'p99_tpot_ms': 1000*percentile([r['tpot_s'] for r in requests], .99) if complete else None,
                'throughput_min_tps': min(c['output_throughput_tps'] for c in cohorts) if complete else None,
                'throughput_max_tps': max(c['output_throughput_tps'] for c in cohorts) if complete else None,
                'cohorts': [{k: v for k, v in c.items() if k not in ['requests', 'payload_sha256']}
                            for c in cohorts],
            }
        complete_pair = row['bf16']['complete'] and row['store4']['complete']
        row['payloads_match_all_repetitions_and_modes'] = True
        row['store_vs_bf16_output_throughput_ratio'] = (row['store4']['output_throughput_tps']/row['bf16']['output_throughput_tps']
                                                       if complete_pair else None)
        row['store_vs_bf16_tpot_ratio'] = (row['store4']['mean_tpot_ms']/row['bf16']['mean_tpot_ms']
                                         if complete_pair else None)
        rows.append(row)
    return {'run_id': run_id, 'formal_cohort_count': 12,
            'warmups_excluded': True,
            'throughput_denominator': 'sum of three complete cohort HTTP-send-to-last-response durations',
            'latency_aggregation': 'pooled successful request timings, only for all-complete mode/scenario',
            'capacity': verify_capacity(root, run_id),
            'scenarios': rows}


def verify_capacity(root, run_id):
    paired = {}
    for mode in ['bf16', 'store4']:
        path = root/'logs'/run_id/f'serve_{mode}_0.log'
        raw = path.read_bytes()
        workers = {}
        evidence = []
        tokens = None
        for number, line in enumerate(raw.decode().splitlines(), 1):
            match = re.search(r'Worker_TP(\d+).*Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB for KV Cache', line)
            if match:
                workers[match[1]] = {'initial_free_gib': float(match[2]), 'kv_budget_gib': float(match[3])}
                evidence.append({'line': number, 'text': line})
            match = re.search(r'GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x', line)
            if match:
                tokens = int(match[1].replace(',', ''))
                assert int(match[2].replace(',', '')) == 4096
                evidence.append({'line': number, 'text': line})
        assert set(workers) == {'0', '1'}
        assert all(w['kv_budget_gib'] == 24 for w in workers.values())
        assert tokens is not None
        paired[mode] = {'workers': workers, 'capacity_tokens': tokens,
                        'source_path': str(path.relative_to(root)),
                        'source_sha256': hashlib.sha256(raw).hexdigest(), 'evidence': evidence}
    assert paired['bf16']['workers'] == paired['store4']['workers']
    paired['store_vs_bf16_capacity_ratio'] = paired['store4']['capacity_tokens']/paired['bf16']['capacity_tokens']
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=pathlib.Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = summarize(args.root, args.run_id)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for row in result['scenarios']:
        print(json.dumps({k: v for k, v in row.items() if k not in ['bf16', 'store4']}))
        for mode in ['bf16', 'store4']:
            print(mode, json.dumps({k: v for k, v in row[mode].items() if k != 'cohorts'}))


if __name__ == '__main__':
    main()
