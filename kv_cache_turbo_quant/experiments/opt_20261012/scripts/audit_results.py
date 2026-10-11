#!/usr/bin/env python3
"""Independently audit completed cohorts, usage, frozen code, and token IDs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.snapshot
    assert (root / 'background.exit').read_text().strip() == '0'
    assert (root / 'background-host.exit').read_text().strip() == '0'
    state = read(root / 'background-status.json')
    assert state['stage'] == 'finished' and state['accepted']
    run_id = read(root / 'status.json')['run_id']
    results = root / 'results' / run_id
    summary = read(results / 'summary.json')
    audited = []
    formal_groups = formal_requests = 0
    for scene, inp, out, concurrency in [(1, 20, 20, 16), (2, 2000, 200, 60)]:
        expected_hashes = None
        for mode in ('bf16', 'original', 'optimized'):
            cohorts = [read(results / f's{scene}_{mode}_in{inp}_out{out}_c{concurrency}_r{r}.json')
                       for r in (1, 2, 3)]
            hashes = [c['payload_sha256'] for c in cohorts]
            if expected_hashes is None:
                expected_hashes = hashes
            assert hashes == expected_hashes
            requests = []
            for cohort in cohorts:
                assert cohort['complete_cohort'] and cohort['failed'] == 0
                assert cohort['successful'] == cohort['num_prompts'] == concurrency
                assert cohort['input_len'] == inp and cohort['output_len'] == out
                assert len(cohort['requests']) == concurrency
                for request in cohort['requests']:
                    assert request['status'] == 'success' and request['http_status'] == 200
                    assert request['usage'] == {'prompt_tokens': inp,
                        'completion_tokens': out, 'total_tokens': inp + out}
                    assert request['finish_reason'] == 'length'
                    requests.append(request)
                formal_groups += 1
            formal_requests += len(requests)
            duration = sum(c['duration_s'] for c in cohorts)
            calculated = {'scene': scene, 'mode': mode,
                'output_tps': sum(r['usage']['completion_tokens'] for r in requests) / duration,
                'mean_ttft_s': sum(r['ttft_s'] for r in requests) / len(requests),
                'mean_tpot_ms': sum(r['tpot_s'] for r in requests) * 1000 / len(requests)}
            reported = next(r for r in summary['rows'] if r['scene'] == scene and r['mode'] == mode)
            for key in ('output_tps', 'mean_ttft_s', 'mean_tpot_ms'):
                assert math.isclose(calculated[key], reported[key], rel_tol=1e-12)
            audited.append(calculated)
    assert formal_groups == 18 and formal_requests == 684
    old = read(results / 'precision_model_original.json')['records']
    new = read(results / 'precision_model_optimized.json')['records']
    assert len(old) == len(new) == 4
    token_count = 0
    for a, b in zip(old, new):
        assert a['payload_sha256'] == b['payload_sha256']
        before = a['response']['choices'][0]['logprobs']['tokens']
        after = b['response']['choices'][0]['logprobs']['tokens']
        assert before == after and len(after) == b['output_len']
        token_count += len(after)
    assert token_count == 440
    assert read(root / 'verify/verify_result.json')['passed_cases'] == 7
    for bits, cases in [(4, 15), (2, 1), (3, 1)]:
        assert read(root / f'verify/integration_bits{bits}.json')['passed'] == cases
    source = root / 'candidate/integration/kvtq_read.py'
    assert source.read_bytes() == (root / 'verify/paged_read_triton_ascend_impl.py').read_bytes()
    for name in ('kvtq_store.py', 'kvtq_vllm_plugin.py', 'torch_ext/turboquant_torch.cpp'):
        assert (root / 'baseline/integration' / name).read_bytes() == (root / 'candidate/integration' / name).read_bytes()
    for scene in (1, 2):
        prior = next(r for r in audited if r['scene'] == scene and r['mode'] == 'original')
        candidate = next(r for r in audited if r['scene'] == scene and r['mode'] == 'optimized')
        assert candidate['output_tps'] >= prior['output_tps']
        assert candidate['mean_ttft_s'] < prior['mean_ttft_s']
        assert candidate['mean_tpot_ms'] < prior['mean_tpot_ms']
    audit = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
        'formal_groups': formal_groups, 'formal_requests': formal_requests,
        'all_usage_exact': True, 'payloads_paired': True, 'greedy_samples': 4,
        'equal_greedy_token_ids': token_count, 'reference_precision_gates': '7/15/1/1 pass',
        'hybrid_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'store_plugin_quantizer_cpp_unchanged': True, 'rows': audited,
        'short_no_decline_and_long_gain': True, 'both_latency_means_improved': True,
        'limitations': 'Four greedy samples do not measure semantic model accuracy; no optimized128K test.'}
    args.output.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
