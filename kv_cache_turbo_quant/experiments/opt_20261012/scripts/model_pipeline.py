#!/usr/bin/env python3
"""Select a bit-exact long reader; gate integration; run paired model trials."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def status(root, stage, **details):
    record = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
              'pipeline_pid': os.getpid(), **details}
    (root / 'background-status.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


def run(root, name, command, env):
    status(root, name, command=command)
    directory = root / 'logs/gates'
    directory.mkdir(exist_ok=True, parents=True)
    (directory / f'{name}.command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (directory / f'{name}.log').open('w') as output:
        child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    (directory / f'{name}.exit').write_text(str(code) + '\n')
    if code:
        raise RuntimeError(f'{name} failed with exit {code}; retained log')


def make_candidate(root):
    choices = []
    for screen in [root, root / 'screen2', root / 'screen3']:
        records = json.loads((screen / 'micro-results.json').read_text())
        baseline = next(r for r in records if r['candidate'] == 'baseline_v1')
        assert baseline['status'] == 'pass' and len(baseline['precision']) == 7
        base_ms = baseline['timings'][1]['median_ms']
        for record in records:
            if record['candidate'] == 'baseline_v1' or record['status'] != 'pass':
                continue
            assert len(record['precision']) == 7 and all(p['mismatch_bits'] == 0 for p in record['precision'])
            choices.append({'path': str(screen / 'candidates' / (record['candidate'] + '.py')),
                            'long_ms': record['timings'][1]['median_ms'],
                            'paired_baseline_ms': base_ms,
                            'ratio': record['timings'][1]['median_ms'] / base_ms})
    selected = min(choices, key=lambda choice: choice['ratio'])
    assert selected['ratio'] < 0.98, 'no credible long-read candidate gain'
    baseline = (root / 'candidates/baseline_v1.py').read_text()
    source = Path(selected['path']).read_text()
    first = source.index('@triton.jit')
    kernels = source[first:source.index('class ModelNew')].replace('_paged_read4', '_long_paged_read4')
    long_forward = source[source.index('    def forward('):].replace('    def forward(', '    def _long_read(', 1)
    long_forward = long_forward.replace('_paged_read4', '_long_paged_read4')
    new = baseline.replace('class ModelNew', kernels + 'class ModelNew', 1)
    needle = '        heads = cache_u8.shape[2]\n'
    dispatch = ('        # A one-page context keeps the verified V1 kernel and allocation.\n'
                '        if max_seq_len > cache_u8.shape[1]:\n'
                '            return self._long_read(cache_u8, block_table, cumulative, centroid_bits,\n'
                '                                   total_tokens, max_seq_len, seq_lens_list)\n')
    new = new.replace(needle, dispatch + needle, 1) + '\n' + long_forward
    integration = root / 'candidate/integration'
    shutil.copytree('/ws/opt_20261011/candidate/integration', root / 'baseline/integration', dirs_exist_ok=True)
    shutil.copytree(root / 'baseline/integration', integration, dirs_exist_ok=True)
    (integration / 'kvtq_read.py').write_text(new)
    verify = root / 'verify'
    verify.mkdir(exist_ok=True)
    for name in ['paged_read.py', 'paged_read_torch.py', 'kvtq_store_reference.py']:
        shutil.copy2(Path('/ws/opt_20261011/verify') / name, verify / name)
    (verify / 'paged_read_triton_ascend_impl.py').write_text(new)
    shutil.copy2('/ws/opt_20261011/source_expected.json', root / 'source_expected.json')
    selected['hybrid_sha256'] = hashlib.sha256(new.encode()).hexdigest()
    selected['short_path'] = 'unchanged V1 kernel; max context <= paged block size'
    selected['status'] = 'micro-selected; not yet adopted, whole-model gate pending'
    (root / 'selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    return selected


def summarize(root):
    model_status = json.loads((root / 'status.json').read_text())
    assert model_status['stage'] == 'model_ab_finished', model_status
    results = Path(model_status['result_dir'])
    rows = []
    for scene, input_len, output_len, concurrency in [(1, 20, 20, 16), (2, 2000, 200, 60)]:
        matched = None
        for mode in ['bf16', 'original', 'optimized']:
            cohorts = [json.loads((results / f's{scene}_{mode}_in{input_len}_out{output_len}_c{concurrency}_r{n}.json').read_text())
                       for n in [1, 2, 3]]
            assert all(c['complete_cohort'] and c['successful'] == concurrency and not c['failed'] for c in cohorts)
            hashes = [c['payload_sha256'] for c in cohorts]
            if matched is None:
                matched = hashes
            assert hashes == matched, (scene, mode, 'payload mismatch')
            requests = [request for cohort in cohorts for request in cohort['requests']]
            duration = sum(c['duration_s'] for c in cohorts)
            rows.append({'scene': scene, 'mode': mode,
                'mode_meaning': {'bf16': 'TQ off', 'original': 'prior V1 TQ', 'optimized': 'candidate hybrid TQ'}[mode],
                'input_len': input_len, 'output_len': output_len, 'concurrency': concurrency,
                'successful': len(requests), 'duration_s': duration,
                'output_tps': len(requests) * output_len / duration,
                'mean_ttft_s': sum(r['ttft_s'] for r in requests) / len(requests),
                'mean_tpot_ms': sum(r['tpot_s'] for r in requests) * 1000 / len(requests),
                'per_round_tps': [c['output_token_throughput'] if 'output_token_throughput' in c else c['successful'] * output_len / c['duration_s'] for c in cohorts]})
    a = json.loads((results / 'precision_model_original.json').read_text())['records']
    b = json.loads((results / 'precision_model_optimized.json').read_text())['records']
    precision = []
    for old, new in zip(a, b, strict=True):
        assert old['payload_sha256'] == new['payload_sha256']
        old_tokens = old['response']['choices'][0]['logprobs']['tokens']
        new_tokens = new['response']['choices'][0]['logprobs']['tokens']
        precision.append({'input_len': old['input_len'], 'output_len': old['output_len'],
                          'tokens_equal': old_tokens == new_tokens, 'tokens': len(old_tokens)})
    gates = []
    for scene in [1, 2]:
        old = next(r for r in rows if r['scene'] == scene and r['mode'] == 'original')
        new = next(r for r in rows if r['scene'] == scene and r['mode'] == 'optimized')
        gates.append({'scene': scene, 'throughput_ratio': new['output_tps'] / old['output_tps'],
                      'throughput_pass': new['output_tps'] >= old['output_tps'] if scene == 1 else new['output_tps'] > old['output_tps'],
                      'ttft_ratio': new['mean_ttft_s'] / old['mean_ttft_s'],
                      'tpot_ratio': new['mean_tpot_ms'] / old['mean_tpot_ms']})
    summary = {'scope': 'same-host paired three formal repetitions, warmup/greedy/profiling excluded',
               'physical_cards': os.environ['TASK_NPU_CARDS'], 'rows': rows, 'greedy_samples': precision,
               'gates': gates, 'accepted': all(g['throughput_pass'] for g in gates) and all(p['tokens_equal'] for p in precision),
               'accuracy_limit': 'four single-request greedy samples, not a semantic model quality benchmark'}
    (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return results, summary


def main():
    root = Path('/ws/opt_20261012')
    lock = (root / 'model-pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'background-pipeline.pid').write_text(str(os.getpid()) + '\n')
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=os.environ['TASK_NPU_CARDS'],
               TORCH_EXTENSIONS_DIR='/ws/torch-extensions', TRITON_CACHE_DIR=str(root / 'model-triton-cache'),
               VLLM_ASCEND_KVTQ='0', VLLM_ASCEND_KVTQ_STORE='0', VLLM_ASCEND_KVTQ_BITS='4',
               PYTHONPATH=str(root / 'candidate/integration') + ':/ws/source/vllm:/ws/source/vllm-ascend')
    env.pop('ASCEND_LAUNCH_BLOCKING', None)
    python = '/ws/.venv/bin/python'
    for name in ['bootstrap.exit', 'tq-op-build.exit', 'model-hash.exit']:
        assert (Path('/ws') / name).read_text().strip() == '0', name
    selected = make_candidate(root)
    status(root, 'candidate_selected', selection=selected)
    verifier = Path('/ws/opt_20261011/verifier')
    run(root, 'freeze_reference', [python, str(verifier / 'freeze_baseline.py'), '--op_name', 'paged_read',
        '--work_dir', str(root / 'verify'), '--mode', 'user', '--source_path', '/ws/opt_20261011/verify/paged_read.py'], env)
    run(root, 'verify_kernel', [python, str(verifier / 'verify.py'), '--op_name', 'paged_read',
        '--verify_dir', str(root / 'verify'), '--output', str(root / 'verify/verify_result.json'), '--timeout', '1800'], env)
    verify = json.loads((root / 'verify/verify_result.json').read_text())
    assert verify['passed_cases'] == verify['total_cases'] == 7
    for bits in [4, 2, 3]:
        run(root, f'integration_bits{bits}', [python, str(root / 'scripts/verify_integration.py')],
            dict(env, VLLM_ASCEND_KVTQ_BITS=str(bits)))
        result = json.loads((root / f'verify/integration_bits{bits}.json').read_text())
        assert result['passed'] == (15 if bits == 4 else 1)
    run(root, 'model_matrix', [python, '-u', str(root / 'scripts/run_model_ab.py'), '--no-profile'], env)
    results, summary = summarize(root)
    status(root, 'finished' if summary['accepted'] else 'finished_gate_rejected',
           result_dir=str(results), gates=summary['gates'], accepted=summary['accepted'])
    (root / 'background.exit').write_text('0\n')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(f'signal {signum}')))
    try:
        main()
    except BaseException as error:
        root = Path('/ws/opt_20261012')
        status(root, 'failed', error_type=type(error).__name__, error=str(error))
        (root / 'background.exit').write_text('1\n')
        with (root / 'issues.jsonl').open('a') as output:
            output.write(json.dumps({'discovered_utc': datetime.now(timezone.utc).isoformat(),
                'phenomenon_and_impact': repr(error), 'raw_evidence': 'logs/gates',
                'confirmed_cause': None, 'handling': 'retain all attempts; no candidate adoption',
                'branch': 'perf/tq-read-throughput-20261012', 'base': 'eb63397b770b0cc5c02b2ad4b17aee8c1c8cf865',
                'status': 'unresolved', 'remaining': 'inspect evidence and run an independently recorded recovery'}) + '\n')
        raise
