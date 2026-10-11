#!/usr/bin/env python3
"""Durable in-container gates, whole-model comparisons, profiling and summary."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path('/ws/opt_20261011')
PYTHON = '/ws/.venv/bin/python'
CHILD = None


def status(stage, **details):
    record = {'updated_utc': datetime.now(timezone.utc).isoformat(),
              'stage': stage, 'pipeline_pid': os.getpid(), **details}
    path = ROOT / 'background-status.json'
    path.with_suffix('.tmp').write_text(json.dumps(record, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(record), flush=True)


def run(name, command, env=None):
    global CHILD
    status(name, command=command)
    logs = ROOT / 'logs/background'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f'{name}.command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (logs / f'{name}.log').open('w') as output:
        CHILD = subprocess.Popen(command, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        if name == 'model_matrix':
            (ROOT / 'model-controller.pid').write_text(str(CHILD.pid) + '\n')
        code = CHILD.wait()
        CHILD = None
    if code:
        raise RuntimeError(f'{name} exited {code}; retained {logs / (name + ".log")}')


def summarize():
    model_status = json.loads((ROOT / 'status.json').read_text())
    assert model_status['stage'] == 'model_ab_finished', model_status
    result_dir = Path(model_status['result_dir'])
    rows = []
    for scene, input_len, output_len, concurrency in [(1, 20, 20, 16), (2, 2000, 200, 60)]:
        matched = None
        for mode in ['bf16', 'optimized', 'original']:
            records = [json.loads((result_dir / f's{scene}_{mode}_in{input_len}_out{output_len}_c{concurrency}_r{n}.json').read_text())
                       for n in [1, 2, 3]]
            assert all(r['complete_cohort'] and r['successful'] == concurrency and not r['failed'] for r in records)
            hashes = [r['payload_sha256'] for r in records]
            if matched is None:
                matched = hashes
            assert hashes == matched, (scene, mode, 'payload mismatch')
            requests = [request for r in records for request in r['requests']]
            rows.append({'scene': scene, 'mode': mode, 'input_len': input_len,
                         'output_len': output_len, 'concurrency': concurrency,
                         'successful': len(requests), 'duration_s': sum(r['duration_s'] for r in records),
                         'output_tps': len(requests) * output_len / sum(r['duration_s'] for r in records),
                         'mean_ttft_s': sum(r['ttft_s'] for r in requests) / len(requests),
                         'mean_tpot_ms': sum(r['tpot_s'] for r in requests) * 1000 / len(requests)})
    original = json.loads((result_dir / 'precision_model_original.json').read_text())['records']
    optimized = json.loads((result_dir / 'precision_model_optimized.json').read_text())['records']
    precision = []
    for a, b in zip(original, optimized, strict=True):
        assert a['payload_sha256'] == b['payload_sha256']
        at = a['response']['choices'][0]['logprobs']['tokens']
        bt = b['response']['choices'][0]['logprobs']['tokens']
        precision.append({'input_len': a['input_len'], 'output_len': a['output_len'],
                          'seed_offset': a['seed_offset'], 'tokens_equal': at == bt,
                          'matched_tokens': sum(x == y for x, y in zip(at, bt, strict=True))})
    summary = {'scope': 'paired same-host formal cohorts; warmup, precision and profiling excluded',
               'physical_cards': os.environ['TASK_NPU_CARDS'], 'rows': rows,
               'greedy_samples': precision,
               'accuracy_limit': 'four greedy samples, not a semantic quality benchmark'}
    (result_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# TQ 整网优化后台实测', '', '| 场景 | 模式 | 输出 tokens/s | TTFT s | TPOT ms | 成功请求 |',
              '|---|---|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {row['scene']} | {row['mode']} | {row['output_tps']:.3f} | {row['mean_ttft_s']:.4f} | {row['mean_tpot_ms']:.3f} | {row['successful']} |")
    report.extend(['', '预热、profiling 和独立精度请求均排除。四组贪心输出结果见 summary.json，不能代替模型质量评测。',
                   '原始逐请求结果、服务日志、启动参数及 profiling 原始/解析数据均保留。'])
    (result_dir / 'background-results.md').write_text('\n'.join(report) + '\n')
    return model_status, result_dir


def main():
    lock = (ROOT / 'background-pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'background-pipeline.pid').write_text(str(os.getpid()) + '\n')
    for name in ['bootstrap.exit', 'tq-op-build.exit', 'model-hash.exit']:
        assert (Path('/ws') / name).read_text().strip() == '0', name
    candidate = ROOT / 'candidate/integration/kvtq_read.py'
    verified = ROOT / 'verify/paged_read_triton_ascend_impl.py'
    assert candidate.read_bytes() == verified.read_bytes(), 'candidate differs from seven-case verified V1'
    env = dict(os.environ, VLLM_ASCEND_KVTQ_STORE='0', VLLM_ASCEND_KVTQ='0',
               ASCEND_RT_VISIBLE_DEVICES=os.environ['TASK_NPU_CARDS'],
               TORCH_EXTENSIONS_DIR='/ws/torch-extensions',
               TRITON_CACHE_DIR=str(ROOT / 'model-triton-cache'),
               PYTHONPATH=str(ROOT / 'candidate/integration') + ':/ws/source/vllm:/ws/source/vllm-ascend')
    env.pop('ASCEND_LAUNCH_BLOCKING', None)
    for bits in [4, 2, 3]:
        run(f'integration_bits{bits}', [PYTHON, str(ROOT / 'scripts/verify_integration.py')],
            dict(env, VLLM_ASCEND_KVTQ_BITS=str(bits)))
        verified_result = json.loads((ROOT / f'verify/integration_bits{bits}.json').read_text())
        assert verified_result['passed'] == (15 if bits == 4 else 1), verified_result
    status('integration_verified', candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    run('model_matrix', [PYTHON, '-u', str(ROOT / 'scripts/run_model_ab.py')], env=dict(env, VLLM_ASCEND_KVTQ_BITS='4'))
    model_status, result_dir = summarize()
    profile_root = ROOT / 'profiles112' / model_status['run_id']
    run('offline_parse', [PYTHON, str(ROOT / 'scripts/analyse_profiles.py'), '--root', str(profile_root),
                          '--modes', 'original', 'optimized'], env=env)
    run('profile_summary', [PYTHON, str(ROOT / 'scripts/summarize_profiles.py'), '--root', str(profile_root),
                            '--modes', 'original', 'optimized', '--output', str(result_dir / 'profile-summary.json')], env=env)
    status('finished', result_dir=str(result_dir), report=str(result_dir / 'background-results.md'),
           note='computed local artifacts; remote publication of final results remains a separate verified step')
    (ROOT / 'background.exit').write_text('0\n')


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        with (ROOT / 'issues.jsonl').open('a') as output:
            output.write(json.dumps({'id': 'background-pipeline', 'discovered_utc': datetime.now(timezone.utc).isoformat(),
                'phenomenon_and_impact': repr(error), 'evidence': 'logs/background', 'confirmed_cause': None,
                'handling': 'stop only owned subprocess group, retain all evidence', 'status': 'unresolved',
                'remaining': 'inspect failure, fix cause, launch a new recorded run'}) + '\n')
        (ROOT / 'background.exit').write_text('1\n')
        raise
    finally:
        if CHILD and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait()
