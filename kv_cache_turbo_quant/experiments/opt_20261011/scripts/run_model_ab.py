#!/usr/bin/env python3
"""Matched whole-model trials, distinct precision and profiler runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, '/ws/scripts')
from run_matrix import MatrixRunner


class Runner(MatrixRunner):
    def __init__(self, args):
        run_id = 'model_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        super().__init__('/ws/opt_20261011', run_id, args.modes, [], 3600, None)
        self.python = '/ws/.venv/bin/python'
        self.scenarios = [(1, 20, 20, 16), (2, 2000, 200, 60)]
        self.args = args

    def issue(self, key, phenomenon, evidence, handling):
        with (self.root / 'issues.jsonl').open('a') as file:
            file.write(json.dumps({'id': key, 'run_id': self.run_id,
                'discovered_utc': datetime.now(timezone.utc).isoformat(),
                'phenomenon_and_impact': phenomenon, 'raw_evidence': str(evidence),
                'confirmed_cause': None, 'hypothesis': 'inspect retained evidence',
                'handling': handling, 'status': 'unresolved',
                'branch': 'perf/tq-store-read-20261011',
                'base_commit': 'd9e527a35100304fe6365ee2d090a08dd75c44d6',
                'verification': None, 'remaining': 'root cause and full-work validation'}) + '\n')

    def start_server(self, mode, attempt, profiling=False):
        suffix = 'profile' if profiling else 'formal'
        log_path = self.logs / f'serve_{mode}_{suffix}.log'
        self.server_log = log_path.open('w')
        cmd = ['bash', str(self.root / 'scripts/serve_optimized.sh'), mode]
        if profiling:
            cmd.append(str(self.root / 'profiles112' / self.run_id / mode))
        (self.logs / f'serve_{mode}_{suffix}.command.json').write_text(json.dumps(cmd) + '\n')
        with (self.logs / f'resource_{mode}_{suffix}.log').open('w') as file:
            subprocess.run(['npu-smi', 'info'], stdout=file, stderr=subprocess.STDOUT, check=True)
        self.server = subprocess.Popen(cmd, stdout=self.server_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.status(stage='server_starting', run_id=self.run_id, mode=mode,
                    profiling=profiling, pid=self.server.pid, log=str(log_path))
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f'{mode} startup exited {self.server.returncode}: {log_path}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18377/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(f'{mode} startup timed out')

    def prerequisites(self):
        old = Path('/ws')
        assert (old / 'bootstrap.exit').read_text().strip() == '0'
        assert (old / 'tq-op-build.exit').read_text().strip() == '0'
        with socket.socket() as port:
            port.bind(('127.0.0.1', 18377))
        source = json.loads((self.root / 'source_expected.json').read_text())
        for name, expected in source['files'].items():
            assert hashlib.sha256((old / 'source/kv_cache_turbo_quant' / name).read_bytes()).hexdigest() == expected, name
        verified = json.loads((self.root / 'verify/verify_result.json').read_text())
        assert verified['passed_cases'] == verified['total_cases'] == 7
        candidate = self.root / 'candidate/integration/kvtq_read.py'
        assert candidate.read_bytes() == (self.root / 'verify/paged_read_triton_ascend_impl.py').read_bytes()
        env = dict(os.environ, TORCH_EXTENSIONS_DIR='/ws/torch-extensions',
                   ASCEND_RT_VISIBLE_DEVICES=os.environ.get('TASK_NPU_CARDS', '2,5'), VLLM_ASCEND_KVTQ_BITS='4',
                   PYTHONPATH='/root/kvtq_integration:/ws/source/vllm:/ws/source/vllm-ascend')
        for name, script in [('glue', '/root/kvtq_integration/torch_ext/test_glue.py'),
                             ('roundtrip', '/root/kvtq_integration/test_store_roundtrip.py')]:
            self.status(stage='precision', run_id=self.run_id, check=name)
            with (self.logs / f'precision_{name}.log').open('w') as file:
                subprocess.run([self.python, script], env=env, stdout=file,
                               stderr=subprocess.STDOUT, check=True, timeout=600)
        with (self.logs / 'environment.log').open('w') as file:
            subprocess.run([self.python, '/ws/scripts/collect_environment.py'], env=env,
                           stdout=file, stderr=subprocess.STDOUT, check=True)

    def run(self):
        self.results.mkdir(exist_ok=True, parents=True)
        self.logs.mkdir(exist_ok=True, parents=True)
        parameters = {'run_id': self.run_id, 'modes': self.modes,
            'scenarios': self.scenarios, 'measured_repetitions': 3,
            'warmup_cohorts_per_scenario_mode': 1,
            'physical_cards': list(map(int, os.environ.get('TASK_NPU_CARDS', '2,5').split(','))),
            'timeout_s': self.request_timeout, 'profiler_active_during_formal': False,
            'base_commit': 'd9e527a35100304fe6365ee2d090a08dd75c44d6',
            'script_sha256': {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.root / 'scripts').glob('*.py')},
            'candidate_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.root / 'candidate/integration').glob('*.py')}}
        (self.logs / 'run_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
        self.prerequisites()
        for mode in self.modes:
            self.start_server(mode, 0)
            assert self.run_client(mode, 1, 64, 16, f'smoke_{mode}', timeout=300) == 0
            cmd = [self.python, str(self.root / 'scripts/check_model_output.py'),
                   '--mode', mode, '--output', str(self.results / f'precision_model_{mode}.json')]
            with (self.logs / f'precision_model_{mode}.command.json').open('w') as file:
                file.write(json.dumps(cmd) + '\n')
            with (self.logs / f'precision_model_{mode}.log').open('w') as file:
                subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT, check=True, timeout=3600)
            for scenario, input_len, output_len, concurrency in self.scenarios:
                prefix = f's{scenario}_{mode}_in{input_len}_out{output_len}_c{concurrency}'
                assert self.run_client(mode, concurrency, input_len, output_len, f'{prefix}_warmup') == 0
                for repeat in range(1, 4):
                    assert self.run_client(mode, concurrency, input_len, output_len, f'{prefix}_r{repeat}') == 0
            self.stop_server()
            if not self.args.no_profile and mode in ('original', 'optimized'):
                self.start_server(mode, 0, profiling=True)
                assert self.run_client(mode, 60, 2000, 8, f'profile_warmup_{mode}') == 0
                cmd = [self.python, str(self.root / 'scripts/profile_load.py'), '--mode', mode,
                       '--output', str(self.results / f'profile_load_{mode}.json')]
                (self.logs / f'profile_load_{mode}.command.json').write_text(json.dumps(cmd) + '\n')
                self.status(stage='profile_collecting', run_id=self.run_id, mode=mode)
                with (self.logs / f'profile_load_{mode}.log').open('w') as file:
                    subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT, check=True, timeout=1200)
                self.stop_server()
        self.status(stage='model_ab_finished', run_id=self.run_id, result_dir=str(self.results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', choices=['bf16', 'original', 'optimized'],
                        default=['bf16', 'optimized', 'original'])
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    runner = Runner(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage='controller_failed', run_id=runner.run_id,
                      error_type=type(error).__name__, error=str(error))
        runner.issue('model-controller', repr(error), runner.logs,
                     'retain all results and stop only owned task process groups')
        raise
    finally:
        runner.stop_server()


if __name__ == '__main__':
    main()
