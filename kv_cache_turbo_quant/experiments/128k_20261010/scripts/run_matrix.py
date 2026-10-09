#!/usr/bin/env python3
"""Run the fixed two-card benchmark inside this task's isolated runtime container."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone



def now():
    return datetime.now(timezone.utc).isoformat()


class MatrixRunner:
    def __init__(self, root, run_id, modes, concurrencies, request_timeout, precision_log_dir):
        self.root = pathlib.Path(root)
        self.run_id = run_id
        self.results = self.root / 'results' / run_id
        self.logs = self.root / 'logs' / run_id
        self.python = str(self.root / '.venv/bin/python')
        self.server = None
        self.server_log = None
        self.modes = modes
        self.concurrencies = concurrencies
        self.request_timeout = request_timeout
        self.precision_log_dir = precision_log_dir

    def status(self, **fields):
        (self.root / 'status.json').write_text(json.dumps({'updated_utc': now(), **fields}, indent=2) + '\n')
        print(json.dumps(fields), flush=True)


    def issue(self, key, phenomenon, evidence, handling):
        with (self.root / 'issues.jsonl').open('a') as file:
            file.write(json.dumps({'id': key, 'run_id': self.run_id, 'discovered_utc': now(),
                                   'phenomenon_and_impact': phenomenon, 'raw_evidence': str(evidence),
                                   'confirmed_cause': None, 'hypothesis': 'inspect server and request logs',
                                   'handling': handling, 'status': 'unresolved',
                                   'branch': 'no framework source modifications',
                                   'verification': None, 'remaining': 'root cause and successful full-work rerun'},
                                  ensure_ascii=False) + '\n')


    def stop_server(self):
        if self.server is not None:
            try:
                os.killpg(self.server.pid, signal.SIGTERM)
                self.server.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.server.wait(timeout=20)
            self.server = None
        # This process is inside the newly created task container's PID namespace.
        subprocess.run(['pkill', '-TERM', '-f', '[V]LLM::'], check=False)
        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None
        time.sleep(5)


    def start_server(self, mode, attempt):
        log_path = self.logs / f'serve_{mode}_{attempt}.log'
        self.server_log = log_path.open('w')
        command = ['bash', '/ws/scripts/serve.sh', '1' if mode == 'store4' else '0']
        (self.logs / f'serve_{mode}_{attempt}.command.json').write_text(json.dumps(command, indent=2) + '\n')
        self.server = subprocess.Popen(command, stdout=self.server_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        self.status(stage='server_starting', mode=mode, attempt=attempt, pid=self.server.pid, log=str(log_path))
        for _ in range(240):
            if self.server.poll() is not None:
                self.issue(f'startup-{mode}-{attempt}', f'Server exited {self.server.returncode} before readiness',
                      log_path, 'Stop only this task server; preserve all logs')
                raise RuntimeError(f'{mode} server startup failed: {log_path}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18377/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        self.issue(f'startup-timeout-{mode}-{attempt}', 'Server readiness exceeded 1200s', log_path,
              'Stop only this task server; preserve all logs')
        raise RuntimeError(f'{mode} startup timeout')


    def run_client(self, mode, concurrency, input_len, output_len, name, timeout=None):
        timeout = self.request_timeout if timeout is None else timeout
        output = self.results / f'{name}.json'
        command = [self.python, '/ws/scripts/bench_client.py', '--mode', mode,
                   '--concurrency', str(concurrency), '--input-len', str(input_len),
                   '--output-len', str(output_len), '--timeout', str(timeout), '--output', str(output)]
        (self.logs / f'{name}.command.json').write_text(json.dumps(command, indent=2) + '\n')
        self.status(stage='benchmark_running', mode=mode, concurrency=concurrency,
               input_len=input_len, output_len=output_len, result=str(output), started_utc=now())
        with (self.logs / f'{name}.client.log').open('w') as log:
            result = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            try:
                result.wait()
            finally:
                if result.poll() is None:
                    os.killpg(result.pid, signal.SIGTERM)
                    try:
                        result.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(result.pid, signal.SIGKILL)
                        result.wait()
        if result.returncode != 0:
            self.issue(name, f'Client exited {result.returncode}; full requested work did not all complete',
                  output if output.exists() else self.logs / f'{name}.client.log',
                  'Keep failed/timeout records; restart own server before next cohort')
        return result.returncode


    def run(self):
        self.results.mkdir(exist_ok=True, parents=True)
        self.logs.mkdir(exist_ok=True, parents=True)
        parameters = {'run_id': self.run_id, 'modes': self.modes,
                      'concurrencies': self.concurrencies,
                      'request_timeout_s': self.request_timeout,
                      'input_len': 131072, 'output_len': 1024,
                      'file_sha256': {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                                      for name in ['scripts/serve.sh', 'scripts/bench_client.py',
                                                   'artifacts/environment.json']}}
        (self.logs / 'run_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
        if (self.root / 'bootstrap.exit').read_text().strip() != '0':
            raise RuntimeError('Framework source build is not successful')
        if (self.root / 'tq-op-build.exit').read_text().strip() != '0':
            raise RuntimeError('TQ operator source build is not successful')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 18377))
        source = self.root / 'source/kv_cache_turbo_quant/code/integration'
        plugin = pathlib.Path('/root/kvtq_integration')
        plugin.mkdir(exist_ok=True)
        for path in source.glob('*.py'):
            shutil.copy2(path, plugin / path.name)
        shutil.copytree(source / 'torch_ext', plugin / 'torch_ext', dirs_exist_ok=True)
        shutil.copy2(self.root / 'source/kv_cache_turbo_quant/results/golden.py', plugin / 'golden.py')
        (plugin / 'golden').mkdir(exist_ok=True)
        shutil.copy2(self.root / 'source/kv_cache_turbo_quant/results/golden.py', plugin / 'golden/golden.py')
        env = dict(os.environ, TORCH_EXTENSIONS_DIR='/ws/torch-extensions', VLLM_ASCEND_KVTQ_BITS='4',
                   ASCEND_RT_VISIBLE_DEVICES='0,1',
                   PYTHONPATH='/ws/source/vllm:/ws/source/vllm-ascend:/root/kvtq_integration:' + os.environ.get('PYTHONPATH', ''))
        if self.precision_log_dir:
            prior = pathlib.Path(self.precision_log_dir)
            assert 'ALL GLUE TESTS PASS' in (prior / 'precision_glue.log').read_text()
            assert 'ROUNDTRIP OK' in (prior / 'precision_roundtrip.log').read_text()
            environment = json.loads((self.root / 'artifacts/environment.json').read_text())
            for package in ['torch', 'torch-npu', 'numpy', 'ml-dtypes']:
                assert metadata.version(package) == environment['packages'][package], package
            for path, artifact in environment['native_artifacts'].items():
                assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == artifact['sha256'], path
            (self.logs / 'precision_reused.json').write_text(json.dumps({
                'log_dir': str(prior), 'native_hashes_verified': True,
                'unchanged_tq_source': '8ad9ef6eaa0fdc7b4cc9acf6aaeac17fd33fa65b',
                'environment': environment['packages']}, indent=2) + '\n')
        precision_checks = [] if self.precision_log_dir else [
            ('glue', [self.python, str(plugin / 'torch_ext/test_glue.py')]),
            ('roundtrip', [self.python, str(plugin / 'test_store_roundtrip.py')]),
        ]
        for name, command in precision_checks:
            self.status(stage='precision', check=name)
            with (self.logs / f'precision_{name}.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                self.issue(f'precision-{name}', 'TQ precision prerequisite failed', self.logs / f'precision_{name}.log',
                      'Do not run performance comparison until the original criterion passes')
                raise RuntimeError(f'{name} precision failed')
        for mode in self.modes:
            attempt = 0
            self.start_server(mode, attempt)
            if self.run_client(mode, 1, 64, 16, f'smoke_short_{mode}', timeout=300):
                raise RuntimeError(f'{mode} short smoke failed')
            if self.run_client(mode, 1, 131072, 1, f'smoke_128k_{mode}'):
                raise RuntimeError(f'{mode} 128K smoke failed')
            for concurrency in self.concurrencies:
                code = self.run_client(mode, concurrency, 131072, 1024, f'{mode}_in131072_out1024_c{concurrency}')
                if code:
                    self.stop_server()
                    if concurrency != self.concurrencies[-1]:
                        attempt += 1
                        self.start_server(mode, attempt)
            self.stop_server()
        self.status(stage='matrix_finished', result_dir=str(self.results),
               note='Each JSON retains actual success/failure counts; failed cohorts have null throughput')

def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', choices=['bf16', 'store4'], default=['bf16', 'store4'])
    parser.add_argument('--concurrencies', nargs='+', type=int, choices=[1, 2, 4, 8, 16, 32],
                        default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--request-timeout', type=float, default=1800)
    parser.add_argument('--precision-log-dir')
    args = parser.parse_args()
    runner = MatrixRunner('/ws', datetime.now(timezone.utc).strftime('matrix_%Y%m%dT%H%M%SZ'),
                          args.modes, args.concurrencies, args.request_timeout, args.precision_log_dir)
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage='controller_failed', error_type=type(error).__name__, error=str(error))
        evidence = runner.logs / 'controller_failure.json'
        shutil.copy2('/ws/status.json', evidence)
        runner.issue('controller-failed', f'{type(error).__name__}: {error}',
                     evidence, 'Preserve the attempt and stop only its clients/server')
        raise
    finally:
        runner.stop_server()


if __name__ == '__main__':
    main()
