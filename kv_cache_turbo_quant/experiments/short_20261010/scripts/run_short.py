#!/usr/bin/env python3
"""Run both short scenarios, retaining warmups and three measured cohorts."""
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone


def main():
    sys.path.insert(0, '/ws/scripts')
    from run_matrix import MatrixRunner

    class ShortRunner(MatrixRunner):
        def __init__(self):
            run_id = datetime.now(timezone.utc).strftime('short_%Y%m%dT%H%M%SZ')
            super().__init__('/ws/short_ab_20261010', run_id,
                             ['bf16', 'store4'], [], 3600, None)
            self.python = '/ws/.venv/bin/python'
            self.scenarios = [(1, 20, 20, 16), (2, 2000, 200, 60)]

        def start_server(self, mode, attempt):
            log_path = self.logs / f'serve_{mode}_{attempt}.log'
            self.server_log = log_path.open('w')
            command = ['bash', str(self.root/'scripts/serve_short.sh'),
                       '1' if mode == 'store4' else '0']
            (self.logs/f'serve_{mode}_{attempt}.command.json').write_text(json.dumps(command, indent=2)+'\n')
            self.server = subprocess.Popen(command, stdout=self.server_log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            self.status(stage='server_starting', mode=mode, pid=self.server.pid,
                        attempt=attempt, log=str(log_path))
            deadline = time.monotonic()+1200
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
            raise RuntimeError(f'{mode} startup exceeded 1200s')

        def prerequisites(self):
            old = pathlib.Path('/ws')
            assert (old/'bootstrap.exit').read_text().strip() == '0'
            assert (old/'tq-op-build.exit').read_text().strip() == '0'
            with socket.socket() as port:
                port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port.bind(('127.0.0.1', 18377))
            previous = json.loads((old/'artifacts/environment.json').read_text())
            current = {package: metadata.version(package) for package in previous['packages']}
            assert current == previous['packages'], current
            native = {}
            for name, expected in previous['native_artifacts'].items():
                actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
                assert actual == expected['sha256'], name
                native[name] = actual
            manifest = json.loads((self.root/'source_expected.json').read_text())
            for name, expected in manifest['files'].items():
                assert hashlib.sha256((old/'source/kv_cache_turbo_quant'/name).read_bytes()).hexdigest() == expected, name
            plugin = pathlib.Path('/root/kvtq_integration')
            for path in (old/'source/kv_cache_turbo_quant/code/integration').glob('*.py'):
                assert hashlib.sha256((plugin/path.name).read_bytes()).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest(), path.name
            (self.logs/'runtime_verified.json').write_text(json.dumps({
                'captured_utc': datetime.now(timezone.utc).isoformat(),
                'packages': current, 'native_hashes': native,
                'tq_source_verified': manifest['commit'],
                'tq_files_verified': len(manifest['files']),
                'reuse': 'preserved compiled runtime; no rebuild in this follow-up',
                'client_sha256': hashlib.sha256((old/'scripts/bench_client.py').read_bytes()).hexdigest(),
                'server_sha256': hashlib.sha256((self.root/'scripts/serve_short.sh').read_bytes()).hexdigest(),
            }, indent=2)+'\n')
            env = dict(os.environ, TORCH_EXTENSIONS_DIR='/ws/torch-extensions',
                       VLLM_ASCEND_KVTQ_BITS='4', ASCEND_RT_VISIBLE_DEVICES='0,1',
                       PYTHONPATH='/ws/source/vllm:/ws/source/vllm-ascend:/root/kvtq_integration:'+os.environ.get('PYTHONPATH', ''))
            for name, script in [('glue', plugin/'torch_ext/test_glue.py'),
                                 ('roundtrip', plugin/'test_store_roundtrip.py')]:
                self.status(stage='precision', check=name)
                with (self.logs/f'precision_{name}.log').open('w') as log:
                    subprocess.run([self.python, str(script)], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=600)

        def run(self):
            self.results.mkdir(exist_ok=True, parents=True)
            self.logs.mkdir(exist_ok=True, parents=True)
            (self.logs/'run_parameters.json').write_text(json.dumps({
                'run_id': self.run_id, 'modes': self.modes,
                'scenarios': self.scenarios, 'measured_repetitions': 3,
                'warmup_cohorts_per_scenario_mode': 1,
                'max_num_seqs': 64, 'max_model_len': 4096,
                'timeout_s': self.request_timeout,
                'mode_order': 'bf16_then_store4',
                'reuse_runner': '/ws/scripts/run_matrix.py',
            }, indent=2)+'\n')
            self.prerequisites()
            for mode in self.modes:
                self.start_server(mode, 0)
                if self.run_client(mode, 1, 64, 16, f'smoke_{mode}', timeout=300):
                    raise RuntimeError(f'{mode} smoke failed')
                for scenario, input_len, output_len, concurrency in self.scenarios:
                    prefix = f's{scenario}_{mode}_in{input_len}_out{output_len}_c{concurrency}'
                    if self.run_client(mode, concurrency, input_len, output_len, f'{prefix}_warmup'):
                        raise RuntimeError(f'{prefix} warmup failed')
                    for repeat in range(1, 4):
                        if self.run_client(mode, concurrency, input_len, output_len, f'{prefix}_r{repeat}'):
                            raise RuntimeError(f'{prefix} measured repeat {repeat} failed')
                self.stop_server()
            self.status(stage='short_finished', result_dir=str(self.results),
                        note='All 12 measured cohorts completed; warmups kept separately')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    runner = ShortRunner()
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage='controller_failed', error_type=type(error).__name__, error=str(error))
        runner.issue('short-controller-failed', f'{type(error).__name__}: {error}',
                     runner.root/'status.json', 'Keep all evidence; stop only own task process groups')
        raise
    finally:
        runner.stop_server()


if __name__ == '__main__':
    main()
