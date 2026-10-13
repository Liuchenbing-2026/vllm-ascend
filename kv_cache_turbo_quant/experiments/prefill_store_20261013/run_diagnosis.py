"""Matched warmed profiles, followed by a separate scheduling-budget screen."""
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import time
import urllib.request
from run_8k1 import Runner as BaseRunner, ROOT


class Runner(BaseRunner):
    def start_server(self, mode, attempt, budget=4096, profiling=False):
        label = f'{mode}_b{budget}_' + ('profile' if profiling else 'screen')
        path = self.logs / f'serve_{label}.log'
        self.server_log = path.open('w')
        profile_dir = str(ROOT / 'profiles' / mode) if profiling else ''
        command = ['bash', str(ROOT / 'scripts/serve.sh'), mode, profile_dir, str(budget)]
        (self.logs / f'serve_{label}.command.json').write_text(json.dumps(command) + '\n')
        self.server = subprocess.Popen(command, stdout=self.server_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.status(stage='server_starting', mode=mode, budget=budget, profiling=profiling, pid=self.server.pid)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f'{label} startup exit {self.server.returncode}: {path}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18377/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(f'{label} startup timed out')

    def run(self):
        self.run_id = 'diagnosis_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.results = ROOT / 'results' / self.run_id
        self.logs = ROOT / 'logs' / self.run_id
        self.results.mkdir(parents=True)
        self.logs.mkdir(parents=True)
        self.prerequisites()
        for mode in ['optimized', 'bf16']:
            self.start_server(mode, 0, profiling=True)
            self.run_client(mode, 64, 8000, 1, f'profile_warmup_{mode}')
            command = [self.python, str(ROOT / 'scripts/profile_load.py'), '--mode', mode,
                       '--output', str(self.results / f'profile_load_{mode}.json')]
            self.status(stage='profiling', mode=mode, window_s=3, concurrency=256, input_len=8000, output_len=1)
            with (self.logs / f'profile_load_{mode}.log').open('w') as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
            self.stop_server()
            for script, extra in [('analyse_profiles.py', []), ('summarize_profiles.py',
                    ['--output', str(ROOT / f'profile-summary-{mode}.json')])]:
                command = [self.python, str(ROOT / 'scripts' / script), '--root', str(ROOT / 'profiles'), '--modes', mode, *extra]
                self.status(stage='profile_parse', mode=mode, script=script)
                with (self.logs / f'{script}_{mode}.log').open('w') as output:
                    subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
        for budget in [4096, 8192]:
            for mode in ['bf16', 'optimized']:
                self.start_server(mode, 0, budget=budget)
                self.run_client(mode, 64, 8000, 1, f'screen_{mode}_b{budget}_warmup')
                for repeat in [1, 2]:
                    self.run_client(mode, 64, 8000, 1, f'screen_{mode}_b{budget}_r{repeat}')
                self.stop_server()
        self.status(stage='diagnosis_finished', result_dir=str(self.results))


if __name__ == '__main__':
    (ROOT / 'controller.pid').write_text(str(os.getpid()) + '\n')
    runner = Runner()
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt(f'signal {sig}')))
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage='controller_failed', error=repr(error))
        runner.issue('diagnosis-controller', repr(error), runner.logs, 'preserve evidence and stop only own service')
        (ROOT / 'background.exit').write_text('1\n')
        raise
    else:
        (ROOT / 'background.exit').write_text('0\n')
    finally:
        runner.stop_server()
