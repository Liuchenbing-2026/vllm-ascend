#!/usr/bin/env python3
"""Two paired modes, one full-shape warmup and three formal C256 cohorts with service max_num_seqs512."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tarfile
import time
import urllib.request

from run_matrix import MatrixRunner

ROOT = Path('/ws/byte_model_host3_attempt4_20261015')


class Runner(MatrixRunner):
    def __init__(self):
        run_id = '8k1_c256s512_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        super().__init__(str(ROOT), run_id, ['bf16', 'optimized'], [256], 3600, None)
        self.python = '/ws/.venv/bin/python'

    def status(self, **fields):
        fields.setdefault('run_id', self.run_id)
        record = {'updated_utc': datetime.now(timezone.utc).isoformat(), **fields}
        pending = ROOT / 'status.pending'
        pending.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(pending, ROOT / 'status.json')
        with (ROOT / 'status-history.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')
        print(json.dumps(fields), flush=True)

    def model_metadata_check(self):
        prior = json.loads(Path('/ws/c128_20261012/model-check.json').read_text())
        model = Path('/models/Qwen3-30B-A3B')
        checks = []
        for old in prior['config_full_hash_now']:
            path = model / old['file']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == old['sha256_now'], path
            checks.append({'file': old['file'], 'bytes': path.stat().st_size, 'sha256_now': digest})
        shards = []
        for old in prior['shards']:
            stat = (model / old['file']).stat()
            matches = stat.st_size == old['bytes'] and stat.st_mtime_ns == old['mtime_ns']
            assert matches, old['file']
            shards.append({**old, 'metadata_matches_prior': matches})
        result = {**prior, 'checked_utc': datetime.now(timezone.utc).isoformat(),
                  'model': str(model), 'config_full_hash_now': checks, 'shards': shards,
                  'new_full_shard_hash_performed': False}
        (ROOT / 'model-check.json').write_text(json.dumps(result, indent=2) + '\n')

    def start_server(self, mode, attempt):
        path = self.logs / f'serve_{mode}.log'
        self.server_log = path.open('w')
        command = ['bash', str(ROOT / 'scripts/serve.sh'), mode]
        (self.logs / f'serve_{mode}.command.json').write_text(json.dumps(command) + '\n')
        self.server = subprocess.Popen(command, stdout=self.server_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.status(stage='server_starting', run_id=self.run_id, mode=mode,
                    pid=self.server.pid, log=str(path))
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f'{mode} startup exit {self.server.returncode}: {path}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18377/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(f'{mode} startup timed out')

    def prerequisites(self):
        for name in ['bootstrap.exit', 'tq-op-build.exit', 'model-hash.exit']:
            assert (Path('/ws') / name).read_text().strip() == '0', name
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', 18377))
        manifest = json.loads((ROOT / 'source-manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
            assert hashlib.sha256((Path('/ws/c128_20261012') / name).read_bytes()).hexdigest() == expected, ('runtime', name)
        # Reuse prior correctness only after verifying the unchanged implementation.
        prior = Path('/ws/opt_20261012')
        for name in manifest['files']:
            assert (ROOT / name).read_bytes() == (prior / name).read_bytes(), name
        revisions = {}
        archive_hashes = json.loads(Path('/ws/artifacts/source-archives.json').read_text())
        for name, expected, archive_name in [('vllm', '2cf0a6915ce544dc493a0990f2ea38d81601128a', 'vllm-source.tar.gz'),
                ('vllm-ascend', '96df623103f921df1a4488170d106f36689acb59', 'ascend-source.tar.gz')]:
            archive_path = Path('/ws') / archive_name
            archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            assert archive_sha == archive_hashes[archive_name], archive_name
            matched, changed = 0, []
            with tarfile.open(archive_path, 'r:gz') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    current = Path('/ws/source') / name / member.name
                    payload = archive.extractfile(member).read()
                    if current.is_file() and current.read_bytes() == payload:
                        matched += 1
                    else:
                        changed.append(member.name)
            revisions[name] = {'commit_provenance': expected,
                'verification_method': 'original fixed export archive SHA256 plus file-by-file comparison; no .git in runtime',
                'archive_sha256': archive_sha, 'matched_source_files': matched, 'changed_source_files': changed}
            assert not changed, (name, changed)
        (ROOT / 'framework-revisions.json').write_text(json.dumps(revisions, indent=2) + '\n')
        result = json.loads((prior / 'verify/verify_result.json').read_text())
        assert result['passed_cases'] == result['total_cases'] == 7
        integration = json.loads((prior / 'verify/integration_bits4.json').read_text())
        assert integration['passed'] == 15
        (self.logs / 'precision_reused.json').write_text(json.dumps({
            'prior_root': str(prior), 'code_hashes_verified': True,
            'kernel_cases': 7, 'real_integration_cases': 15,
            'limitation': 'prior unchanged implementation verification; new8K single-request usage smoke, not semantic quality benchmark'}, indent=2) + '\n')
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=os.environ['TASK_NPU_CARDS'],
                   VLLM_ASCEND_KVTQ_BITS='4', TORCH_EXTENSIONS_DIR='/ws/torch-extensions',
                   TRITON_CACHE_DIR='/ws/c128_20261012/model-triton-cache',
                   PYTHONPATH='/ws/c128_20261012/candidate/integration' + ':/ws/source/vllm:/ws/source/vllm-ascend')
        self.environment = env
        env['VLLM_PLUGINS'] = 'ascend,ascend_kv_connector,ascend_model,ascend_model_loader,kv_cache_turbo_quant_shadow'
        self.model_metadata_check()
        for name, script in [('environment_before', 'collect_environment.py')]:
            self.status(stage=name, run_id=self.run_id)
            with (self.logs / f'{name}.log').open('w') as output:
                subprocess.run([self.python, str(ROOT / 'scripts' / script)], env=env,
                               stdout=output, stderr=subprocess.STDOUT, check=True, timeout=600)
        (ROOT / 'environment_before.json').write_bytes((ROOT / 'environment.json').read_bytes())
        current = json.loads((ROOT / 'environment_before.json').read_text())
        previous = json.loads(Path('/ws/8k1_20261013/environment_after.json').read_text())
        assert current['packages'] == previous['packages'], 'dependencies drifted since prior gate'
        stable = lambda d: {k: v for k, v in d.items() if '/torch-extensions/' not in k}
        assert stable(current['native_artifacts']) == stable(previous['native_artifacts']), 'native kernel artifacts drifted'
        # C++ source and ACLNN/kernel ABI are unchanged; the glue may have been
        # rebuilt for an equivalent source filename in the prior candidate.
        cpp = Path('/ws/c128_20261012/candidate/integration/torch_ext/turboquant_torch.cpp')
        assert hashlib.sha256(cpp.read_bytes()).hexdigest() == '51fdaeb74f26aa2c9aece180955651dab6e858de943cee9ab105a8022066d6b5'


    def run_client(self, mode, concurrency, input_len, output_len, name, timeout=None):
        # Same original client and exact payloads; retain the commands locally.
        timeout = self.request_timeout if timeout is None else timeout
        output = self.results / f'{name}.json'
        command = [self.python, str(ROOT / 'scripts/bench_client.py'), '--mode', mode,
                   '--concurrency', str(concurrency), '--input-len', str(input_len),
                   '--output-len', str(output_len), '--timeout', str(timeout), '--output', str(output)]
        (self.logs / f'{name}.command.json').write_text(json.dumps(command, indent=2) + '\n')
        self.status(stage='benchmark_running', run_id=self.run_id, mode=mode,
                    concurrency=concurrency, input_len=input_len, output_len=output_len,
                    result=str(output), started_utc=datetime.now(timezone.utc).isoformat())
        with (self.logs / f'{name}.client.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        if code:
            self.issue(name, f'client exit {code}; work did not all complete', output,
                       'retain failed and timeout requests; stop only this run')
            raise RuntimeError(f'{name} failed: {code}')
        return code

    def run(self):
        self.results.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        parameters = {'run_id': self.run_id, 'modes': self.modes, 'input_len': 8000,
            'output_len': 1, 'concurrency': 256, 'max_num_seqs': 512, 'max_model_len': 8192, 'max_num_batched_tokens': 4096,
            'warmup_cohorts_per_mode': 1, 'formal_repetitions': 3,
            'physical_cards': [2, 5], 'tp': 2, 'kv_cache_bytes_per_card': 25769803776,
            'profiler_active_during_formal': False, 'request_timeout_s': self.request_timeout,
            'mode_order': self.modes,
            'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / 'scripts').glob('*') if p.is_file()},
            'sources': json.loads((ROOT / 'source-manifest.json').read_text())}
        (self.logs / 'run_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
        self.prerequisites()
        for mode in self.modes:
            self.start_server(mode, 0)
            self.run_client(mode, 1, 64, 16, f'smoke_{mode}', timeout=300)
            self.run_client(mode, 1, 8000, 1, f'smoke_8k_{mode}', timeout=300)
            prefix = f'{mode}_in8000_out1_c256'
            self.run_client(mode, 256, 8000, 1, prefix + '_warmup')
            for repeat in range(1, 4):
                self.run_client(mode, 256, 8000, 1, prefix + f'_r{repeat}')
            self.stop_server()
        with (self.logs / 'environment_after.log').open('w') as output:
            subprocess.run([self.python, str(ROOT / 'scripts/collect_environment.py')],
                           env=self.environment, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=600)
        (ROOT / 'environment_after.json').write_bytes((ROOT / 'environment.json').read_bytes())
        self.status(stage='8k1_finished', run_id=self.run_id, result_dir=str(self.results))


def main():
    (ROOT / 'controller.pid').write_text(str(os.getpid()) + '\n')
    runner = Runner()
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt(f'signal {sig}')))
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage='controller_failed', run_id=runner.run_id, error=repr(error))
        runner.issue('8k1-controller', repr(error), runner.logs,
                     'retain all original evidence and stop only the dedicated task server')
        (ROOT / 'background.exit').write_text('1\n')
        raise
    else:
        (ROOT / 'background.exit').write_text('0\n')
    finally:
        runner.stop_server()


if __name__ == '__main__':
    main()
