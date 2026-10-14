"""Matched old-TQ/new-TQ/BF16 formal cohorts; profiles remain separate."""
import json
import os
from pathlib import Path
import signal
import subprocess
from run_diagnosis import Runner as ProfileRunner
from run_8k1 import ROOT


class Runner(ProfileRunner):
    def run(self):
        from datetime import datetime, timezone

        self.run_id = 'write_ab_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.results = ROOT / 'results' / self.run_id
        self.logs = ROOT / 'logs' / self.run_id
        self.results.mkdir(parents=True)
        self.logs.mkdir(parents=True)
        self.prerequisites()
        native_root = Path('/ws/native_read_20261013')
        queued_root = Path('/ws/paired_read_20261014')
        assert (queued_root / 'gates-attempt2.exit').read_text().strip() == '0'
        native = json.loads((queued_root / 'attempt2/functional/verify-native-read.json').read_text())
        assert native['passed'] == 15 and native['async_passed'] == 20 and len(native['lookup_records']) == 2
        memory = json.loads((queued_root / 'attempt2/memory/memory-review.json').read_text())
        assert memory['native_invocations'] == memory['invocations_without_access_error'] == 36
        import hashlib
        for artifact, expected in json.loads((queued_root / 'build-manifest.json').read_text()).items():
            assert hashlib.sha256((queued_root / artifact).read_bytes()).hexdigest() == expected['sha256'], artifact
        build = json.loads((native_root / 'build-manifest.json').read_text())
        import hashlib
        for artifact, expected in build['files'].items():
            assert hashlib.sha256((native_root / artifact).read_bytes()).hexdigest() == expected['sha256'], artifact
        (ROOT / 'native-build-verified.json').write_text(json.dumps(build, indent=2) + '\n')
        (ROOT / 'native-memory-review.json').write_text(json.dumps(memory, indent=2) + '\n')
        command = [self.python, str(ROOT / 'scripts/verify_pair.py')]
        environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='2,5', VLLM_ASCEND_KVTQ_BITS='4',
            PYTHONPATH=str(ROOT / 'candidate_write/integration') + ':/ws/source/vllm:/ws/source/vllm-ascend')
        self.status(stage='paired_precision')
        with (self.logs / 'verify_pair.log').open('w') as output:
            subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
        assert json.loads((ROOT / 'verify-pair.json').read_text())['passed'] == 90
        gate = json.loads(Path('/ws/diagnosis_20261013/verify-write.json').read_text())
        assert gate['passed'] == 78
        expected = json.loads((ROOT / 'writer-manifest.json').read_text())
        import hashlib

        for name, digest in expected.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        scenarios = [('20_20_c16', 20, 20, 16), ('2k200_c60', 2000, 200, 60), ('8k1_c256', 8000, 1, 256)]
        (self.logs / 'run_parameters.json').write_text(json.dumps({
            'run_id': self.run_id, 'modes': ['candidate', 'bf16', 'optimized'],
            'scenarios': scenarios, 'max_num_seqs': 512, 'max_num_batched_tokens': 4096,
            'max_model_len': 8192, 'tp': 2, 'physical_cards': [2, 5],
            'kv_cache_bytes_per_card': 25769803776, 'warmups': 1, 'formal_repeats': 3,
            'profiler_active_during_formal': False, 'writer_manifest': expected}, indent=2) + '\n')
        for mode in ['candidate', 'bf16', 'optimized']:
            self.start_server(mode, 0)
            self.run_client(mode, 1, 8000, 1, f'smoke_8k_{mode}', timeout=300)
            command = [self.python, str(ROOT / 'scripts/check_model_output.py'), '--mode', mode,
                       '--output', str(self.results / f'precision_model_{mode}.json')]
            with (self.logs / f'precision_model_{mode}.log').open('w') as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
            for scenario, input_len, output_len, concurrency in scenarios:
                prefix = f'{scenario}_{mode}'
                self.run_client(mode, concurrency, input_len, output_len, prefix + '_warmup')
                for repeat in [1, 2, 3]:
                    self.run_client(mode, concurrency, input_len, output_len, prefix + f'_r{repeat}')
            self.stop_server()
        previous = json.loads((self.results / 'precision_model_optimized.json').read_text())
        current = json.loads((self.results / 'precision_model_candidate.json').read_text())
        assert len(previous['records']) == len(current['records']) == 4
        for a, b in zip(previous['records'], current['records']):
            assert a['payload_sha256'] == b['payload_sha256']
            assert a['response']['choices'][0]['logprobs']['tokens'] == b['response']['choices'][0]['logprobs']['tokens']
        (ROOT / 'model-token-equality.json').write_text(json.dumps({'passed_samples': 4, 'equal_token_ids': 440,
            'scope': 'old TQ versus fused-writer TQ; not TQ versus BF16 semantic quality'}) + '\n')
        self.status(stage='write_ab_finished', result_dir=str(self.results))


if __name__ == '__main__':
    (ROOT / 'controller.pid').write_text(str(os.getpid()) + '\n')
    runner = Runner()
    signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt(f'signal {sig}')))
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage='controller_failed', error=repr(error))
        runner.issue('writer-ab-controller', repr(error), runner.logs, 'retain all raw evidence and stop only own service')
        (ROOT / 'background.exit').write_text('1\n')
        raise
    else:
        (ROOT / 'background.exit').write_text('0\n')
    finally:
        runner.stop_server()
