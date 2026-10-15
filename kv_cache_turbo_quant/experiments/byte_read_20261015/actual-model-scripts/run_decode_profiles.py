"""Matched profiling after the completed native-reader throughput matrix."""

import hashlib
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_8k1 import ROOT
from run_diagnosis import Runner as ProfileRunner


class Runner(ProfileRunner):
    def prerequisites(self):
        import tarfile
        proof = json.loads((ROOT/'model-recheck.json').read_text())
        assert proof['passed'] and len(proof['shards']) == 16
        expected = json.loads(Path('/ws/artifacts/source-archives.json').read_text())
        records = {}
        for component, archive_name in [('vllm', 'vllm-source.tar.gz'), ('vllm-ascend', 'ascend-source.tar.gz')]:
            p = Path('/ws') / archive_name
            assert hashlib.sha256(p.read_bytes()).hexdigest() == expected[archive_name]
            count = 0
            with tarfile.open(p) as tar:
                for m in tar:
                    if m.isfile():
                        assert (Path('/ws/source') / component / m.name).read_bytes() == tar.extractfile(m).read(), m.name
                        count += 1
            records[component] = {'matched_files': count, 'archive_sha256': expected[archive_name]}
        (ROOT / 'framework-revisions.json').write_text(json.dumps(records, indent=2) + '\n')
        env = dict(os.environ, PYTHONPATH='/ws/source/vllm:/ws/source/vllm-ascend', TASK_NPU_CARDS=os.environ['TASK_NPU_CARDS'])
        subprocess.run([self.python, str(ROOT / 'scripts/collect_environment.py')], env=env, check=True)
        actual = json.loads((ROOT / 'environment.json').read_text())
        prior = json.loads(Path('/ws/paired_model_inference_20261014/environment_before.json').read_text())
        assert actual['packages'] == prior['packages'], 'package drift must be reviewed before profiling'
        assert (ROOT / 'migration-golden.exit').read_text().strip() == '0'
        gate = json.loads((ROOT / 'migration-native-artifacts.json').read_text())
        for path, expected in gate.items():
            assert actual['native_artifacts'][path]['sha256'] == expected['sha256'], path
        self.environment = env
        with __import__('socket').socket() as s:
            s.bind(('127.0.0.1', 18377))

    def run(self):
        self.run_id = "decode_profiles_" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.results = ROOT / "results" / self.run_id
        self.logs = ROOT / "logs" / self.run_id
        self.results.mkdir(parents=True)
        self.logs.mkdir(parents=True)
        self.prerequisites()
        previous = Path("/ws/paired_model_inference_20261014")
        assert (previous / "background.exit").read_text().strip() == "0"
        assert json.loads((previous / "verify-pair.json").read_text())["passed"] == 90
        assert (previous / 'host.exit').read_text().strip() == '0'
        expected = json.loads((ROOT / 'writer-manifest.json').read_text())
        for name, digest in expected.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            assert (ROOT / name).read_bytes() == (previous / name).read_bytes(), name
        gates = Path('/ws/paired_read_inference_20261014')
        assert (gates / 'gates.exit').read_text().strip() == '0'
        for artifact, expected in json.loads((gates / 'build-manifest.json').read_text()).items():
            assert hashlib.sha256((gates / artifact).read_bytes()).hexdigest() == expected['sha256'], artifact
        parameters = {
            "modes": ["candidate", "bf16"],
            "tp": 2,
            "physical_cards": [3, 6],
            "max_num_seqs": 512,
            "budget": 4096,
            "kv_bytes_per_card": 25769803776,
            "nominal_window_s": 3,
            "decode_gate": "all requests first text received, none ended",
            "phase_order": [
                [20, 20, 16, "prefill"],
                [20, 20, 16, "decode"],
                [2000, 200, 60, "prefill"],
                [2000, 200, 60, "decode"],
            ],
            "excluded_from_formal_throughput": True,
        }
        (self.logs / "parameters.json").write_text(
            json.dumps(parameters, indent=2) + "\n"
        )
        for mode in ["candidate", "bf16"]:
            self.start_server(mode, 0, profiling=True)
            for input_len, output_len, concurrency, phases in [
                (20, 20, 16, ["prefill", "decode"]),
                (2000, 200, 60, ["prefill", "decode"]),
            ]:
                self.run_client(
                    mode,
                    concurrency,
                    input_len,
                    output_len,
                    f"warmup_{mode}_{input_len}_{output_len}_c{concurrency}",
                )
                for phase in phases:
                    tag = f"{mode}_{input_len}_{output_len}_c{concurrency}_{phase}"
                    command = [
                        self.python,
                        str(ROOT / "scripts/profile_decode.py"),
                        "--mode",
                        mode,
                        "--phase",
                        phase,
                        "--input-len",
                        str(input_len),
                        "--output-len",
                        str(output_len),
                        "--concurrency",
                        str(concurrency),
                        "--output",
                        str(self.results / (tag + ".json")),
                    ]
                    (self.logs / (tag + ".command.json")).write_text(
                        json.dumps(command, indent=2) + "\n"
                    )
                    self.status(
                        stage="profiling",
                        mode=mode,
                        phase=phase,
                        input_len=input_len,
                        output_len=output_len,
                        concurrency=concurrency,
                        window_s=3,
                    )
                    with (self.logs / (tag + ".log")).open("w") as output:
                        subprocess.run(
                            command,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=1800,
                        )
            self.stop_server()
            for script, extra in [
                ("analyse_profiles.py", []),
                (
                    "summarize_profiles.py",
                    ["--output", str(ROOT / ("profile-summary-" + mode + ".json"))],
                ),
            ]:
                command = [
                    self.python,
                    str(ROOT / "scripts" / script),
                    "--root",
                    str(ROOT / "profiles"),
                    "--modes",
                    mode,
                    *extra,
                ]
                self.status(stage="profile_parse", mode=mode, script=script)
                with (self.logs / (script + "_" + mode + ".log")).open("w") as output:
                    subprocess.run(
                        command,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=2400,
                    )
        self.status(stage="decode_profiles_finished", result_dir=str(self.results))


if __name__ == "__main__":
    (ROOT / "controller.pid").write_text(str(os.getpid()) + "\n")
    runner = Runner()
    signal.signal(
        signal.SIGTERM,
        lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {sig}")),
    )
    try:
        runner.run()
    except BaseException as error:
        runner.status(stage="controller_failed", error=repr(error))
        runner.issue(
            "decode-profiling",
            repr(error),
            runner.logs,
            "preserve raw evidence and stop only own service",
        )
        (ROOT / "background.exit").write_text("1\n")
        raise
    else:
        (ROOT / "background.exit").write_text("0\n")
    finally:
        runner.stop_server()
