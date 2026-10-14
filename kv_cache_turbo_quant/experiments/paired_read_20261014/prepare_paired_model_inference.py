"""Release a matched model matrix only after positive reader access and latency gates."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    runtime = root / "paired-model-inference-runtime"
    assert not runtime.exists()
    shutil.copytree(root / "queued-model-runtime", runtime, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (runtime / "scripts").glob("*"):
        if path.is_file():
            path.write_text(path.read_text().replace("queued_model_20261014", "paired_model_inference_20261014"))
    source = root.parent.parent / "source/tq-paired-read-20261014/kv_cache_turbo_quant/code/integration"
    shutil.rmtree(runtime / "candidate_write")
    shutil.copytree(source, runtime / "candidate_write/integration", ignore=shutil.ignore_patterns("__pycache__"))
    (runtime / "writer-manifest.json").write_text(json.dumps({str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                            for p in (runtime / "candidate_write").rglob("*") if p.is_file()}, indent=2) + "\n")
    serve = runtime / "scripts/serve.sh"
    text = serve.read_text().replace("source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash",
                                    "source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash\nsource /ws/paired_read_20261014/opp/vendors/tqreadpair/bin/set_env.bash")
    serve.write_text(text)
    controller = runtime / "scripts/run_write_ab.py"
    text = controller.read_text()
    start = text.index("        queued_root = Path(")
    stop = text.index("        build = json.loads", start)
    text = text[:start] + """        queued_root = Path('/ws/paired_read_inference_20261014')
        assert (queued_root / 'gates.exit').read_text().strip() == '0'
        native = json.loads((queued_root / 'functional/verify-native-read.json').read_text())
        assert native['passed'] == 15 and native['async_passed'] == 20 and len(native['lookup_records']) == 2 and native['inference_passed'] == 4
        memory = json.loads((queued_root / 'memory/memory-review.json').read_text())
        assert memory['native_invocations'] == memory['invocations_without_access_error'] == 40
        import hashlib
        for artifact, expected in json.loads((queued_root / 'build-manifest.json').read_text()).items():
            assert hashlib.sha256((queued_root / artifact).read_bytes()).hexdigest() == expected['sha256'], artifact
""" + text[stop:]
    controller.write_text(text)
    host = runtime / "scripts/host.py"
    text = host.read_text().replace("previous = Path('/data02/tq-opt-20261011/queued_read_20261014')", "previous = Path('/data02/tq-opt-20261011/paired_read_inference_20261014')")
    text = text.replace("['gate-attempt1/gate.exit', 'memory-attempt1/execution.exit']", "['gates.exit']")
    needle = "    inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]"
    text = text.replace(needle, """    micro = json.loads((previous / 'micro/verify-native-read.json').read_text())
    timings = micro['timings']
    paired = []
    for tokens in [320, 120000]:
        old = next(x['device_event_mean_us'] for x in timings if x['tokens'] == tokens and x['reader'] == 'native_old')
        new = next(x['device_event_mean_us'] for x in timings if x['tokens'] == tokens and x['reader'] == 'native')
        paired.append({'tokens': tokens, 'old_us': old, 'new_us': new, 'reduction_percent': (1 - new / old) * 100})
    (ROOT / 'micro-release.json').write_text(json.dumps({'pairs': paired, 'is_model_gain': False}, indent=2) + '\\n')
    assert all(x['reduction_percent'] >= 3 for x in paired), 'no sufficient same-shape reader micro benefit; preserve candidate without model launch'
""" + needle, 1)
    host.write_text(text)
    subprocess.run(["python3", "-m", "py_compile", *map(str, (runtime / "scripts").glob("*.py"))], check=True)
    subprocess.run(["bash", "-n", str(serve)], check=True)
    archive = root / "paired-model-inference-input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in runtime.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(runtime)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/paired_model_inference_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=(runtime / "scripts/deploy_remote.py").read_bytes(), capture_output=True)
    (root / "paired-model-inference-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    print(result.stdout.decode())


if __name__ == "__main__":
    main()
