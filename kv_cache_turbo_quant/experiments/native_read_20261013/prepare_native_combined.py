"""Queue the verified reader plus paired writer with fresh matched cohorts."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

root = Path(__file__).resolve().parent
runtime = root / "native-combined-runtime"
assert not runtime.exists()
shutil.copytree(root / "paired-attempt3-runtime", runtime, ignore=shutil.ignore_patterns("__pycache__"))
name = "native_combined_20261013"
for path in (runtime / "scripts").glob("*"):
    if path.is_file():
        path.write_text(path.read_text().replace("paired_write_20261013_attempt3", name))
source = root.parent.parent / "source/tq-native-read-20261013/kv_cache_turbo_quant/code/integration"
shutil.rmtree(runtime / "candidate_write")
shutil.copytree(source, runtime / "candidate_write/integration", ignore=shutil.ignore_patterns("__pycache__"))
manifest = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (runtime / "candidate_write").rglob("*") if p.is_file()}
(runtime / "writer-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
serve = runtime / "scripts/serve.sh"
serve.write_text(serve.read_text().replace("mode=$1", "mode=$1\nsource /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash"))
controller = runtime / "scripts/run_write_ab.py"
text = controller.read_text()
anchor = "        self.prerequisites()\n"
extra = """        native_root = Path('/ws/native_read_20261013')
        native = json.loads((native_root / 'gate-attempt3/verify-native-read.json').read_text())
        assert native['passed'] == 15
        memory = json.loads((native_root / 'memory-attempt2/memory-review.json').read_text())
        assert memory['native_invocations'] == memory['invocations_without_error'] == 14
        build = json.loads((native_root / 'build-manifest.json').read_text())
        import hashlib
        for artifact, expected in build['files'].items():
            assert hashlib.sha256((native_root / artifact).read_bytes()).hexdigest() == expected['sha256'], artifact
        (ROOT / 'native-build-verified.json').write_text(json.dumps(build, indent=2) + '\\n')
        (ROOT / 'native-memory-review.json').write_text(json.dumps(memory, indent=2) + '\\n')
"""
text = text.replace(anchor, anchor + extra, 1)
text = text.replace("scenarios = [('8k1_c256', 8000, 1, 256), ('20_20_c16', 20, 20, 16), ('2k200_c60', 2000, 200, 60)]", "scenarios = [('20_20_c16', 20, 20, 16), ('2k200_c60', 2000, 200, 60), ('8k1_c256', 8000, 1, 256)]")
text = text.replace("['optimized', 'candidate', 'bf16']", "['candidate', 'bf16', 'optimized']")
text = text.replace("paired K/V quantizer/writer", "paired K/V writer and native reader")
controller.write_text(text)
host = runtime / "scripts/host.py"
text = host.read_text()
start = text.index("    previous = Path('/data02/tq-opt-20261011/write_ab_20261013')")
stop = text.index("    inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]", start)
text = text[:start] + """    previous = Path('/data02/tq-opt-20261011/native_read_20261013')
    for required in ['gate-attempt3/gate.exit', 'memory-attempt2/execution.exit', 'leak-reference-attempt1/execution.exit']:
        while not (previous / required).exists():
            (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'waiting_for_verified_reader',
                'required': required, 'utc': datetime.now(timezone.utc).isoformat()}) + '\\n')
            time.sleep(20)
        assert (previous / required).read_text().strip() == '0', required
    report = json.loads((previous / 'gate-attempt3/verify-native-read.json').read_text())
    assert report['passed'] == 15
    times = [r for r in report['timings'] if r.get('tokens') == 120000]
    old = next(r['device_event_mean_us'] for r in times if r['reader'] == 'triton')
    new = next(r['device_event_mean_us'] for r in times if r['reader'] == 'native')
    assert new < old, 'reader did not improve target microcase'
""" + text[stop:]
text = text.replace("if __name__ == '__main__':\n    main()", """if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (ROOT / 'host.exit').write_text('1\\n')
        (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'host_failed',
            'error': repr(error), 'utc': datetime.now(timezone.utc).isoformat()}) + '\\n')
        raise
""")
host.write_text(text)
subprocess.run(["python3", "-m", "py_compile", *map(str, (runtime / "scripts").glob("*.py"))], check=True)
subprocess.run(["bash", "-n", str(serve)], check=True)
archive = root / "native-combined-input.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for path in runtime.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            tar.add(path, arcname=str(path.relative_to(runtime)))
with archive.open("rb") as payload:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", f"cat > /data02/tq-opt-20261011/{name}.input.tar.gz"], stdin=payload, capture_output=True)
result.check_returncode()
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=(runtime / "scripts/deploy_remote.py").read_bytes(), capture_output=True)
(root / "native-combined-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
