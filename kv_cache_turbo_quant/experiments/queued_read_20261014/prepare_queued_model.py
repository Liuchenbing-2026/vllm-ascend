"""Durable matched model test, released by codec, ordering and memory gates."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

root = Path(__file__).resolve().parent
runtime = root / "queued-model-runtime"
assert not runtime.exists()
shutil.copytree(root.parent / "tq-opt-20261013/native-combined-runtime", runtime,
                ignore=shutil.ignore_patterns("__pycache__"))
name = "queued_model_20261014"
for path in (runtime / "scripts").glob("*"):
    if path.is_file():
        path.write_text(path.read_text().replace("native_combined_20261013", name))
source = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant/code/integration"
shutil.rmtree(runtime / "candidate_write")
shutil.copytree(source, runtime / "candidate_write/integration", ignore=shutil.ignore_patterns("__pycache__"))
(runtime / "writer-manifest.json").write_text(json.dumps({str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (runtime / "candidate_write").rglob("*") if p.is_file()}, indent=2) + "\n")
serve = runtime / "scripts/serve.sh"
serve.write_text(serve.read_text().replace("  optimized) integration=/ws/c128_20261012/candidate/integration ;;",
                                         "  optimized) integration=/ws/native_combined_20261013/candidate_write/integration ;;"))
controller = runtime / "scripts/run_write_ab.py"
text = controller.read_text()
start = text.index("        native = json.loads")
stop = text.index("        build = json.loads", start)
text = text[:start] + """        queued_root = Path('/ws/queued_read_20261014')
        native = json.loads((queued_root / 'gate-attempt1/verify-native-read.json').read_text())
        assert native['passed'] == 15 and native['async_passed'] == 20
        memory = json.loads((queued_root / 'memory-attempt1/memory-review.json').read_text())
        assert memory['native_invocations'] == memory['invocations_without_access_error'] == 38
""" + text[stop:]
controller.write_text(text)
host = runtime / "scripts/host.py"
text = host.read_text()
start = text.index("    previous = Path('/data02/tq-opt-20261011/native_read_20261013')")
stop = text.index("    inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]", start)
text = text[:start] + """    previous = Path('/data02/tq-opt-20261011/queued_read_20261014')
    for required in ['gate-attempt1/gate.exit', 'memory-attempt1/execution.exit']:
        while not (previous / required).exists():
            (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'waiting_for_queued_read_gate',
                'required': required, 'utc': datetime.now(timezone.utc).isoformat()}) + '\\n')
            time.sleep(20)
        assert (previous / required).read_text().strip() == '0', required
""" + text[stop:]
host.write_text(text)
subprocess.run(["python3", "-m", "py_compile", *map(str, (runtime / "scripts").glob("*.py"))], check=True)
subprocess.run(["bash", "-n", str(serve)], check=True)
archive = root / "queued-model-input.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for path in runtime.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            tar.add(path, arcname=str(path.relative_to(runtime)))
with archive.open("rb") as payload:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", f"cat > /data02/tq-opt-20261011/{name}.input.tar.gz"], stdin=payload, capture_output=True)
result.check_returncode()
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=(runtime / "scripts/deploy_remote.py").read_bytes(), capture_output=True)
(root / "queued-model-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode();print(result.stdout.decode())
