"""Deploy frozen codec gates and a matched matrix behind their completion signal."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

root = Path(__file__).resolve().parent
runtime = root / "queued-codec-model-runtime"
assert not runtime.exists()
shutil.copytree(root / "queued-model-runtime", runtime, ignore=shutil.ignore_patterns("__pycache__"))
for path in (runtime / "scripts").glob("*"):
    if path.is_file():
        path.write_text(path.read_text().replace("queued_model_20261014", "queued_codec_model_20261014"))
source = root.parent.parent / "source/tq-queued-codec-20261014/kv_cache_turbo_quant/code/integration"
shutil.rmtree(runtime / "candidate_write")
shutil.copytree(source, runtime / "candidate_write/integration", ignore=shutil.ignore_patterns("__pycache__"))
(runtime / "writer-manifest.json").write_text(json.dumps({str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (runtime / "candidate_write").rglob("*") if p.is_file()}, indent=2) + "\n")
serve = runtime / "scripts/serve.sh"
text = serve.read_text().replace("  optimized) integration=/ws/native_combined_20261013/candidate_write/integration ;;",
                                "  optimized) integration=/ws/queued_model_20261014/candidate_write/integration ;;")
serve.write_text(text)
controller = runtime / "scripts/run_write_ab.py"
text = controller.read_text()
start = text.index("        command = [self.python, str(ROOT / 'scripts/verify_pair.py')]")
stop = text.index("        gate = json.loads", start)
text = text[:start] + """        quant_root = Path('/ws/queued_codec_20261014')
        assert (quant_root / 'gates.exit').read_text().strip() == '0'
        quant = json.loads((quant_root / 'gate-attempt1/verify-queued-quant.json').read_text())
        assert quant['passed'] == 15 and quant['async_passed'] == 20
        quant_memory = json.loads((quant_root / 'memory-attempt1/memory-review.json').read_text())
        assert quant_memory['checked_invocations'] == quant_memory['invocations_without_access_error'] == 54
        assert json.loads((ROOT / 'verify-pair.json').read_text())['passed'] == 90
        for artifact, expected in json.loads((quant_root / 'build-manifest.json').read_text()).items():
            assert hashlib.sha256((quant_root / artifact).read_bytes()).hexdigest() == expected['sha256'], artifact
""" + text[stop:]
controller.write_text(text)
host = runtime / "scripts/host.py"
text = host.read_text().replace("previous = Path('/data02/tq-opt-20261011/queued_read_20261014')",
                                "previous = Path('/data02/tq-opt-20261011/queued_codec_20261014')")
text = text.replace("['gate-attempt1/gate.exit', 'memory-attempt1/execution.exit']", "['gates.exit']")
host.write_text(text)
subprocess.run(["python3", "-m", "py_compile", *map(str, (runtime / "scripts").glob("*.py"))], check=True)
subprocess.run(["bash", "-n", str(serve)], check=True)
codec = root / "queued-codec-gate-input"
assert not codec.exists()
shutil.copytree(source, codec / "code/integration", ignore=shutil.ignore_patterns("__pycache__"))
(codec / "scripts").mkdir()
shutil.copy2(root / "verify_queued_quant.py", codec / "scripts/verify_queued_quant.py")
shutil.copy2(root / "host_queued_codec.py", codec / "host.py")
(codec / "source-manifest.json").write_text(json.dumps({str(p.relative_to(codec)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in codec.rglob("*") if p.is_file()}, indent=2) + "\n")
for name, input_directory in [("queued_codec_20261014", codec), ("queued_codec_model_20261014", runtime)]:
    archive = root / (name + ".input.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in input_directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(input_directory)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/" + name + ".input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/queued_codec_20261014')
assert not root.exists()
root.mkdir()
with tarfile.open(str(root)+'.input.tar.gz') as tar:tar.extractall(root)
with (root/'host.log').open('w') as output:
 p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':p.pid,'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root)}
(root/'background-launch.json').write_text(json.dumps(record)+'\\n')
print(json.dumps(record))
"""
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "queued-codec-gate-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=(runtime / "scripts/deploy_remote.py").read_bytes(), capture_output=True)
(root / "queued-codec-model-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
