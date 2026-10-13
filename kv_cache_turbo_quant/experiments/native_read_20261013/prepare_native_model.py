"""Concrete next model candidate, released only by strict native-reader gates."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

root = Path(__file__).resolve().parent
runtime = root / 'native-model-runtime'
assert not runtime.exists()
shutil.copytree(root / 'paired-attempt3-runtime', runtime, ignore=shutil.ignore_patterns('__pycache__'))
for path in (runtime / 'scripts').glob('*'):
    if path.is_file():
        path.write_text(path.read_text().replace('paired_write_20261013_attempt3', 'native_model_20261013'))
source = root.parent.parent / 'source/tq-native-read-20261013/kv_cache_turbo_quant/code/integration'
shutil.rmtree(runtime / 'candidate_write')
shutil.copytree(source, runtime / 'candidate_write/integration', ignore=shutil.ignore_patterns('__pycache__'))
manifest = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (runtime / 'candidate_write').rglob('*') if p.is_file()}
(runtime / 'writer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
serve = runtime / 'scripts/serve.sh'
text = serve.read_text().replace('mode=$1', 'mode=$1\nsource /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash')
serve.write_text(text)
controller = runtime / 'scripts/run_write_ab.py'
text = controller.read_text()
start = text.index("        command = [self.python, str(ROOT / 'scripts/verify_pair.py')]")
stop = text.index("        gate = json.loads(Path('/ws/diagnosis_20261013/verify-write.json').read_text())", start)
text = text[:start] + "        native = json.loads(Path('/ws/native_read_20261013/verify-native-read.json').read_text())\n        assert native['passed'] >= 9\n" + text[stop:]
controller.write_text(text)
host = runtime / 'scripts/host.py'
text = host.read_text()
start = text.index("    previous = Path('/data02/tq-opt-20261011/write_ab_20261013')")
stop = text.index("    inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]", start)
text = text[:start] + """    previous = Path('/data02/tq-opt-20261011/native_read_20261013')
    while not (previous / 'gate.exit').exists():
        (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'waiting_for_native_read_gate',
            'utc': datetime.now(timezone.utc).isoformat()}) + '\\n')
        time.sleep(20)
    assert (previous / 'gate.exit').read_text().strip() == '0', 'do not benchmark a reader that failed bit equality'
    report = json.loads((previous / 'verify-native-read.json').read_text())
    times = [r for r in report['timings'] if r.get('tokens') == 120000]
    old = next(r['device_event_mean_us'] for r in times if r['reader'] == 'triton')
    new = next(r['device_event_mean_us'] for r in times if r['reader'] == 'native')
    assert new < old, 'native reader did not improve target microcase; optimize before full model'
""" + text[stop:]
host.write_text(text)
subprocess.run(['python3', '-m', 'py_compile', *map(str, (runtime / 'scripts').glob('*.py'))], check=True)
subprocess.run(['bash', '-n', str(serve)], check=True)
archive = root / 'native-model-input.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for path in runtime.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            tar.add(path, arcname=str(path.relative_to(runtime)))
with archive.open('rb') as payload:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'root@112.29.145.3',
        'cat > /data02/tq-opt-20261011/native_model_20261013.input.tar.gz'], stdin=payload, capture_output=True)
result.check_returncode()
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'root@112.29.145.3', 'python3 -I -'],
    input=(runtime / 'scripts/deploy_remote.py').read_bytes(), capture_output=True)
(root / 'native-model-deploy.log').write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
