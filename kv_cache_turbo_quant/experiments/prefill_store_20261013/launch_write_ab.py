from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

root = Path(__file__).resolve().parent
runtime = root / 'write-runtime'
assert not runtime.exists(), 'new run root required'
shutil.copytree(root / 'runtime', runtime, ignore=shutil.ignore_patterns('__pycache__'))
for path in (runtime / 'scripts').glob('*'):
    if path.is_file():
        path.write_text(path.read_text().replace('diagnosis_20261013', 'write_ab_20261013'))
for name in ['run_write_ab.py', 'run_diagnosis.py']:
    text = (root / name).read_text().replace("ROOT / 'results'", "ROOT / 'results'")
    (runtime / 'scripts' / name).write_text(text)
script = runtime / 'scripts/host.py'
script.write_text(script.read_text().replace('scripts/run_diagnosis.py', 'scripts/run_write_ab.py'))
serve = runtime / 'scripts/serve.sh'
serve.write_text(serve.read_text().replace('  original)', '  candidate) integration=/ws/write_ab_20261013/candidate_write/integration ;;\n  original)'))
source = root.parent.parent / 'source/tq-prefill-20261013/kv_cache_turbo_quant/code/integration'
shutil.copytree(source, runtime / 'candidate_write/integration', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(root.parent / 'tq-opt-20261012/scripts/check_model_output.py', runtime / 'scripts/check_model_output.py')
manifest = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (runtime / 'candidate_write').rglob('*') if p.is_file()}
(runtime / 'writer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run(['python3', '-m', 'py_compile', *map(str, (runtime / 'scripts').glob('*.py'))], check=True)
subprocess.run(['bash', '-n', str(serve)], check=True)
archive = root / 'writer-ab-input.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in runtime.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            tar.add(p, arcname=str(p.relative_to(runtime)))
with archive.open('rb') as payload:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'root@112.29.145.3',
                            'cat > /data02/tq-opt-20261011/write_ab_20261013.input.tar.gz'], stdin=payload, capture_output=True)
result.check_returncode()
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'root@112.29.145.3', 'python3 -I -'],
                        input=(runtime / 'scripts/deploy_remote.py').read_bytes(), capture_output=True)
(root / 'writer-ab-deploy.log').write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
