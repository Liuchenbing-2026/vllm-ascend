from pathlib import Path
import json
import subprocess
import tarfile
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
runtime = root / 'runtime'
profile = (root.parent / 'tq-opt-20261012/scripts/profile_load.py').read_text()
profile = profile.replace('range(60)', 'range(256)').replace('k=2000', 'k=8000')
profile = profile.replace("'max_tokens':32", "'max_tokens':1").replace('limit=70', 'limit=266')
profile = profile.replace("len(first_seen)==60", "len(first_seen)==256")
profile = profile.replace("{'prompt_tokens':2000,'completion_tokens':32,'total_tokens':2032}", "{'prompt_tokens':8000,'completion_tokens':1,'total_tokens':8001}")
profile = profile.replace("            await asyncio.wait_for(all_started.wait(),timeout=900)\n            await api('start_profile','decode');profiling=True\n            await asyncio.sleep(3)\n            await api('stop_profile','decode');profiling=False\n", '')
profile = profile.replace("'input_len':2000", "'input_len':8000").replace("'output_len':32", "'output_len':1").replace("'concurrency':60", "'concurrency':256")
profile = profile.replace('The 32-token profiling load is separate from the 200-token formal benchmark.', 'The 8000/1 C256 load is diagnostic only; output1 has no decode interval.')
(runtime / 'scripts/profile_load.py').write_text(profile)
(runtime / 'scripts/run_diagnosis.py').write_bytes((root / 'run_diagnosis.py').read_bytes())
scripts = list((runtime / 'scripts').glob('*.py'))
subprocess.run(['python3', '-m', 'py_compile', *map(str, scripts)], check=True)
subprocess.run(['bash', '-n', str(runtime / 'scripts/serve.sh')], check=True)
archive = root / 'input.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for file in runtime.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts:
            tar.add(file, arcname=str(file.relative_to(runtime)))
host = 'root@112.29.145.3'
target = '/data02/tq-opt-20261011/diagnosis_20261013.input.tar.gz'
with archive.open('rb') as payload:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'cat > ' + target], stdin=payload, capture_output=True)
(root / 'upload.log').write_bytes(result.stdout + result.stderr)
result.check_returncode()
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'python3 -I -'],
                        input=(runtime / 'scripts/deploy_remote.py').read_bytes(), capture_output=True)
(root / 'deploy.log').write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
(root / 'launched.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'remote_stdout': result.stdout.decode()}, indent=2) + '\n')
