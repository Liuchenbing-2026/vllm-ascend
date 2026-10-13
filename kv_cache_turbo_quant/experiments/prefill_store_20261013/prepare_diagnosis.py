from pathlib import Path
import json
import shutil

root = Path(__file__).resolve().parent
prior = root.parent / 'tq-8k1-c256-s512-20261013/final-snapshot'
runtime = root / 'runtime'
(runtime / 'scripts').mkdir(parents=True, exist_ok=True)
(runtime / 'logs').mkdir(exist_ok=True)
for name in ['run_8k1.py', 'run_matrix.py', 'bench_client.py', 'collect_environment.py', 'host.py', 'monitor.py', 'serve.sh', 'deploy_remote.py']:
    text = (prior / 'scripts' / name).read_text().replace('8k1_c256_s512_20261013', 'diagnosis_20261013')
    if name == 'host.py':
        text = text.replace('scripts/run_8k1.py', 'scripts/run_diagnosis.py')
    if name == 'serve.sh':
        text = text.replace('--max-num-batched-tokens 4096', '--max-num-batched-tokens "${3:-4096}"')
    if name == 'deploy_remote.py':
        text = text.replace('root.mkdir()', "root.mkdir(exist_ok=True)\nassert not (root / 'background-launch.json').exists(), 'already launched; preserve this run'")
        text = text.replace('tar.extractall(root)', "tar.extractall(root)\n(root / 'logs').mkdir(exist_ok=True)")
    (runtime / 'scripts' / name).write_text(text)
shutil.copytree(prior / 'candidate', runtime / 'candidate', dirs_exist_ok=True)
shutil.copy2(prior / 'source-manifest.json', runtime / 'source-manifest.json')
for name in ['analyse_profiles.py', 'summarize_profiles.py']:
    shutil.copy2(root.parent / 'tq-opt-20261012/scripts' / name, runtime / 'scripts' / name)
print(runtime)
