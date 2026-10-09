#!/usr/bin/env python3
"""Run on the selected host; inspect idle cards and create this task's container."""
import json
import pathlib
import subprocess
from datetime import datetime

def main():
    ROOT = pathlib.Path('/data1/tq-128k-ab-20261010')
    NAME = 'tq-128k-ab-20261010-runtime'
    IMAGE = 'vllm-ascend:dspark-a2-028'
    MODEL = '/data1/models/Qwen3-30B-A3B'
    if not ROOT.exists():
        # Docker creates only this task's bind directory; no parent-tree changes.
        subprocess.run(['docker', 'run', '--rm', '--user', '0', '-v', f'{ROOT}:/work',
                        IMAGE, 'chown', '1001:1001', '/work'], check=True)
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    report = subprocess.check_output(['npu-smi', 'info'], text=True)
    (ROOT / 'logs' / 'npu-before-container.log').write_text(report)
    for card in (0, 1):
        if f'No running processes found in NPU {card}' not in report:
            raise SystemExit(f'Card {card} is not confirmed idle; refusing allocation')
    names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
    if NAME in names:
        raise SystemExit(f'Container {NAME} already exists; inspect it before proceeding')
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE], text=True))[0]
    config = json.loads(pathlib.Path(MODEL, 'config.json').read_text())
    manifest = {
        'captured_at': datetime.now().astimezone().isoformat(),
        'host': subprocess.check_output(['hostname'], text=True).strip(),
        'image_tag': IMAGE, 'image_id': image['Id'], 'repo_digests': image['RepoDigests'],
        'cards': [0, 1], 'container': NAME, 'model_path': MODEL, 'model_config': config,
    }
    cmd = ['docker', 'run', '-d', '--name', NAME, '--label', 'task=tq-128k-ab-20261010',
           '--network', 'host', '--shm-size', '32g', '--user', '0',
           '--runtime', 'ascend', '--privileged',
           '--device', '/dev/davinci0', '--device', '/dev/davinci1',
           '--device', '/dev/davinci_manager', '--device', '/dev/devmm_svm',
           '--device', '/dev/hisi_hdc',
           '-e', 'ASCEND_RT_VISIBLE_DEVICES=0,1',
           '-e', 'ASCEND_VISIBLE_DEVICES=0,1',
           '-v', f'{ROOT}:/ws', '-v', f'{MODEL}:/models/Qwen3-30B-A3B:ro',
           '-v', '/usr/local/dcmi:/usr/local/dcmi:ro',
           '-v', '/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro',
           '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
           '-v', '/usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro',
           '-v', '/etc/ascend_install.info:/etc/ascend_install.info:ro',
           IMAGE, 'sleep', 'infinity']
    manifest['docker_run_command'] = cmd
    (ROOT / 'container_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(subprocess.check_output(cmd, text=True).strip())
    probe = ['docker', 'exec', NAME, 'bash', '-c',
             'command -v python3 cmake ninja msopgen gcc; '
             'ls -d /usr/local/Ascend/*; '
             "python3 -c 'import torch,torch_npu,importlib.metadata as m; "
             "print(torch.__version__,torch_npu.__version__); "
             "print({k:m.version(k) for k in [\"vllm\",\"vllm-ascend\",\"transformers\"]})'; "
             'ls -d /vllm-workspace/* /usr/local/python* 2>/dev/null']
    result = subprocess.run(probe, capture_output=True, text=True)
    (ROOT / 'logs' / 'image-probe.log').write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr)
    print('Created', NAME, 'using physical cards 0,1')


if __name__ == '__main__':
    main()
