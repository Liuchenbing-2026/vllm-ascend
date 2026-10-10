#!/usr/bin/env python3
"""Read the preserved runtime identity and current shared-host resources."""
import datetime
import json
import pathlib
import socket
import subprocess


def main():
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
    subprocess.run(['npu-smi', 'info'], check=True)
    name = 'tq-128k-ab-20261010-runtime'
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert container['Id'].startswith('0354eaaf')
    assert container['Image'] == 'sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14'
    assert any(m['Source'] == '/data1/tq-128k-ab-20261010' and m['Destination'] == '/ws'
               for m in container['Mounts'])
    print(json.dumps({'name': name, 'id': container['Id'], 'image': container['Image'],
                      'state': container['State'], 'mounts': container['Mounts'],
                      'devices': [s for s in container['Config']['Env']
                                  if s.startswith(('ASCEND_VISIBLE_DEVICES=', 'ASCEND_RT_VISIBLE_DEVICES='))]}, indent=2), flush=True)
    with socket.socket() as port:
        port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port.bind(('127.0.0.1', 18377))
    print('port 18377 bindable', flush=True)
    root = pathlib.Path('/data1/tq-128k-ab-20261010')
    print(json.dumps({'prior_status': json.loads((root/'status.json').read_text()),
                      'venv_exists': (root/'.venv/bin/python').exists(),
                      'source_exists': (root/'source/vllm').exists()}), flush=True)


if __name__ == '__main__':
    main()
