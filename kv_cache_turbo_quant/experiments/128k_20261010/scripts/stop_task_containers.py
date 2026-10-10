#!/usr/bin/env python3
"""Stop this completed experiment's two containers after preserving its evidence."""
import datetime
import json
import pathlib
import socket
import subprocess

root = pathlib.Path('/data1/tq-128k-ab-20261010')
assert json.loads((root / 'status.json').read_text())['stage'] == 'matrix_finished'
assert (root / 'deadline-continuation-recovered2.exit').read_text().strip() == '0'
assert (root / 'artifacts/model-weights-sha256.json').is_file()
expected = {'tq-128k-ab-20261010': 'ea464bbe',
            'tq-128k-ab-20261010-runtime': '0354eaaf'}
for name, prefix in expected.items():
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert container['Id'].startswith(prefix), name
    assert container['Image'] == 'sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14'
    assert any(mount['Source'] == str(root) and mount['Destination'] == '/ws'
               for mount in container['Mounts'])
for name in expected:
    subprocess.run(['docker', 'stop', '--time', '20', name], check=True, timeout=45)
for name in expected:
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert not container['State']['Running'], name
    print(json.dumps({'name': name, 'id': container['Id'],
                      'status': container['State']['Status'],
                      'finished_at': container['State']['FinishedAt']}), flush=True)
with socket.socket() as connection:
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.bind(('127.0.0.1', 18377))
print(json.dumps({'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'port_18377_bindable': True,
                  'scope': 'only the two task container IDs; containers and mounted artifacts retained'}), flush=True)
subprocess.run(['npu-smi', 'info'], check=True)
