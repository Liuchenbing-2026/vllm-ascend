#!/usr/bin/env python3
"""Release only this runtime after all formal short cohorts have finished."""
import datetime
import json
import pathlib
import socket
import subprocess


def main():
    root = pathlib.Path('/data1/tq-128k-ab-20261010/short_ab_20261010')
    assert (root/'controller.exit').read_text().strip() == '0'
    status = json.loads((root/'status.json').read_text())
    assert status['stage'] == 'short_finished'
    run_id = pathlib.PurePosixPath(status['result_dir']).name
    checked = 0
    for scenario, input_len, output_len, concurrency in [(1, 20, 20, 16), (2, 2000, 200, 60)]:
        for mode in ['bf16', 'store4']:
            for repeat in range(1, 4):
                path = root/'results'/run_id/f's{scenario}_{mode}_in{input_len}_out{output_len}_c{concurrency}_r{repeat}.json'
                result = json.loads(path.read_text())
                assert result['complete_cohort'] and result['successful'] == concurrency and result['failed'] == 0
                assert all(r['usage']['prompt_tokens'] == input_len and r['usage']['completion_tokens'] == output_len
                           for r in result['requests'])
                checked += 1
    name = 'tq-128k-ab-20261010-runtime'
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert container['Id'].startswith('0354eaaf')
    assert container['Image'] == 'sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14'
    assert any(m['Source'] == str(root.parent) and m['Destination'] == '/ws' for m in container['Mounts'])
    processes = subprocess.check_output(['docker', 'top', name, '-eo', 'pid,args'], text=True)
    assert 'run_short.py' not in processes and 'VLLM::' not in processes
    subprocess.run(['docker', 'stop', '--time', '20', name], check=True, timeout=45)
    after = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert not after['State']['Running']
    with socket.socket() as port:
        port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port.bind(('127.0.0.1', 18377))
    info = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert 'No running processes found in NPU 0' in info
    assert 'No running processes found in NPU 1' in info
    print(json.dumps({'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'formal_cohorts_verified': checked, 'container': name,
                      'id': after['Id'], 'state': after['State'],
                      'port_18377_bindable': True,
                      'scope': 'only preserved task runtime stopped; all mounted evidence retained'}), flush=True)
    print(info, flush=True)


if __name__ == '__main__':
    main()
