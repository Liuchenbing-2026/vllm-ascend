#!/usr/bin/env python3
"""Durable detached supervisor, lock and resource-owner evidence."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/data02/tq-opt-20261011/native_combined_20261013')
CONTAINER = 'tq-opt-20261011-model'
CARDS = [2, 5]

def snapshot(container_id):
    text = subprocess.check_output(['npu-smi', 'info'], text=True)
    owners = []
    for line in text.splitlines():
        match = re.match(r'\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|', line)
        if match is None or int(match[1]) not in CARDS:
            continue
        try:
            group = Path(f'/proc/{int(match[2])}/cgroup').read_text()
        except FileNotFoundError:
            continue
        owners.append({'card': int(match[1]), 'pid': int(match[2]), 'owned': container_id in group})
    record = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'cards': CARDS, 'owners': owners}
    with (ROOT / 'resources.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    return record, text

def main():
    lock = (ROOT / 'host.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT / 'background.exit').exists(), 'preserve previous run; use a new attempt root'
    (ROOT / 'host.pid').write_text(str(os.getpid()) + '\n')
    previous = Path('/data02/tq-opt-20261011/native_read_20261013')
    for required in ['gate-attempt3/gate.exit', 'memory-attempt2/execution.exit', 'leak-reference-attempt1/execution.exit']:
        while not (previous / required).exists():
            (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'waiting_for_verified_reader',
                'required': required, 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
            time.sleep(20)
        assert (previous / required).read_text().strip() == '0', required
    report = json.loads((previous / 'gate-attempt3/verify-native-read.json').read_text())
    assert report['passed'] == 15
    times = [r for r in report['timings'] if r.get('tokens') == 120000]
    old = next(r['device_event_mean_us'] for r in times if r['reader'] == 'triton')
    new = next(r['device_event_mean_us'] for r in times if r['reader'] == 'native')
    assert new < old, 'reader did not improve target microcase'
    inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]
    if not inspect['State']['Running']:
        before, _ = snapshot(inspect['Id'])
        assert not before['owners'], 'wait for actual free cards before restarting own container'
        subprocess.run(['docker', 'start', CONTAINER], check=True)
        inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]
    assert inspect['State']['Running'], inspect['State']
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', inspect['Image']]))[0]
    (ROOT / 'container.json').write_text(json.dumps({
        'id': inspect['Id'], 'name': CONTAINER, 'image_tag': inspect['Config']['Image'],
        'image_id': inspect['Image'], 'image_repo_digests': image.get('RepoDigests'),
        'mounts': inspect['Mounts'], 'devices': inspect['HostConfig']['Devices'],
        'ipc_mode': inspect['HostConfig']['IpcMode'], 'network_mode': inspect['HostConfig']['NetworkMode'],
        'privileged': inspect['HostConfig']['Privileged'], 'physical_cards': CARDS}, indent=2) + '\n')
    while True:
        record, text = snapshot(inspect['Id'])
        if not record['owners']:
            (ROOT / 'resource-preflight.log').write_text(text)
            break
        (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'waiting_for_free_cards', **record}) + '\n')
        time.sleep(20)
    command = ['docker', 'exec', '-e', 'TASK_NPU_CARDS=2,5', CONTAINER, 'bash', '-lc',
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec /ws/.venv/bin/python -u /ws/native_combined_20261013/scripts/run_write_ab.py']
    (ROOT / 'launch.command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (ROOT / 'logs/launch.log').open('w') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        while child.poll() is None:
            record, _ = snapshot(inspect['Id'])
            if any(not item['owned'] for item in record['owners']):
                (ROOT / 'resource-conflict.json').write_text(json.dumps(record, indent=2) + '\n')
                pid_file = ROOT / 'controller.pid'
                if pid_file.exists():
                    subprocess.run(['docker', 'exec', CONTAINER, 'kill', '-TERM', pid_file.read_text().strip()], check=False)
                print('interrupted own controller due to foreign NPU owner', flush=True)
                break
            time.sleep(20)
        code = child.wait()
    (ROOT / 'host.exit').write_text(str(code) + '\n')
    print('host supervised run exit', code, flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (ROOT / 'host.exit').write_text('1\n')
        (ROOT / 'host-status.json').write_text(json.dumps({'stage': 'host_failed',
            'error': repr(error), 'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        raise

