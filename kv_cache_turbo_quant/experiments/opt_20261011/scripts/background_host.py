#!/usr/bin/env python3
"""Detached host supervisor; queue until cards are free, preserve ownership logs."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/root/tq-opt-20261011')
CONTAINER = 'tq-opt-20261011-model'
CARDS = [2, 5]


def snapshot(container_id):
    smi = subprocess.check_output(['npu-smi', 'info'], text=True)
    owners = []
    for line in smi.splitlines():
        match = re.match(r'\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|', line)
        if not match or int(match[1]) not in CARDS:
            continue
        card, pid = int(match[1]), int(match[2])
        try:
            cgroup = Path(f'/proc/{pid}/cgroup').read_text()
        except FileNotFoundError:
            continue
        owners.append({'card': card, 'host_pid': pid, 'owned': container_id in cgroup,
                       'cgroup': cgroup.strip()})
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'cards': CARDS, 'owners': owners}
    with (ROOT / 'resource112-background.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    return record, smi


def main():
    lock = (ROOT / 'background-host.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'background.exit').exists():
        attempt = ROOT / 'attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        attempt.mkdir(parents=True, exist_ok=False)
        for name in ['background.exit', 'background-host.exit', 'background-status.json',
                     'background-pipeline.pid', 'model-controller.pid',
                     'logs/background', 'logs/background-launch.log']:
            source = ROOT / name
            if source.exists():
                target = attempt / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
    (ROOT / 'background-host.pid').write_text(str(os.getpid()) + '\n')
    inspect = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER]))[0]
    assert inspect['State']['Running'], inspect['State']
    (ROOT / 'background-cards.json').write_text(json.dumps({'physical_cards': CARDS,
        'container': CONTAINER, 'container_id': inspect['Id']}, indent=2) + '\n')
    while True:
        record, smi = snapshot(inspect['Id'])
        if not record['owners']:
            (ROOT / 'resource112-background-preflight.log').write_text(smi)
            break
        (ROOT / 'background-status.json').write_text(json.dumps({'stage': 'waiting_for_free_cards', **record}, indent=2) + '\n')
        time.sleep(20)
    command = ['docker', 'exec', '-e', 'TASK_NPU_CARDS=2,5', CONTAINER, 'bash', '-lc',
               'source /usr/local/Ascend/cann-9.1.0/set_env.sh; exec /ws/.venv/bin/python -u /ws/opt_20261011/scripts/background_pipeline.py']
    (ROOT / 'background-launch.command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (ROOT / 'logs/background-launch.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            record, _ = snapshot(inspect['Id'])
            if any(not x['owned'] for x in record['owners']):
                (ROOT / 'resource112-background-conflict.json').write_text(json.dumps(record, indent=2) + '\n')
                pid_file = ROOT / 'background-pipeline.pid'
                if pid_file.exists():
                    pid = int(pid_file.read_text())
                    subprocess.run(['docker', 'exec', CONTAINER, 'kill', '-TERM', str(pid)], check=False)
                print('foreign card process detected; interrupted only owned pipeline', flush=True)
                break
            time.sleep(20)
        code = child.wait()
    (ROOT / 'background-host.exit').write_text(str(code) + '\n')
    print('background pipeline exit', code, flush=True)


if __name__ == '__main__':
    main()
