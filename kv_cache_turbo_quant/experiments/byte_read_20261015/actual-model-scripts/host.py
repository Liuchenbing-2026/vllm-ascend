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

ROOT = Path('/data02/tq-opt-20261011/byte_model_host3_attempt4_20261015')
CONTAINER = 'tq-byte-model-20261015-host3'
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
    previous = Path('/data02/tq-opt-20261011/paired_model_inference_20261014')
    assert (previous / 'host.exit').read_text().strip() == '0'
    assert (previous / 'background.exit').read_text().strip() == '0'
    profile=json.loads((ROOT/'accepted-host18-profiles.json').read_text())
    for mode in ['candidate','bf16']:
        records=profile['profile-summary-'+mode+'.json'];assert len(records)==8 and all(x['valid'] and len(x['kernel_columns'])>20 for x in records)
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
    command = ['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=2', '-e', 'TORCH_EXTENSIONS_DIR=/ws/torch-extensions', CONTAINER,
        'bash', '-lc', 'source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash && exec /ws/.venv/bin/python -u /ws/byte_model_host3_attempt4_20261015/migration_golden.py']
    (ROOT / 'migration-golden.command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (ROOT / 'logs/migration-golden.log').open('w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    (ROOT / 'migration-golden.exit').write_text(str(result.returncode) + '\n')
    assert result.returncode == 0, 'retain original gate failure'
    gate_command=['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=2','-e','VLLM_ASCEND_KVTQ_BITS=4','-e','OMP_NUM_THREADS=1',CONTAINER,'bash','-lc',
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash && source /ws/paired_read_20261014/opp/vendors/tqreadpair/bin/set_env.bash && source /ws/byte_read_20261015/opp/vendors/tqreadbyte/bin/set_env.bash && exec /ws/.venv/bin/python /ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py --precision-only --output-dir /ws/byte_model_host3_attempt4_20261015/host3-native-gate']
    (ROOT/'host3-native-gate.command.json').write_text(json.dumps(gate_command,indent=2)+'\n')
    with (ROOT/'logs/native-gate.log').open('w') as f:result=subprocess.run(gate_command,stdout=f,stderr=subprocess.STDOUT)
    (ROOT/'host3-native-gate.exit').write_text(str(result.returncode)+'\n');assert result.returncode==0
    import hashlib
    copied = json.loads((Path('/data02/tq-opt-20261011/paired_model_inference_20261014/environment_before.json')).read_text())['native_artifacts']
    new_native = {}
    for path, original in copied.items():
        actual_path = Path('/data02/tq-opt-20261011/byte-vendor-customize-20261015/op_api/lib/libcust_opapi.so') if path.startswith('/usr/local/') else Path('/data02/tq-opt-20261011') / path[4:]
        digest = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        if '/torch-extensions/' not in path:
            assert digest == original['sha256'], path
        else:
            source = Path('/data02/tq-opt-20261011/paired_model_inference_20261014/candidate_write/integration/torch_ext/turboquant_torch.cpp')
            assert hashlib.sha256(source.read_bytes()).hexdigest() == '51fdaeb74f26aa2c9aece180955651dab6e858de943cee9ab105a8022066d6b5'
        new_native[path] = {'sha256': digest, 'bytes': actual_path.stat().st_size}
    (ROOT / 'migration-native-artifacts.json').write_text(json.dumps(new_native, indent=2) + '\n')
    command = ['docker', 'exec', '-e', 'TASK_NPU_CARDS=2,5', CONTAINER, 'bash', '-lc',
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec /ws/.venv/bin/python -u /ws/byte_model_host3_attempt4_20261015/scripts/run_copy_matrix.py']
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

