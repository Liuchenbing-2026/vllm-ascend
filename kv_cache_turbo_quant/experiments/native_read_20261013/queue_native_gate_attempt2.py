"""Wait for the current matrix, then run strict native-reader gates on owned cards."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/data02/tq-opt-20261011/native_read_20261013/gate-attempt3')
PREVIOUS = Path('/data02/tq-opt-20261011/paired_write_20261013_attempt3')


def main():
    while not (PREVIOUS / 'host.exit').exists():
        (ROOT / 'gate-status.json').write_text(json.dumps({'stage': 'waiting_for_paired_matrix',
            'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        time.sleep(20)
    assert (PREVIOUS / 'host.exit').read_text().strip() == '0', 'retain failed matrix before continuing'
    assert (ROOT.parent / 'build.exit').read_text().strip() == '0', 'native build must pass first'
    npu = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
    (ROOT / 'gate-resource-before.log').write_text(npu)
    cmd = ['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=2,5', '-e', 'VLLM_ASCEND_KVTQ_BITS=4',
        'tq-opt-20261011-model', 'bash', '-lc',
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && '
        'export PYTHONPATH=/ws/native_read_20261013/code/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && '
        'exec /ws/.venv/bin/python -u /ws/native_read_20261013/gate-attempt3/verify_native_read.py --output-dir /ws/native_read_20261013/gate-attempt3']
    (ROOT / 'gate-status.json').write_text(json.dumps({'stage': 'native_precision_and_micro',
        'utc': datetime.now(timezone.utc).isoformat(), 'command': cmd}) + '\n')
    with (ROOT / 'logs/verify-native-read.log').open('w') as output:
        result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        command = cmd.copy()
        command[-1] = command[-1].replace(
            'exec /ws/.venv/bin/python -u',
            'exec mssanitizer --tool=memcheck --leak-check=yes --log-file=/ws/native_read_20261013/gate-attempt3/mssanitizer-memcheck.log /ws/.venv/bin/python -u')
        command[-1] += ' --precision-only --output-dir /ws/native_read_20261013/gate-attempt3/memcheck'
        (ROOT / 'memcheck.command.json').write_text(json.dumps(command) + '\n')
        with (ROOT / 'logs/memcheck.stdout.log').open('w') as output:
            memory = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        (ROOT / 'memcheck.exit').write_text(str(memory.returncode) + '\n')
        # Tool coverage and the full sanitizer log still require review. Do not
        # mistake an exit code for an inspected memory-safety result.
    (ROOT / 'gate.exit').write_text(str(result.returncode) + '\n')
    (ROOT / 'gate-status.json').write_text(json.dumps({'stage': 'native_gate_finished' if result.returncode == 0 else 'native_gate_failed',
        'utc': datetime.now(timezone.utc).isoformat(), 'exit': result.returncode}) + '\n')
    result.check_returncode()


if __name__ == '__main__':
    main()
