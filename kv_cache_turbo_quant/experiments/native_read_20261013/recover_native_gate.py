"""Preserve attempt1 and run corrected imports, strict gates, then memcheck."""

import argparse
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", type=int, default=2)
args = parser.parse_args()
target = f"/data02/tq-opt-20261011/native_read_20261013/gate-attempt{args.attempt}"
container_target = f"/ws/native_read_20261013/gate-attempt{args.attempt}"
queue = (root / "queue_native_gate.py").read_text()
queue = queue.replace("ROOT = Path('/data02/tq-opt-20261011/native_read_20261013')", f"ROOT = Path('{target}')")
queue = queue.replace("(ROOT / 'build.exit')", "(ROOT.parent / 'build.exit')")
queue = queue.replace("exec /ws/.venv/bin/python -u /ws/native_read_20261013/scripts/verify_native_read.py", f"exec /ws/.venv/bin/python -u {container_target}/verify_native_read.py --output-dir {container_target}")
needle = "    (ROOT / 'gate.exit').write_text(str(result.returncode) + '\\n')"
start = queue.index(needle)
queue = queue[:start] + f"""    if result.returncode == 0:
        command = cmd.copy()
        command[-1] = command[-1].replace(
            'exec /ws/.venv/bin/python -u',
            'exec mssanitizer --tool=memcheck --leak-check=yes --log-file={container_target}/mssanitizer-memcheck.log /ws/.venv/bin/python -u')
        command[-1] += ' --precision-only --output-dir {container_target}/memcheck'
        (ROOT / 'memcheck.command.json').write_text(json.dumps(command) + '\\n')
        with (ROOT / 'logs/memcheck.stdout.log').open('w') as output:
            memory = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        (ROOT / 'memcheck.exit').write_text(str(memory.returncode) + '\\n')
        # Tool coverage and the full sanitizer log still require review. Do not
        # mistake an exit code for an inspected memory-safety result.
    (ROOT / 'gate.exit').write_text(str(result.returncode) + '\\n')
""" + queue[start + len(needle) + 1:]
(root / "queue_native_gate_attempt2.py").write_text(queue)
compile(queue, "queue_native_gate_attempt2.py", "exec")
payload = {"verify_native_read.py": (root / "verify_native_read.py").read_text(),
           "queue_native_gate_attempt2.py": queue}
deploy = """import json,pathlib,subprocess,os
root=pathlib.Path(%r)
assert not root.exists(), 'preserve previous attempt; do not overwrite'
(root/'logs').mkdir(parents=True)
payload=json.loads(%r)
for name,content in payload.items():
 (root/name).write_text(content)
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'launch-resource.log').write_text(npu)
with (root/'logs/host.log').open('w') as log:
 process=subprocess.Popen(['python3','-I','-u',str(root/'queue_native_gate_attempt2.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'host.pid').write_text(str(process.pid)+'\\n')
print(json.dumps({'pid':process.pid,'root':str(root)}))
""" % (target, json.dumps(payload))
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "native-gate-attempt2-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
