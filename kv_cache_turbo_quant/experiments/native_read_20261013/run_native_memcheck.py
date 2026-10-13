"""Run the corrected SDK CLI and require actual precision-test execution."""

import argparse
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", type=int, default=2)
args = parser.parse_args()
host_root = f"/data02/tq-opt-20261011/native_read_20261013/memory-attempt{args.attempt}"
container_root = f"/ws/native_read_20261013/memory-attempt{args.attempt}"
kernel_name = "TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0"
command = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc",
           "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && "
           "export PYTHONPATH=/ws/native_read_20261013/code/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && "
           f"exec mssanitizer --tool=memcheck --kernel-name={kernel_name} --leak-check=yes --log-file={container_root}/memcheck.log -- /ws/.venv/bin/python -u "
           f"/ws/native_read_20261013/gate-attempt3/verify_native_read.py --precision-only --output-dir {container_root}"]
host = f'''import json,pathlib,subprocess
root=pathlib.Path({host_root!r})
command=json.loads({json.dumps(command)!r})
try:
 with (root/'stdout.log').open('w') as output:
  result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
 (root/'tool.exit').write_text(str(result.returncode)+'\\n')
 result.check_returncode()
 report=json.loads((root/'verify-native-read.json').read_text())
 assert report['passed']==15, 'CLI exit without all actual test cases is not a successful run'
 text=(root/'memcheck.log').read_text()
 starts=text.count('Start memcheck sanitizer on kernel {kernel_name}')
 clean=text.count('Sanitizer finished on kernel {kernel_name}. No error detected.')
 assert starts==clean==14, (starts,clean)
 assert '[mssanitizer] Error' not in text and '[mssanitizer] ERROR' not in text, 'inspect full error log'
 (root/'memory-review.json').write_text(json.dumps({{'tool':'memcheck','kernel':{kernel_name!r},'passed_precision_cases':15,'native_invocations':starts,'invocations_without_error':clean,'scope':'native-reader memory checks only; not semantic quality or a racecheck'}})+'\\n')
 (root/'execution.exit').write_text('0\\n')
except BaseException:
 (root/'execution.exit').write_text('1\\n')
 raise
'''
(root / "native_memory_host.py").write_text(host)
deploy = f'''import pathlib,json,subprocess
root=pathlib.Path({host_root!r})
assert not root.exists(), 'preserve old attempts'
root.mkdir()
(root/'host.py').write_text({host!r})
(root/'command.json').write_text({json.dumps(command)!r}+'\\n')
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resource-before.log').write_text(npu)
with (root/'host.log').open('w') as output:
 process=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'host.pid').write_text(str(process.pid)+'\\n')
print(json.dumps({{'pid':process.pid,'root':str(root)}}))
'''
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "native-memory-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
