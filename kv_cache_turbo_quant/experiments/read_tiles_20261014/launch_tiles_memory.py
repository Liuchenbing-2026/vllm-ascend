"""Complete access checks for the already rejected tiled reader in isolation."""

from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parent
target = "/data02/tq-opt-20261011/read_tiles_20261014/memory-attempt1"
kernel = "TqPagedReadV2_5e3fe7929c7efdac33a221b7dfbed4b7_0"
host = f'''from pathlib import Path
import subprocess,json,re,traceback
root=Path({target!r});code=1
try:
 devices=subprocess.check_output(['npu-smi','info'],text=True)
 assert 'No running processes found in NPU 2' in devices and 'No running processes found in NPU 5' in devices
 (root/'resources.log').write_text(devices)
 command=['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=2,5','-e','VLLM_ASCEND_KVTQ_BITS=4','tq-opt-20261011-model','bash','-lc',
  'source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/read_tiles_20261014/opp/vendors/tqreadv2/bin/set_env.bash && export PYTHONPATH=/ws/read_tiles_20261014/code/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && exec mssanitizer --tool=memcheck --kernel-name={kernel} --leak-check=yes --log-file=/ws/read_tiles_20261014/memory-attempt1/memcheck.log -- /ws/.venv/bin/python -u /ws/read_tiles_20261014/gate-attempt2/verify_read_tiles.py --precision-only --output-dir /ws/read_tiles_20261014/memory-attempt1']
 (root/'command.json').write_text(json.dumps(command)+'\\n')
 with (root/'stdout.log').open('w') as out:r=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=1800)
 (root/'tool.exit').write_text(str(r.returncode)+'\\n');r.check_returncode()
 log=(root/'memcheck.log').read_text();start=log.count('Start memcheck sanitizer on kernel {kernel}');clean=log.count('Sanitizer finished on kernel {kernel}. No error detected.')
 assert start==clean==14,(start,clean)
 assert json.loads((root/'verify-native-read.json').read_text())['passed']==15
 leaks=[int(x) for x in re.findall(r'SUMMARY: ([0-9]+) byte\\(s\\) leaked',log)]
 (root/'memory-review.json').write_text(json.dumps({{'checked_invocations':start,'invocations_without_access_error':clean,'exit_leak_bytes':leaks,'all_memory_issues_closed':not leaks,'candidate_rejected_for_performance':True}})+'\\n')
 code=0
except Exception:
 (root/'failure.log').write_text(traceback.format_exc());raise
finally:(root/'execution.exit').write_text(str(code)+'\\n')
'''
deploy = f'''from pathlib import Path
import subprocess,json,datetime
root=Path({target!r});assert not root.exists();root.mkdir()
(root/'host.py').write_text({host!r})
with (root/'host.log').open('w') as out:p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={{'pid':p.pid,'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root)}}
(root/'background-launch.json').write_text(json.dumps(record)+'\\n');print(json.dumps(record))
'''
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "tiles-memory-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
