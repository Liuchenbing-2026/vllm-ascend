"""Require positive sanitizer coverage, retaining the known global exit leak."""

from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parent
remote = "/data02/tq-opt-20261011/queued_read_20261014/memory-attempt1"
mapped = "/ws/queued_read_20261014/memory-attempt1"
kernel = "TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0"
command = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc",
           "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && "
           "export PYTHONPATH=/ws/queued_read_20261014:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && "
           f"exec mssanitizer --tool=memcheck --kernel-name={kernel} --leak-check=yes --log-file={mapped}/memcheck.log -- /ws/.venv/bin/python -u "
           f"/ws/queued_read_20261014/gate-attempt1/verify_queued_read.py --precision-only --output-dir {mapped}"]
host = f'''import pathlib,subprocess,json,re
root=pathlib.Path({remote!r})
assert (root.parent/'gate-attempt1/gate.exit').read_text().strip()=='0'
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resources.log').write_text(npu)
with (root/'stdout.log').open('w') as output:
 result=subprocess.run(json.loads({json.dumps(command)!r}),stdout=output,stderr=subprocess.STDOUT,timeout=1800)
(root/'tool.exit').write_text(str(result.returncode)+'\\n')
result.check_returncode()
report=json.loads((root/'verify-native-read.json').read_text())
assert report['passed']==15 and report['async_passed']==20
log=(root/'memcheck.log').read_text()
starts=log.count('Start memcheck sanitizer on kernel {kernel}')
clean=log.count('Sanitizer finished on kernel {kernel}. No error detected.')
assert starts==clean==38, (starts,clean)
leaks=[int(x) for x in re.findall(r'SUMMARY: ([0-9]+) byte\(s\) leaked',log)]
errors=[line for line in log.splitlines() if 'ERROR:' in line and 'LeakCheck' not in line]
assert not errors, errors
(root/'memory-review.json').write_text(json.dumps({{'native_invocations':starts,'invocations_without_access_error':clean,'passed_codec_cases':15,'passed_async_cases':20,'exit_leak_bytes':leaks,'all_memory_issues_closed':not leaks,'scope':'known baseline2MiB exit leak remains separately open; native access checks are positive coverage'}})+'\\n')
(root/'execution.exit').write_text('0\\n')
'''
deploy = f'''import pathlib,subprocess,json
root=pathlib.Path({remote!r})
assert not root.exists()
root.mkdir()
(root/'host.py').write_text({host!r})
(root/'command.json').write_text({json.dumps(command)!r})
with (root/'host.log').open('w') as output:
 p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'host.pid').write_text(str(p.pid))
print(json.dumps({{'pid':p.pid,'root':str(root)}}))
'''
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "queued-memory-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode();print(result.stdout.decode())
