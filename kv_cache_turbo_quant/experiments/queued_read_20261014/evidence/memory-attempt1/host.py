import pathlib,subprocess,json,re
root=pathlib.Path('/data02/tq-opt-20261011/queued_read_20261014/memory-attempt1')
assert (root.parent/'gate-attempt1/gate.exit').read_text().strip()=='0'
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resources.log').write_text(npu)
with (root/'stdout.log').open('w') as output:
 result=subprocess.run(json.loads('["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && export PYTHONPATH=/ws/queued_read_20261014:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && exec mssanitizer --tool=memcheck --kernel-name=TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0 --leak-check=yes --log-file=/ws/queued_read_20261014/memory-attempt1/memcheck.log -- /ws/.venv/bin/python -u /ws/queued_read_20261014/gate-attempt1/verify_queued_read.py --precision-only --output-dir /ws/queued_read_20261014/memory-attempt1"]'),stdout=output,stderr=subprocess.STDOUT,timeout=1800)
(root/'tool.exit').write_text(str(result.returncode)+'\n')
result.check_returncode()
report=json.loads((root/'verify-native-read.json').read_text())
assert report['passed']==15 and report['async_passed']==20
log=(root/'memcheck.log').read_text()
starts=log.count('Start memcheck sanitizer on kernel TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0')
clean=log.count('Sanitizer finished on kernel TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0. No error detected.')
assert starts==clean==38, (starts,clean)
leaks=[int(x) for x in re.findall(r'SUMMARY: ([0-9]+) byte\(s\) leaked',log)]
errors=[line for line in log.splitlines() if 'ERROR:' in line and 'LeakCheck' not in line]
assert not errors, errors
(root/'memory-review.json').write_text(json.dumps({'native_invocations':starts,'invocations_without_access_error':clean,'passed_codec_cases':15,'passed_async_cases':20,'exit_leak_bytes':leaks,'all_memory_issues_closed':not leaks,'scope':'known baseline2MiB exit leak remains separately open; native access checks are positive coverage'})+'\n')
(root/'execution.exit').write_text('0\n')
