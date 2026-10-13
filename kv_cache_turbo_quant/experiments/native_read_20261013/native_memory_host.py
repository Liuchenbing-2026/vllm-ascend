import json,pathlib,subprocess
root=pathlib.Path('/data02/tq-opt-20261011/native_read_20261013/memory-attempt2')
command=json.loads('["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && export PYTHONPATH=/ws/native_read_20261013/code/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && exec mssanitizer --tool=memcheck --kernel-name=TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0 --leak-check=yes --log-file=/ws/native_read_20261013/memory-attempt2/memcheck.log -- /ws/.venv/bin/python -u /ws/native_read_20261013/gate-attempt3/verify_native_read.py --precision-only --output-dir /ws/native_read_20261013/memory-attempt2"]')
try:
 with (root/'stdout.log').open('w') as output:
  result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
 (root/'tool.exit').write_text(str(result.returncode)+'\n')
 result.check_returncode()
 report=json.loads((root/'verify-native-read.json').read_text())
 assert report['passed']==15, 'CLI exit without all actual test cases is not a successful run'
 text=(root/'memcheck.log').read_text()
 starts=text.count('Start memcheck sanitizer on kernel TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0')
 clean=text.count('Sanitizer finished on kernel TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0. No error detected.')
 assert starts==clean==14, (starts,clean)
 assert '[mssanitizer] Error' not in text and '[mssanitizer] ERROR' not in text, 'inspect full error log'
 (root/'memory-review.json').write_text(json.dumps({'tool':'memcheck','kernel':'TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0','passed_precision_cases':15,'native_invocations':starts,'invocations_without_error':clean,'scope':'native-reader memory checks only; not semantic quality or a racecheck'})+'\n')
 (root/'execution.exit').write_text('0\n')
except BaseException:
 (root/'execution.exit').write_text('1\n')
 raise
