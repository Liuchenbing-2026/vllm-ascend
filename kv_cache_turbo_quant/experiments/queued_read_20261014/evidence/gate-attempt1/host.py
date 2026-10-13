import pathlib,json,subprocess
root=pathlib.Path('/data02/tq-opt-20261011/queued_read_20261014/gate-attempt1')
assert (root.parent/'build.exit').read_text().strip()=='0'
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resource-before.log').write_text(npu)
with (root/'stdout.log').open('w') as output:
 result=subprocess.run(json.loads('["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && export PYTHONPATH=/ws/queued_read_20261014:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && exec /ws/.venv/bin/python -u /ws/queued_read_20261014/gate-attempt1/verify_queued_read.py --output-dir /ws/queued_read_20261014/gate-attempt1"]'),stdout=output,stderr=subprocess.STDOUT,timeout=1800)
(root/'gate.exit').write_text(str(result.returncode)+'\n')
result.check_returncode()
report=json.loads((root/'verify-native-read.json').read_text())
assert report['passed']==15 and report['async_passed']==20
