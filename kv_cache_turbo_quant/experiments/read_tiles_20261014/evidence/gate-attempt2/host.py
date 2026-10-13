import json,pathlib,subprocess,hashlib,datetime
root=pathlib.Path('/data02/tq-opt-20261011/read_tiles_20261014/gate-attempt2')
previous=root.parent
assert (previous/'build.exit').read_text().strip()=='0'
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resource-before.log').write_text(npu)
paths=[p for section in ['code','opp','torch-extensions'] for p in (previous/section).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.o.d','.ninja','.lock'))]
(previous/'build-manifest.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'build_exit':0,'files':{str(p.relative_to(previous)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths}},indent=2)+'\n')
with (root/'stdout.log').open('w') as output:
 result=subprocess.run(json.loads('["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && source /ws/read_tiles_20261014/opp/vendors/tqreadv2/bin/set_env.bash && exec /ws/.venv/bin/python -u /ws/read_tiles_20261014/gate-attempt2/verify_read_tiles.py --output-dir /ws/read_tiles_20261014/gate-attempt2"]'),stdout=output,stderr=subprocess.STDOUT,timeout=1800)
(root/'gate.exit').write_text(str(result.returncode)+'\n')
result.check_returncode()
assert json.loads((root/'verify-native-read.json').read_text())['passed']==15
