"""Run the remaining isolated comparison on free card4; preserve the interrupted root."""
from pathlib import Path
import importlib.util
r=Path(__file__).resolve().parent
code=r'''from pathlib import Path
import json,subprocess,re,time,traceback
base=Path('/data2/tq-profile-20261014');old=base/'byte_read_attempt2_20261015';root=base/'byte_read_micro_20261015';assert not root.exists();root.mkdir()
assert (old/'functional/exit').read_text().strip()=='0'
assert (old/'memory/exit').read_text().strip()=='0'
assert json.loads((old/'memory-review.json').read_text())['positive_calls']==40
controller="""from pathlib import Path
import subprocess,json,re,time,traceback
root=Path('/data2/tq-profile-20261014/byte_read_micro_20261015')
try:
 while True:
  s=subprocess.check_output(['npu-smi','info'],universal_newlines=True)
  owned=[line for line in s.splitlines() if re.match(r'\\|\\s*4\\s+0\\s*\\|\\s*\\d+\\s*\\|',line)]
  if not owned:break
  (root/'status.json').write_text(json.dumps({'stage':'waiting_for_free_card4'})+'\\n');time.sleep(20)
 (root/'resources-before.log').write_text(s)
 cmd=['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=4','-e','VLLM_ASCEND_KVTQ_BITS=4','-e','OMP_NUM_THREADS=1','tq-profile-20261014-host18','bash','-lc','source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && source /ws/paired_read_20261014/opp/vendors/tqreadpair/bin/set_env.bash && source /ws/byte_read_20261015/opp/vendors/tqreadbyte/bin/set_env.bash && exec /ws/.venv/bin/python -u /ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py --output-dir /ws/byte_read_micro_20261015/micro']
 (root/'command.json').write_text(json.dumps(cmd,indent=2)+'\\n');(root/'status.json').write_text(json.dumps({'stage':'micro','card':4})+'\\n')
 with (root/'stdout.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 (root/'host.exit').write_text(str(r.returncode)+'\\n');assert r.returncode==0
 (root/'status.json').write_text(json.dumps({'stage':'micro_complete','scope':'one device-event sample per shape, includes host gaps; no whole-model acceptance'})+'\\n')
except BaseException:
 (root/'failure.log').write_text(traceback.format_exc());(root/'host.exit').write_text('1\\n');raise
"""
compile(controller,'host.py','exec');(root/'host.py').write_text(controller)
with (root/'host.log').open('w') as f:p=subprocess.Popen(['nohup','python3','-I','-u',str(root/'host.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(root/'background-launch.json').write_text(json.dumps({'pid':p.pid,'card':4,'reuses_passed_precision_memory':str(old)})+'\n');print(json.dumps({'pid':p.pid}))
'''
compile(code,'deploy','exec');s=importlib.util.spec_from_file_location('a','/Users/liucb/.codex/memory/private/task18_remote_access.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);x=a.run('18',code,45);(r/'byte-read-micro-launch.log').write_text(x.stdout+x.stderr);x.check_returncode();print(x.stdout)
