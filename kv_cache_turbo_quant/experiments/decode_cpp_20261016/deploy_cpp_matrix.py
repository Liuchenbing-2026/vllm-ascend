"""Create a fresh immutable matrix root; start detached owned supervisors."""
from pathlib import Path
import json,subprocess

r=Path(__file__).resolve().parent
controller=(r/'run_cpp_matrix.py').read_text()
finalizer=(r/'finalize_cpp_matrix.py').read_text()
code=r'''from pathlib import Path
import hashlib,json,shutil,subprocess
base=Path('/data02/tq-opt-20261011');previous=base/'byte_model_host3_attempt4_20261015';root=base/'decode_cpp_model_20261016';gate=base/'decode_cpp_20261016'
assert (previous/'host.exit').read_text().strip()=='0' and (previous/'background.exit').read_text().strip()=='0'
assert (gate/'verify.exit').read_text().strip()=='0'
assert not root.exists()
shutil.copytree(previous,root,ignore=shutil.ignore_patterns('logs','profiles','results','status*','host-status.json','*.exit','*.pid','host.lock','background-launch.json','resource-conflict.json','resources.jsonl','finalizer*','host3-native-gate','acceptance*.json','summary*.json','runtime-monitor.jsonl','__pycache__'))
for p in (root/'scripts').glob('*'):
 if not p.is_file():continue
 s=p.read_text().replace('byte_model_host3_attempt4_20261015','decode_cpp_model_20261016')
 if p.name=='serve.sh':
  s=s.replace('optimized) integration=/ws/byte_cbe_reference_20261015/integration','optimized) integration=/ws/byte_model_host3_attempt4_20261015/candidate_write/integration')
 if p.name=='host.py':
  s=s.replace("previous = Path('/data02/tq-opt-20261011/paired_model_inference_20261014')","previous = Path('/data02/tq-opt-20261011/byte_model_host3_attempt4_20261015')").replace('scripts/run_copy_matrix.py','scripts/run_cpp_matrix.py')
 if p.name=='summarize_matrix.py':
  s=s.replace('SCENES = ["20_20_c16", "2k200_c60", "8k1_c256"]','SCENES = ["20_20_c16", "2k200_c60"]').replace('len(rows) == 9','len(rows) == 6').replace('total == 2988','total == 684')
 if p.suffix=='.py':compile(s,str(p),'exec')
 p.write_text(s)
(root/'scripts/run_cpp_matrix.py').write_text(CONTROLLER)
(root/'scripts/finalize_cpp_matrix.py').write_text(FINALIZER)
shutil.copytree(gate/'integration',root/'candidate_write/integration',dirs_exist_ok=True)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'candidate_write/integration').rglob('*') if p.is_file() and p.suffix in ['.py','.cpp']}
(root/'writer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
cpp=gate/'torch-extensions/tq_decode/tq_decode.so'
(root/'decode-binary.json').write_text(json.dumps({'path':'/ws/decode_cpp_20261016/torch-extensions/tq_decode/tq_decode.so','sha256':hashlib.sha256(cpp.read_bytes()).hexdigest(),'bytes':cpp.stat().st_size},indent=2)+'\n')
delivery=json.loads((gate/'source-delivery.json').read_text())
for name,digest in delivery['production_files'].items():
 assert hashlib.sha256((root/'candidate_write/integration'/name).read_bytes()).hexdigest()==digest,name
(root/'decode-delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
p=root/'migration_golden.py';p.write_text(p.read_text().replace('byte_model_host3_attempt4_20261015','decode_cpp_model_20261016'))
(root/'logs').mkdir();children={}
for tag in ['host','monitor','finalizer']:
 script='finalize_cpp_matrix.py' if tag=='finalizer' else tag+'.py'
 cmd=['nohup','python3','-I','-u',str(root/'scripts'/script)]
 if tag=='finalizer':cmd+=['--root',str(root)]
 with (root/'logs'/(tag+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children[tag]=p.pid
(root/'background-launch.json').write_text(json.dumps({'pids':children,'cards':[2,5],'container':'tq-byte-model-20261015-host3','modes':['bf16','candidate_cpp','optimized_previous_byte'],'formal_requests':684,'token_comparison':440,'source':delivery},indent=2)+'\n')
print(json.dumps({'root':str(root),'pids':children}))
'''.replace('CONTROLLER',repr(controller)).replace('FINALIZER',repr(finalizer))
compile(code,'remote-deploy','exec')
(r/'deploy_cpp_matrix_remote.py').write_text(code)
x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','root@112.29.145.3','python3 -I -'],input=code.encode(),capture_output=True)
(r/'cpp-matrix-launch.log').write_bytes(x.stdout+x.stderr);x.check_returncode();print(x.stdout.decode())
