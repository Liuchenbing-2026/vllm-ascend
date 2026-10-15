"""Copy only task sources into a new root and detach the finite gate controller."""
import hashlib,importlib.util,json
from pathlib import Path
r=Path(__file__).resolve().parent;src=r.parent.parent/'source/tq-byte-read-20261015/kv_cache_turbo_quant'
files={}
for p in (src/'code/integration').rglob('*'):
 if p.is_file() and p.suffix in ['.py','.cpp'] and '__pycache__' not in p.parts:files[str(p.relative_to(src))]=p.read_text()
for name in ['TqPagedReadByte.json','op_host/tq_paged_read_byte.cpp','op_kernel/tq_paged_read_byte.cpp','op_kernel/tq_paged_read_byte_tiling.h']:files['code/op/'+name]=(src/'code/op'/name).read_text()
for name in ['host_byte_reader.py','verify_byte_read.py','build_byte_read.sh']:files[('host.py' if name=='host_byte_reader.py' else 'scripts/'+name)]=(r/name).read_text()
files['source-manifest.json']=json.dumps({n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},indent=2)+'\n'
for name,s in files.items():
 if name.endswith('.py'):compile(s,name,'exec')
code="""from pathlib import Path
import json,subprocess
root=Path('/data2/tq-profile-20261014/byte_read_20261015');assert not root.exists();root.mkdir()
files=FILES
for name,text in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
with (root/'host.log').open('w') as f:
 c=subprocess.Popen(['nohup','python3','-I','-u',str(root/'host.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'background-launch.json').write_text(json.dumps({'pid':c.pid,'physical_cards':[2,3],'gate_card':2,'new_root':True,'finite':True})+'\\n');print(json.dumps({'pid':c.pid}))
""".replace('FILES',repr(files));compile(code,'deploy','exec')
s=importlib.util.spec_from_file_location('a','/Users/liucb/.codex/memory/private/task18_remote_access.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);x=a.run('18',code,60);(r/'byte-read-launch.log').write_text(x.stdout+x.stderr);x.check_returncode();print(x.stdout)
