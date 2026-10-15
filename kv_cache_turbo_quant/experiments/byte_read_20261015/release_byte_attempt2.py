"""Restore exact archived fixtures, retaining the failed first gate root."""
from pathlib import Path
import hashlib,importlib.util,json
r=Path(__file__).resolve().parent
fixture=r.parent/'tq-opt-20261012/final-snapshot/verify'
files={'scripts/'+n:(fixture/n).read_text() for n in ['paged_read_torch.py','kvtq_store_reference.py']}
s=(r/'verify_byte_read.py').read_text().replace('/ws/opt_20261012/verify','/ws/byte_read_attempt2_20261015/scripts');files['scripts/verify_byte_read.py']=s
s=(r/'host_byte_reader.py').read_text();s=s.replace("root=Path('/data2/tq-profile-20261014/byte_read_20261015');container=", "root=Path('/data2/tq-profile-20261014/byte_read_attempt2_20261015');origin=root.parent/'byte_read_20261015';container=")
s=s.replace("run('build','exec bash /ws/byte_read_20261015/scripts/build_byte_read.sh')", "assert (origin/'build/exit').read_text().strip()=='0', 'retain original build failure'")
s=s.replace("(root/'opp')", "(origin/'opp')").replace('str(p.relative_to(root))','str(p.relative_to(origin))')
s=s.replace('/ws/byte_read_20261015/scripts/verify_byte_read.py','/ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py')
for name in ['functional','memory','micro']:s=s.replace('/ws/byte_read_20261015/'+name,'/ws/byte_read_attempt2_20261015/'+name)
files['host.py']=s
for name,s in files.items():compile(s,name,'exec')
(r/'host_byte_attempt2.py').write_text(files['host.py']);(r/'verify_byte_attempt2.py').write_text(files['scripts/verify_byte_read.py'])
files['source-manifest.json']=json.dumps({n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},indent=2)+'\n'
code="""from pathlib import Path
import subprocess,json
old=Path('/data2/tq-profile-20261014/byte_read_20261015');root=old.parent/'byte_read_attempt2_20261015';assert (old/'host.exit').read_text().strip()=='1';assert not root.exists();root.mkdir()
files=FILES
for n,s in files.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
with (root/'host.log').open('w') as f:c=subprocess.Popen(['nohup','python3','-I','-u',str(root/'host.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(root/'background-launch.json').write_text(json.dumps({'pid':c.pid,'preserved_failed_root':str(old),'reused_build_exit':0,'gate_card':2})+'\\n');print(json.dumps({'pid':c.pid}))
""".replace('FILES',repr(files));compile(code,'deploy','exec')
s=importlib.util.spec_from_file_location('a','/Users/liucb/.codex/memory/private/task18_remote_access.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);x=a.run('18',code,45);(r/'byte-read-attempt2-launch.log').write_text(x.stdout+x.stderr);x.check_returncode();print(x.stdout)
