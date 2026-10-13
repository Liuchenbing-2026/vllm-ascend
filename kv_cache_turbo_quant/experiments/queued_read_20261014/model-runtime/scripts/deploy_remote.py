
import subprocess,tarfile,json,datetime
from pathlib import Path
root=Path('/data02/tq-opt-20261011/queued_model_20261014')
root.mkdir(exist_ok=True)
assert not (root / 'background-launch.json').exists(), 'already launched; preserve this run'
with tarfile.open('/data02/tq-opt-20261011/queued_model_20261014.input.tar.gz') as tar:tar.extractall(root)
(root / 'logs').mkdir(exist_ok=True)
subprocess.run(['docker','exec','tq-opt-20261011-model','/ws/.venv/bin/python','-m','py_compile',*['/ws/queued_model_20261014/scripts/'+name for name in ['run_8k1.py','bench_client.py','collect_environment.py']]],check=True)
subprocess.run(['docker','exec','tq-opt-20261011-model','bash','-n','/ws/queued_model_20261014/scripts/serve.sh'],check=True)
children={}
for name in ['host','monitor']:
 with (root/'logs'/f'{name}.log').open('w') as output:
  process=subprocess.Popen(['nohup','python3','-I','-u',str(root/'scripts'/f'{name}.py')],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  children[name]=process.pid
record={'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'background_pids':children}
(root/'background-launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
