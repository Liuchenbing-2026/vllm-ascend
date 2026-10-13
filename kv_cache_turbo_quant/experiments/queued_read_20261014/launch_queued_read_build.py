"""Build queue-aware glue; retain original ACLNN and device kernel artifacts."""

from pathlib import Path
import io
import json
import subprocess
import tarfile

root = Path(__file__).resolve().parent
source = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant/code/integration"
files = {name: (source / name).read_bytes() for name in ["kvtq_native_read.py", "torch_ext/tq_queued_read.cpp", "torch_ext/build_queued_read.py"]}
host = '''import pathlib,subprocess,json,time
root=pathlib.Path('/data02/tq-opt-20261011/queued_read_20261014')
prior=pathlib.Path('/data02/tq-opt-20261011/read_tiles_20261014/gate-attempt2/gate.exit')
while not prior.exists():time.sleep(10)
command=['docker','exec','tq-opt-20261011-model','bash','-lc',
 'source /usr/local/Ascend/cann-9.1.0/set_env.sh && export TORCH_EXTENSIONS_DIR=/ws/queued_read_20261014/torch-extensions MAX_JOBS=2 && exec /ws/.venv/bin/python /ws/queued_read_20261014/torch_ext/build_queued_read.py']
(root/'build.command.json').write_text(json.dumps(command)+'\\n')
with (root/'build.log').open('w') as out:
 result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
(root/'build.exit').write_text(str(result.returncode)+'\\n')
result.check_returncode()
'''
files["host.py"] = host.encode()
archive = root / "queued-read-input.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for name, content in files.items():
        info = tarfile.TarInfo(name);info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
with archive.open("rb") as payload:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/queued_read_20261014.input.tar.gz"], stdin=payload, capture_output=True)
result.check_returncode()
deploy = '''import pathlib,tarfile,subprocess,json
root=pathlib.Path('/data02/tq-opt-20261011/queued_read_20261014')
assert not root.exists()
root.mkdir()
with tarfile.open('/data02/tq-opt-20261011/queued_read_20261014.input.tar.gz') as tar:tar.extractall(root)
with (root/'host.log').open('w') as out:
 p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'host.pid').write_text(str(p.pid))
print(json.dumps({'pid':p.pid,'root':str(root)}))
'''
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "queued-build-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode();print(result.stdout.decode())
