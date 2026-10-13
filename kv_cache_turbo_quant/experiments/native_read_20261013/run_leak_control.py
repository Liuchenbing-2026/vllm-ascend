"""Compare the SDK exit-leak report with the unchanged reference path."""

import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
original = (root / "verify_native_read.py").read_text()
script = original.replace("    from kvtq_native_read import NativePagedReader", """    class NativePagedReader:
        def __call__(self, cache, table, lengths, lookup, metadata):
            return reference._dequant_dense_reference(None, cache.view(torch.bfloat16), table, lengths)
""")
(root / "verify_leak_reference.py").write_text(script)
remote = "/data02/tq-opt-20261011/native_read_20261013/leak-reference-attempt1"
mapped = "/ws/native_read_20261013/leak-reference-attempt1"
command = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc",
           "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && "
           f"exec mssanitizer --tool=memcheck --kernel-name=TqPagedRead_dc907b552f63b34c0dcc1d2434a41daf_0 --leak-check=yes --log-file={mapped}/memcheck.log -- "
           f"/ws/.venv/bin/python -u {mapped}/reference.py --precision-only --output-dir {mapped}"]
host = f'''import pathlib,subprocess,json
root=pathlib.Path({remote!r})
with (root/'stdout.log').open('w') as out:
 result=subprocess.run(json.loads({json.dumps(command)!r}),stdout=out,stderr=subprocess.STDOUT,timeout=1200)
(root/'tool.exit').write_text(str(result.returncode)+'\\n')
result.check_returncode()
assert json.loads((root/'verify-native-read.json').read_text())['passed']==15
(root/'execution.exit').write_text('0\\n')
'''
deploy = f'''import pathlib,subprocess,json
root=pathlib.Path({remote!r})
assert not root.exists()
root.mkdir()
(root/'reference.py').write_text({script!r})
(root/'host.py').write_text({host!r})
(root/'command.json').write_text({json.dumps(command)!r})
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resources.log').write_text(npu)
with (root/'host.log').open('w') as out:
 p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'host.pid').write_text(str(p.pid))
print(json.dumps({{'pid':p.pid,'root':str(root)}}))
'''
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "native-leak-control-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode()
print(result.stdout.decode())
