"""Correlate wrapper allocation ranges with the symbol-enabled memory warning."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    source = root.parent.parent / "source/tq-quant-debug-20261014/kv_cache_turbo_quant/code/integration/torch_ext"
    target = root / "quant-trace-input"
    assert not target.exists()
    target.mkdir()
    for name in ["tq_quant_trace.cpp", "build_quant_trace.py"]:
        shutil.copy2(source / name, target / name)
    single = (root / "quant-debug-input/scripts/single.py").read_text()
    single = single.replace("/ws/quant_debug_20261014/torch-extensions/tq_quant_debug/tq_quant_debug.so",
                            "/ws/quant_trace_20261014/torch-extensions/tq_quant_trace/tq_quant_trace.so").replace("kv_cache_turbo_quant_debug", "kv_cache_turbo_quant_trace")
    (target / "single.py").write_text(single)
    shutil.copy2(root / "host_quant_trace.py", target / "host.py")
    (target / "source-manifest.json").write_text(json.dumps({str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in target.rglob("*") if p.is_file()}, indent=2) + "\n")
    archive = root / "quant_trace_20261014.input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in target.iterdir():
            tar.add(path, arcname=path.name)
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/quant_trace_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/quant_trace_20261014')
assert not root.exists()
root.mkdir()
with tarfile.open(str(root)+'.input.tar.gz') as tar:tar.extractall(root)
with (root/'host.log').open('w') as output:
 p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':p.pid,'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root)}
(root/'background-launch.json').write_text(json.dumps(record)+'\\n')
print(json.dumps(record))
"""
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
    (root / "quant-trace-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    print(result.stdout.decode())


if __name__ == "__main__":
    main()
