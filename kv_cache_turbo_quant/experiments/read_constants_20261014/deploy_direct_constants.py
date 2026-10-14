"""Retain the constant-table device kernel and isolate direct PTA submission."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    source = root.parent.parent / "source/tq-read-constants-20261014/kv_cache_turbo_quant"
    baseline = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant"
    ext = source / "code/integration/torch_ext"
    text = (baseline / "code/integration/torch_ext/tq_native_read.cpp").read_text()
    text = text.replace("aclnnTqPagedRead", "aclnnTqPagedReadConst").replace('"tq_paged_read', '"tq_paged_read_const_direct')
    text = text.replace("/ws/native_read_20261013/opp/vendors/tqread", "/ws/read_constants_20261014/opp/vendors/tqreadconst")
    text = text.replace("lookup.numel() == 512", "lookup.numel() == 5696")
    (ext / "tq_direct_constants.cpp").write_text(text)
    text = (baseline / "code/integration/torch_ext/build_native_read.py").read_text().replace("tq_native_read", "tq_direct_constants")
    (ext / "build_direct_constants.py").write_text(text)
    queued = root / "read-constants-input/code/integration/kvtq_native_read.py"
    text = queued.read_text().replace("read_constants_20261014/torch-extensions/tq_read_constants/tq_read_constants.so",
                                    "read_direct_constants_20261014/torch-extensions/tq_direct_constants/tq_direct_constants.so")
    text = text.replace('"tq_paged_read_const"', '"tq_paged_read_const_direct"').replace("torch.ops.turboquant.tq_paged_read_const(", "torch.ops.turboquant.tq_paged_read_const_direct(")
    (source / "code/integration/kvtq_native_read.py").write_text(text)
    target = root / "read-direct-constants-input"
    assert not target.exists()
    shutil.copytree(root / "read-constants-input", target, ignore=shutil.ignore_patterns("__pycache__", "source-manifest.json"))
    shutil.rmtree(target / "code/integration")
    shutil.copytree(source / "code/integration", target / "code/integration", ignore=shutil.ignore_patterns("__pycache__"))
    verify = (target / "scripts/verify.py").read_text().replace("read_constants_20261014", "read_direct_constants_20261014")
    (target / "scripts/verify.py").write_text(verify)
    host = (root / "host_read_constants.py").read_text().replace("read_constants_20261014", "read_direct_constants_20261014")
    host = host.replace('previous = root.parent / "quant_debug_20261014/diagnostic-attempt3.exit"', 'previous = root.parent / "read_constants_20261014/host-attempt2.exit"')
    host = host.replace('command("build", "exec bash /ws/read_direct_constants_20261014/scripts/build_read_constants.sh")',
                        'command("build", "export TORCH_EXTENSIONS_DIR=/ws/read_direct_constants_20261014/torch-extensions MAX_JOBS=1 && exec /ws/.venv/bin/python /ws/read_direct_constants_20261014/code/integration/torch_ext/build_direct_constants.py")')
    host = host.replace('(root / "opp").rglob', '(root.parent / "read_constants_20261014/opp").rglob')
    host = host.replace('str(p.relative_to(root))', '"../read_constants_20261014/" + str(p.relative_to(root.parent / "read_constants_20261014"))')
    host = host.replace('root / "torch-extensions/tq_read_constants/tq_read_constants.so"', 'root / "torch-extensions/tq_direct_constants/tq_direct_constants.so"')
    host = host.replace('/ws/read_direct_constants_20261014/opp/vendors/tqreadconst', '/ws/read_constants_20261014/opp/vendors/tqreadconst')
    (target / "host.py").write_text(host)
    (root / "host_direct_constants.py").write_text(host)
    (target / "source-manifest.json").write_text(json.dumps({str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in target.rglob("*") if p.is_file()}, indent=2) + "\n")
    subprocess.run(["python3", "-m", "py_compile", str(target / "host.py"), str(target / "scripts/verify.py")], check=True)
    archive = root / "read_direct_constants_20261014.input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in target.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(target)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/read_direct_constants_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/read_direct_constants_20261014')
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
    (root / "direct-constants-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    print(result.stdout.decode())


if __name__ == "__main__":
    main()
