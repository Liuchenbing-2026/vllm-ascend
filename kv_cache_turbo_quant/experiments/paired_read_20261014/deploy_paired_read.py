"""Launch isolated paired-reader gates; retain predecessors and their failures."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(["python3", str(root / "prepare_paired_read.py")], check=True)
    source = root.parent.parent / "source/tq-paired-read-20261014/kv_cache_turbo_quant"
    target = root / "paired-read-input"
    assert not target.exists()
    for relative in ["code/op/TqPagedReadPair.json", "code/op/op_host/tq_paged_read_pair.cpp", "code/op/op_kernel/tq_paged_read_pair.cpp",
                     "code/op/op_kernel/tq_paged_read_pair_tiling.h"]:
        p = target / relative;p.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(source / relative, p)
    shutil.copytree(source / "code/integration", target / "code/integration", ignore=shutil.ignore_patterns("__pycache__"))
    (target / "scripts").mkdir()
    shutil.copy2(root / "build_paired_read.sh", target / "scripts/build_paired_read.sh")
    shutil.copy2(root / "verify_paired_read.py", target / "scripts/verify.py")
    host = (root / "host_read_constants.py").read_text().replace("read_constants_20261014", "paired_read_20261014")
    host = host.replace('previous = root.parent / "quant_debug_20261014/diagnostic-attempt3.exit"', 'previous = root.parent / "quant_trace_20261014/host.exit"')
    host = host.replace("build_read_constants.sh", "build_paired_read.sh").replace("tq_read_constants", "tq_paired_read").replace("TqPagedReadConst_", "TqPagedReadPair_").replace("tqreadconst", "tqreadpair")
    host = host.replace("starts == clean == 40", "starts == clean == 36")
    (target / "host.py").write_text(host);(root / "host_paired_read.py").write_text(host)
    (target / "source-manifest.json").write_text(json.dumps({str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in target.rglob("*") if p.is_file()}, indent=2) + "\n")
    subprocess.run(["python3", "-m", "py_compile", str(target / "host.py"), str(target / "scripts/verify.py")], check=True)
    subprocess.run(["bash", "-n", str(target / "scripts/build_paired_read.sh")], check=True)
    archive = root / "paired_read_20261014.input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in target.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:tar.add(p, arcname=str(p.relative_to(target)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/paired_read_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/paired_read_20261014')
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
    (root / "paired-read-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode();print(result.stdout.decode())


if __name__ == "__main__":
    main()
