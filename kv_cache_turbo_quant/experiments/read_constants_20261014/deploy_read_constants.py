"""Freeze and deploy the constant-table reader; preserve exact input hashes."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(["python3", str(root / "prepare_read_constants.py")], check=True)
    source = root.parent.parent / "source/tq-read-constants-20261014/kv_cache_turbo_quant"
    target = root / "read-constants-input"
    assert not target.exists()
    for relative in ["code/op/TqPagedReadConst.json", "code/op/op_host/tq_paged_read_const.cpp", "code/op/op_kernel/tq_paged_read_const.cpp",
                     "code/op/op_kernel/tq_paged_read_const_tiling.h"]:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, path)
    shutil.copytree(source / "code/integration", target / "code/integration", ignore=shutil.ignore_patterns("__pycache__"))
    (target / "scripts").mkdir()
    shutil.copy2(root / "build_read_constants.sh", target / "scripts/build_read_constants.sh")
    verify = (root / "verify_queued_read.py").read_text().replace("queued_read_20261014", "read_constants_20261014")
    needle = "    order_records = []"
    verify = verify.replace(needle, """    lookup_records = []
    changed = lookup.clone()
    for mutation in [False, True]:
        if mutation:
            changed.mul_(2)
        expected = old_reader(cache, table, lengths, changed, metadata)
        actual = reader(cache, table, lengths, changed, metadata)
        torch.npu.synchronize()
        mismatch = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        assert mismatch == 0, mismatch
        lookup_records.append({"inplace_mutation": mutation, "mismatched_bf16_bits": mismatch})
""" + needle)
    verify = verify.replace('"async_passed":', '"lookup_records": lookup_records,\n                "async_passed":')
    (target / "scripts/verify.py").write_text(verify)
    shutil.copy2(root / "host_read_constants.py", target / "host.py")
    (target / "source-manifest.json").write_text(json.dumps({str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in target.rglob("*") if p.is_file()}, indent=2) + "\n")
    subprocess.run(["python3", "-m", "py_compile", str(target / "host.py"), str(target / "scripts/verify.py")], check=True)
    subprocess.run(["bash", "-n", str(target / "scripts/build_read_constants.sh")], check=True)
    archive = root / "read_constants_20261014.input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in target.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(target)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/read_constants_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/read_constants_20261014')
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
    (root / "read-constants-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    print(result.stdout.decode())


if __name__ == "__main__":
    main()
