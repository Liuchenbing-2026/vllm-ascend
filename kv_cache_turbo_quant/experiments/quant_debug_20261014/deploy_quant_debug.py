"""Launch a durable symbol-enabled diagnostic, without a model-service release."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(["python3", str(root / "prepare_quant_debug.py")], check=True)
    source = root.parent.parent / "source/tq-quant-debug-20261014/kv_cache_turbo_quant"
    target = root / "quant-debug-input"
    assert not target.exists()
    for relative in ["code/op/TqQuantDebug.json", "code/op/op_host/tq_quant_debug.cpp", "code/op/op_kernel/tq_quant_debug.cpp",
                     "code/op/op_kernel/tq_quant_debug_tiling.h", "code/integration/torch_ext/tq_quant_debug.cpp",
                     "code/integration/torch_ext/build_quant_debug.py"]:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, path)
    (target / "scripts").mkdir()
    shutil.copy2(root / "build_quant_debug.sh", target / "scripts/build_quant_debug.sh")
    verify = (root / "verify_queued_quant.py").read_text()
    verify = verify.replace('root = Path("/ws/queued_codec_20261014")', 'root = Path("/ws/quant_debug_20261014")')
    verify = verify.replace("    from kvtq_queued_quant import QueuedQuantizer", '    torch.ops.load_library("/ws/quant_debug_20261014/torch-extensions/tq_quant_debug/tq_quant_debug.so")')
    verify = verify.replace("quantizer = QueuedQuantizer()", "quantizer = torch.ops.turboquant.kv_cache_turbo_quant_debug")
    (target / "scripts/verify.py").write_text(verify)
    single = (root / "quant_memory_control.py").read_text()
    single = single.replace('choices=["original", "queued"]', 'choices=["original", "queued", "debug"]')
    single = single.replace('    else:\n        torch.ops.load_library', '    elif args.variant == "debug":\n        torch.ops.load_library("/ws/quant_debug_20261014/torch-extensions/tq_quant_debug/tq_quant_debug.so")\n        output = torch.ops.turboquant.kv_cache_turbo_quant_debug(vectors, rotation, qjl, 4)\n    else:\n        torch.ops.load_library')
    (target / "scripts/single.py").write_text(single)
    shutil.copy2(root / "host_quant_debug.py", target / "host.py")
    manifest = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob("*") if p.is_file()}
    (target / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["python3", "-m", "py_compile", str(target / "host.py"), *map(str, (target / "scripts").glob("*.py"))], check=True)
    subprocess.run(["bash", "-n", str(target / "scripts/build_quant_debug.sh")], check=True)
    archive = root / "quant_debug_20261014.input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in target.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(target)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/quant_debug_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/quant_debug_20261014')
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
    (root / "quant-debug-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    print(result.stdout.decode())


if __name__ == "__main__":
    main()
