"""Reuse immutable device binaries, gate the inference-mode integration fix."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    source = root.parent.parent / "source/tq-paired-read-20261014/kv_cache_turbo_quant/code/integration"
    target = root / "paired-inference-input"
    assert not target.exists()
    shutil.copytree(source, target / "code/integration", ignore=shutil.ignore_patterns("__pycache__"))
    (target / "scripts").mkdir()
    shutil.copy2(root / "verify_inference_lookup.py", target / "scripts/cpu.py")
    verify = (root / "verify_paired_read.py").read_text().replace("/ws/paired_read_20261014/code/integration", "/ws/paired_read_inference_20261014/code/integration")
    needle = "    assert len(records) == 15 and len(async_records) == 20 and len(lookup_records) == 2"
    insert = """    candidate = load("candidate_store", "/ws/paired_read_inference_20261014/code/integration/kvtq_store.py")
    inference_records = []
    with torch.inference_mode():
        tracked = candidate._get_byte_lut(key.device)
        assert not torch.is_inference(tracked), "production constant LUT must retain its version counter"
        untracked = lookup.clone()
        assert torch.is_inference(untracked)
        for name, source in [("production_tracked", tracked), ("untracked_input", untracked)]:
            for mutation in [False, True]:
                if mutation:
                    source.mul_(2)
                expected = (old(key, table, lengths, source, metadata), old(value, table, lengths, source, metadata))
                count = compare(call(key, value, table, lengths, source, metadata), expected)
                inference_records.append({"source": name, "inplace_mutation": mutation, "mismatched_bf16_bits": count})
    assert len(inference_records) == 4
"""
    assert needle in verify
    verify = verify.replace(needle, insert + needle).replace("invocations == 36", "invocations == 40")
    verify = verify.replace('"expected_filtered_kernel_invocations": invocations', '"inference_passed": len(inference_records), "inference_records": inference_records, "expected_filtered_kernel_invocations": invocations')
    (target / "scripts/verify.py").write_text(verify)
    host = (root / "host_paired_read.py").read_text().replace("paired_read_20261014", "paired_read_inference_20261014")
    host = host.replace('previous = root.parent / "quant_trace_20261014/host.exit"', 'previous = root.parent / "paired_model_attempt2_20261014/host.exit"')
    host = host.replace('command("build", "exec bash /ws/paired_read_inference_20261014/scripts/build_paired_read.sh")',
                        'assert (root.parent / "paired_read_20261014/build/tool.exit").read_text().strip() == "0"\n        command("cpu", "exec /ws/.venv/bin/python -u /ws/paired_read_inference_20261014/scripts/cpu.py")')
    host = host.replace('(root / "opp").rglob', '(root.parent / "paired_read_20261014/opp").rglob')
    host = host.replace('str(p.relative_to(root))', '"../paired_read_20261014/" + str(p.relative_to(root.parent / "paired_read_20261014"))')
    host = host.replace('root / "torch-extensions/tq_paired_read/tq_paired_read.so"', 'root.parent / "paired_read_20261014/torch-extensions/tq_paired_read/tq_paired_read.so"')
    host = host.replace('str(glue.relative_to(root))', '"../paired_read_20261014/" + str(glue.relative_to(root.parent / "paired_read_20261014"))')
    host = host.replace('/ws/paired_read_inference_20261014/opp', '/ws/paired_read_20261014/opp')
    host = host.replace("starts == clean == 36", "starts == clean == 40")
    host = host.replace('and len(result["lookup_records"]) == 2', 'and len(result["lookup_records"]) == 2 and result["inference_passed"] == 4')
    (target / "host.py").write_text(host);(root / "host_paired_inference.py").write_text(host)
    (target / "source-manifest.json").write_text(json.dumps({str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in target.rglob("*") if p.is_file()}, indent=2) + "\n")
    subprocess.run(["python3", "-m", "py_compile", str(target / "host.py"), str(target / "scripts/cpu.py"), str(target / "scripts/verify.py")], check=True)
    archive = root / "paired_read_inference_20261014.input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in target.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:tar.add(p, arcname=str(p.relative_to(target)))
    with archive.open("rb") as payload:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "cat > /data02/tq-opt-20261011/paired_read_inference_20261014.input.tar.gz"], stdin=payload, capture_output=True)
    result.check_returncode()
    deploy = """from pathlib import Path
import subprocess,tarfile,json,datetime
root=Path('/data02/tq-opt-20261011/paired_read_inference_20261014')
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
    (root / "paired-inference-deploy.log").write_bytes(result.stdout + result.stderr)
    result.check_returncode();print(result.stdout.decode())


if __name__ == "__main__":
    main()
