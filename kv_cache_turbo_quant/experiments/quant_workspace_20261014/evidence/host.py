"""Durable serial gates for queued quantization; never overlap a formal matrix."""

import hashlib
import json
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data02/tq-opt-20261011/quant_ws_20261014")
CONTAINER = "tq-opt-20261011-model"
TILES_KERNEL = "TqPagedReadV2_5e3fe7929c7efdac33a221b7dfbed4b7_0"


def status(stage, **fields):
    record = dict(updated_utc=datetime.now(timezone.utc).isoformat(), stage=stage, **fields)
    (ROOT / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    with (ROOT / "status.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")


def command(tag, body, timeout=1800):
    directory = ROOT / tag
    directory.mkdir(exist_ok=True)
    launch = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", CONTAINER,
              "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && " + body]
    (directory / "command.json").write_text(json.dumps(launch) + "\n")
    status(tag)
    code = 1
    try:
        with (directory / "stdout.log").open("w") as output:
            result = subprocess.run(launch, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        code = result.returncode
        result.check_returncode()
    finally:
        (directory / "execution.exit").write_text(str(code) + "\n")


def review(path, kernel, invocations):
    log = path.read_text()
    starts = log.count("Start memcheck sanitizer on kernel " + kernel)
    clean = log.count("Sanitizer finished on kernel " + kernel + ". No error detected.")
    assert starts == clean == invocations, (starts, clean, invocations)
    errors = [line for line in log.splitlines() if "ERROR:" in line and "LeakCheck" not in line]
    assert not errors, errors
    leaks = [int(x) for x in re.findall(r"SUMMARY: ([0-9]+) byte\(s\) leaked", log)]
    assert all(x == 2097152 for x in leaks), leaks
    return {"kernel": kernel, "checked_invocations": starts, "invocations_without_access_error": clean,
            "exit_leak_bytes": leaks, "all_memory_issues_closed": not leaks,
            "limitation": "Known baseline 2MiB process-exit leak remains unisolated"}


def run():
    previous = ROOT.parent / "queued_model_20261014/background.exit"
    deadline = time.monotonic() + 10800
    while not previous.exists():
        assert time.monotonic() < deadline, "previous matrix did not finish"
        status("waiting_previous_matrix")
        time.sleep(20)
    assert previous.read_text().strip() == "0", "previous matrix failed; inspect before continuing"
    devices = subprocess.check_output(["npu-smi", "info"], text=True)
    (ROOT / "resources-before.log").write_text(devices)
    assert "No running processes found in NPU 2" in devices and "No running processes found in NPU 5" in devices
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    tile_gate = ROOT.parent / "read_tiles_20261014/memory-attempt1/execution.exit"
    while not tile_gate.exists():
        status("waiting_tiled_reader_memory")
        time.sleep(10)
    assert tile_gate.read_text().strip() == "0"
    devices = subprocess.check_output(["npu-smi", "info"], text=True)
    assert "No running processes found in NPU 2" in devices and "No running processes found in NPU 5" in devices
    command("build", "exec bash /ws/quant_ws_20261014/scripts/build_quant_ws.sh")
    capture = "from pathlib import Path;import hashlib,json,re;root=Path('/ws/quant_ws_20261014');files=list((root/'code').rglob('*'))+list((root/'opp').rglob('*'));files.append(root/'torch-extensions/tq_quant_ws/tq_quant_ws.so');names=set();\nfor p in (root/'opp').rglob('*.o'):\n names.update(x.decode() for x in re.findall(rb'TqQuantWs_[0-9a-f]+_[0-9]+',p.read_bytes()))\nprint(json.dumps({'files':{str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files if p.is_file() and '__pycache__' not in p.parts},'kernel_names':sorted(names)}))"
    build = json.loads(subprocess.check_output(["docker", "exec", CONTAINER, "/ws/.venv/bin/python", "-I", "-c", capture]))
    (ROOT / "build-manifest.json").write_text(json.dumps(build["files"], indent=2) + "\n")
    assert len(build["kernel_names"]) == 1, build["kernel_names"]
    kernel = build["kernel_names"][0]
    (ROOT / "kernel-name.json").write_text(json.dumps({"kernel": kernel}) + "\n")
    common = "source /ws/quant_ws_20261014/opp/vendors/tqquantws/bin/set_env.bash && export PYTHONPATH=/ws/quant_ws_20261014/code/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && "
    gate = "/ws/quant_ws_20261014/scripts/verify_quant_ws.py"
    command("gate-attempt1", common + f"exec /ws/.venv/bin/python -u {gate} --output-dir /ws/quant_ws_20261014/gate-attempt1")
    result = json.loads((ROOT / "gate-attempt1/verify-queued-quant.json").read_text())
    assert result["passed"] == 15 and result["async_passed"] == 20
    command("paired-gate", "source /ws/quant_ws_20261014/opp/vendors/tqquantws/bin/set_env.bash && export PYTHONPATH=/ws/quant_ws_model_20261014/candidate_write/integration:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && "
            "exec /ws/.venv/bin/python -u /ws/quant_ws_model_20261014/scripts/verify_pair.py")
    paired = json.loads((ROOT.parent / "quant_ws_model_20261014/verify-pair.json").read_text())
    assert paired["passed"] == 90
    command("memory-attempt1", common + f"exec mssanitizer --tool=memcheck --kernel-name={kernel} --leak-check=yes "
            "--log-file=/ws/quant_ws_20261014/memory-attempt1/memcheck.log -- /ws/.venv/bin/python -u "
            + gate + " --output-dir /ws/quant_ws_20261014/memory-attempt1")
    memory = review(ROOT / "memory-attempt1/memcheck.log", kernel, 35)
    (ROOT / "memory-attempt1/memory-review.json").write_text(json.dumps(memory, indent=2) + "\n")
    (ROOT / "gates.exit").write_text("0\n")
    status("gates_finished_model_released", codec_cases=15, async_cases=20, writer_cases=90, checked_quant_invocations=35)


if __name__ == "__main__":
    code = 1
    try:
        run()
        code = 0
    except Exception:
        (ROOT / "failure.log").write_text(traceback.format_exc())
        status("failed")
        (ROOT / "gates.exit").write_text("1\n")
        raise
    finally:
        (ROOT / "host.exit").write_text(str(code) + "\n")
