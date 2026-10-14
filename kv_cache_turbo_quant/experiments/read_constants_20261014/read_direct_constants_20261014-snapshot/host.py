"""Serial build, strict functional checks, positive memory coverage and microbench."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone


def main():
    root = Path("/data02/tq-opt-20261011/read_direct_constants_20261014")

    def status(stage, **fields):
        record = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        (root / "status.json").write_text(json.dumps(record) + "\n")
        with (root / "status.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")

    def command(tag, body):
        directory = root / tag
        directory.mkdir()
        cmd = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model",
               "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && " + body]
        (directory / "command.json").write_text(json.dumps(cmd) + "\n")
        status(tag)
        with (directory / "stdout.log").open("w") as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=1800)
        (directory / "tool.exit").write_text(str(result.returncode) + "\n")
        result.check_returncode()

    code = 1
    try:
        previous = root.parent / "read_constants_20261014/host-attempt2.exit"
        while not previous.exists():
            status("waiting_quant_diagnostic")
            time.sleep(10)
        devices = subprocess.check_output(["npu-smi", "info"], text=True)
        (root / "resources-before.log").write_text(devices)
        assert all("No running processes found in NPU " + str(n) in devices for n in [2, 5])
        for name, sha in json.loads((root / "source-manifest.json").read_text()).items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
        command("build", "export TORCH_EXTENSIONS_DIR=/ws/read_direct_constants_20261014/torch-extensions MAX_JOBS=1 && exec /ws/.venv/bin/python /ws/read_direct_constants_20261014/code/integration/torch_ext/build_direct_constants.py")
        names = set()
        files = {}
        for p in (root.parent / "read_constants_20261014/opp").rglob("*"):
            if p.is_file():
                data = p.read_bytes()
                files["../read_constants_20261014/" + str(p.relative_to(root.parent / "read_constants_20261014"))] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                if p.suffix == ".o":
                    names.update(x.decode() for x in re.findall(rb'TqPagedReadConst_[0-9a-f]+_0(?![0-9])', data))
        glue = root / "torch-extensions/tq_direct_constants/tq_direct_constants.so"
        files[str(glue.relative_to(root))] = {"sha256": hashlib.sha256(glue.read_bytes()).hexdigest(), "bytes": glue.stat().st_size}
        assert len(names) == 1, names
        kernel = names.pop()
        (root / "build-manifest.json").write_text(json.dumps(files, indent=2) + "\n")
        (root / "kernel-name.json").write_text(json.dumps({"kernel": kernel}) + "\n")
        common = "source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && source /ws/read_constants_20261014/opp/vendors/tqreadconst/bin/set_env.bash && "
        entry = "/ws/.venv/bin/python -u /ws/read_direct_constants_20261014/scripts/verify.py"
        command("functional", common + "exec " + entry + " --precision-only --output-dir /ws/read_direct_constants_20261014/functional")
        result = json.loads((root / "functional/verify-native-read.json").read_text())
        assert result["passed"] == 15 and result["async_passed"] == 20 and len(result["lookup_records"]) == 2
        command("memory", common + "exec mssanitizer --tool=memcheck --kernel-name=" + kernel +
                " --leak-check=yes --log-file=/ws/read_direct_constants_20261014/memory/memcheck.log -- " + entry +
                " --precision-only --output-dir /ws/read_direct_constants_20261014/memory")
        log = (root / "memory/memcheck.log").read_text()
        starts = log.count("Start memcheck sanitizer on kernel " + kernel)
        clean = log.count("Sanitizer finished on kernel " + kernel + ". No error detected.")
        leaks = [int(x) for x in re.findall(r"SUMMARY: ([0-9]+) byte\(s\) leaked", log)]
        assert starts == clean == 40, (starts, clean)
        assert all(x == 2097152 for x in leaks), leaks
        assert not [l for l in log.splitlines() if "ERROR:" in l and "LeakCheck" not in l]
        report = {"native_invocations": starts, "invocations_without_access_error": clean, "exit_leak_bytes": leaks,
                  "all_memory_issues_closed": not leaks, "scope": "Changed reader has positive access coverage; original quantizer warnings and baseline exit leak remain open"}
        (root / "memory/memory-review.json").write_text(json.dumps(report, indent=2) + "\n")
        command("micro", common + "exec " + entry + " --output-dir /ws/read_direct_constants_20261014/micro")
        (root / "gates.exit").write_text("0\n")
        status("gates_and_micro_complete", kernel=kernel)
        code = 0
    except Exception:
        (root / "failure.log").write_text(traceback.format_exc())
        (root / "gates.exit").write_text("1\n")
        status("failed")
        raise
    finally:
        (root / "host.exit").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
