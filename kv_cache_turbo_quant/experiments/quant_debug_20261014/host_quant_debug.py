"""Build and locate memory diagnostics; preserve failures and exact commands."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import traceback
from datetime import datetime, timezone


def main():
    root = Path("/data02/tq-opt-20261011/quant_debug_20261014")
    container = "tq-opt-20261011-model"

    def status(stage):
        record = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat()}
        (root / "status.json").write_text(json.dumps(record) + "\n")

    def command(tag, body):
        directory = root / tag
        directory.mkdir()
        cmd = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", container,
               "bash", "-lc", "source /usr/local/Ascend/cann-9.1.0/set_env.sh && " + body]
        (directory / "command.json").write_text(json.dumps(cmd) + "\n")
        status(tag)
        with (directory / "stdout.log").open("w") as out:
            result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=1800)
        (directory / "execution.exit").write_text(str(result.returncode) + "\n")
        result.check_returncode()

    code = 1
    try:
        devices = subprocess.check_output(["npu-smi", "info"], text=True)
        (root / "resources-before.log").write_text(devices)
        assert all("No running processes found in NPU " + str(n) in devices for n in [2, 5])
        for name, sha in json.loads((root / "source-manifest.json").read_text()).items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
        command("build", "exec bash /ws/quant_debug_20261014/scripts/build_quant_debug.sh")
        names = set()
        files = {}
        for p in (root / "opp").rglob("*"):
            if p.is_file():
                data = p.read_bytes()
                files[str(p.relative_to(root))] = hashlib.sha256(data).hexdigest()
                if p.suffix == ".o":
                    names.update(x.decode() for x in re.findall(rb'TqQuantDebug_[0-9a-f]+_0(?![0-9])', data))
        assert len(names) == 1, names
        kernel = names.pop()
        (root / "build-manifest.json").write_text(json.dumps({"files": files, "kernel": kernel}, indent=2) + "\n")
        common = "source /ws/quant_debug_20261014/opp/vendors/tqquantdebug/bin/set_env.bash && "
        command("functional", common + "exec /ws/.venv/bin/python -u /ws/quant_debug_20261014/scripts/verify.py --output-dir /ws/quant_debug_20261014/functional")
        for variant in ["original", "debug"]:
            filter_name = "KvCacheTurboQuant_0bce30e4690f2b43628265dfd3b6214f_0" if variant == "original" else kernel
            command(variant, common + "exec mssanitizer --tool=memcheck --kernel-name=" + filter_name +
                    " --leak-check=yes --log-file=/ws/quant_debug_20261014/" + variant +
                    "/memcheck.log -- /ws/.venv/bin/python -u /ws/quant_debug_20261014/scripts/single.py --variant " + variant)
        status("diagnostics_complete_model_not_released")
        code = 0
    except Exception:
        (root / "failure.log").write_text(traceback.format_exc())
        status("failed")
        raise
    finally:
        (root / "host.exit").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
