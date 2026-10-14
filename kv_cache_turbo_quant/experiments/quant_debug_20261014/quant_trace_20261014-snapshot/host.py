"""Durable single-invocation allocation diagnostic after reader tests finish."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback


def main():
    root = Path("/data02/tq-opt-20261011/quant_trace_20261014")
    code = 1
    try:
        for name, sha in json.loads((root / "source-manifest.json").read_text()).items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
        def command(tag, body):
            directory = root / tag
            directory.mkdir()
            cmd = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "tq-opt-20261011-model", "bash", "-lc",
                   "source /usr/local/Ascend/cann-9.1.0/set_env.sh && " + body]
            (directory / "command.json").write_text(json.dumps(cmd) + "\n")
            with (directory / "stdout.log").open("w") as out:
                result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=1200)
            (directory / "tool.exit").write_text(str(result.returncode) + "\n")
            result.check_returncode()
        command("build", "export TORCH_EXTENSIONS_DIR=/ws/quant_trace_20261014/torch-extensions MAX_JOBS=1 && exec /ws/.venv/bin/python /ws/quant_trace_20261014/build_quant_trace.py")
        while not (root.parent / "read_direct_constants_20261014/host.exit").exists():
            time.sleep(10)
        devices = subprocess.check_output(["npu-smi", "info"], text=True)
        (root / "resources-before.log").write_text(devices)
        assert all("No running processes found in NPU " + str(n) in devices for n in [2, 5])
        command("memory", "source /ws/quant_debug_20261014/opp/vendors/tqquantdebug/bin/set_env.bash && exec mssanitizer --tool=memcheck "
                "--kernel-name=TqQuantDebug_ea037d55979b407c5d1246855d9dec18_0 --leak-check=yes --log-file=/ws/quant_trace_20261014/memory/memcheck.log "
                "-- /ws/.venv/bin/python -u /ws/quant_trace_20261014/single.py --variant debug")
        code = 0
    except Exception:
        (root / "failure.log").write_text(traceback.format_exc())
        raise
    finally:
        (root / "host.exit").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
