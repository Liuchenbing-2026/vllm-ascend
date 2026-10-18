"""CPU-only Bash teardown reproduction. No Docker/NPU calls."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    root = Path(tempfile.mkdtemp(prefix="task4-cleanup-cpu-"))
    results = []
    for mode in ("legacy", "fixed"):
        path = root / mode
        path.mkdir()
        if mode == "legacy":
            script = """#!/bin/bash
    set -Eeuo pipefail
    exec > >(tee -a controller.log) 2>&1
    (sleep 5; echo UNEXPECTED_ACTION > action) & watchdog=$!
    sleep 0.2
    kill "$watchdog"
    echo 0 > controller.exit
    echo FINISHED
    exit 0
    """
        else:
            script = """#!/bin/bash
    set -Eeuo pipefail
    exec >> controller.log 2>&1
    python3 -c 'import time; from pathlib import Path; time.sleep(5); Path("action").touch()' & watchdog=$!
    sleep 0.2
    kill "$watchdog"
    wait "$watchdog" || true
    echo 0 > controller.exit
    echo FINISHED
    exit 0
    """
        (path / "controller.sh").write_text(script)
        with (path / "outer.log").open("w") as output:
            start = time.monotonic()
            process = subprocess.Popen(["bash", "controller.sh"], cwd=path, stdout=output, stderr=subprocess.STDOUT)
            time.sleep(0.6)
            matches = []
            for p in Path("/proc").glob("[0-9]*"):
                try:
                    if os.readlink(p / "cwd") == str(path):
                        matches.append(
                            {
                                "pid": int(p.name),
                                "cmd": (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                                "stat": (p / "stat").read_text(),
                                "wchan": (p / "wchan").read_text(),
                            }
                        )
                except OSError:
                    pass
            early = {
                "process_returncode": process.poll(),
                "exit_file": (path / "controller.exit").exists(),
                "processes": matches,
            }
            code = process.wait(timeout=8)
            elapsed = time.monotonic() - start
            time.sleep(max(0, 5.5 - elapsed))
            results.append(
                {
                    "mode": mode,
                    "early_0_6s": early,
                    "exit": code,
                    "wait_elapsed_seconds": elapsed,
                    "future_action": (path / "action").exists(),
                }
            )
    assert results[0]["early_0_6s"]["exit_file"]
    assert results[0]["early_0_6s"]["processes"], "Legacy orphan not reproduced"
    assert not results[1]["early_0_6s"]["processes"], "Fixed variant leaked processes"
    assert not any(r["future_action"] for r in results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
