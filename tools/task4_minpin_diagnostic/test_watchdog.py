"""CPU-only watchdog checks: Docker is replaced with an inert local stub."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="task4-watchdog-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "action"
        stub = self.root / "docker"
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TASK4_ACTION_LOG"\n')
        stub.chmod(0o700)
        self.env = dict(
            os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"], TASK4_ACTION_LOG=str(self.log)
        )

    def start(self, deadline, container="task4-cpuoffload-031-20261017"):
        return subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("watchdog.py")),
                "--deadline",
                str(deadline),
                "--container",
                container,
            ],
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_cancel_has_no_future_action(self):
        process = self.start(int(time.time()) + 2)
        time.sleep(0.1)
        self.assertIsNone(process.poll())
        process.terminate()
        self.assertEqual(process.wait(timeout=2), -15)
        time.sleep(2.1)
        self.assertFalse(self.log.exists())

    def test_deadline_executes_only_owned_stop(self):
        process = self.start(int(time.time()) - 1)
        self.assertEqual(process.wait(timeout=2), 0)
        self.assertEqual(self.log.read_text().splitlines(), ["stop", "--time", "10", "task4-cpuoffload-031-20261017"])

    def test_other_container_rejected(self):
        process = self.start(int(time.time()) - 1, "someone-else")
        self.assertEqual(process.wait(timeout=2), 2)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
