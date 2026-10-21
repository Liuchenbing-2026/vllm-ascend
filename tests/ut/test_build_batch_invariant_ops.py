# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Test the complete installer shell script using local packages, without NPU."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestBatchInvariantInstall(unittest.TestCase):
    def run_install(self, scenario, soc="ascend910b", arch="x86_64"):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            commands = work / "commands"
            commands.mkdir()
            (commands / "python").symlink_to(sys.executable)
            (commands / "uname").write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_ARCH"\n')
            curl_source = """#!/bin/bash
set -eu
url="${!#}"
file="${url##*/}"
if [[ "$file" == *.run ]]; then
    [[ "$TEST_SCENARIO" != run_download_failure ]] || exit 7
    [[ "$TEST_SCENARIO" != missing_run_file ]] || exit 0
    code=0
    [[ "$TEST_SCENARIO" != run_install_failure ]] || code=17
    printf '#!/bin/bash\\necho run >> "$TEST_EVENTS"\\nexit %s\\n' "$code" > "$file"
else
    [[ "$TEST_SCENARIO" != wheel_download_failure ]] || exit 7
    [[ "$TEST_SCENARIO" != missing_wheel_file ]] || exit 0
    if [[ "$TEST_SCENARIO" == invalid_zip ]]; then
        printf "not a ZIP" > "$file"
        exit 0
    fi
    python - "$file" <<'PYZIP'
import os, sys, zipfile
scenario = os.environ["TEST_SCENARIO"]
name = ("other/file" if scenario == "missing_package_directory"
        else "torch_ops_extension/batch_invariant_ops/build_and_install.sh")
code = 23 if scenario == "wheel_install_failure" else 0
with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr(name, 'echo wheel >> "$TEST_EVENTS"\\nexit %s\\n' % code)
PYZIP
fi
"""
            (commands / "curl").write_text(curl_source)
            for command in ("curl", "uname"):
                (commands / command).chmod(0o755)
            events = work / "events"
            environment = {
                **os.environ,
                "PATH": str(commands) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
                "TEST_SCENARIO": scenario,
                "TEST_EVENTS": str(events),
                "TEST_ARCH": arch,
            }
            script = Path(__file__).resolve().parents[2] / "csrc/build_batch_invariant_ops.sh"
            result = subprocess.run(
                [shutil.which("bash"), str(script), soc],
                cwd=work,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            event_lines = events.read_text().splitlines() if events.exists() else []
            self.assertFalse(list(work.glob("*.run")), result.stdout + result.stderr)
            self.assertFalse(list(work.glob("*.zip")), result.stdout + result.stderr)
            self.assertFalse((work / "torch_ops_extension").exists(), result.stdout + result.stderr)
            return result, event_lines

    def assert_install_fails(self, scenario, expected_message):
        result, _ = self.run_install(scenario)
        self.assertIn(expected_message, result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("build completed successfully", result.stdout)

    def test_run_download_failure(self):
        self.assert_install_fails("run_download_failure", "Failed to download batch_invariant run package")

    def test_missing_run_download(self):
        self.assert_install_fails("missing_run_file", "Failed to download batch_invariant run package")

    def test_run_install_failure(self):
        self.assert_install_fails("run_install_failure", "Failed to install batch_invariant run package")

    def test_wheel_download_failure(self):
        self.assert_install_fails("wheel_download_failure", "Failed to download batch_invariant whl package")

    def test_missing_wheel_download(self):
        self.assert_install_fails("missing_wheel_file", "Failed to download batch_invariant whl package")

    def test_invalid_zip(self):
        self.assert_install_fails("invalid_zip", "Failed to unzip batch_invariant whl package")

    def test_missing_package_directory(self):
        self.assert_install_fails("missing_package_directory", "batch_invariant_ops directory not found")

    def test_wheel_install_failure(self):
        self.assert_install_fails("wheel_install_failure", "Failed to build and install batch_invariant whl package")

    def test_success(self):
        result, events = self.run_install("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(events, ["run", "wheel"])
        self.assertNotIn("Failed", result.stdout)

    def test_supported_soc_families(self):
        for soc in ("ascend910b", "ascend910_93", "ascend950pr"):
            with self.subTest(soc=soc):
                result, events = self.run_install("success", soc=soc, arch="aarch64")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(events, ["run", "wheel"])

    def test_unsupported_soc_retains_skip(self):
        result, events = self.run_install("success", soc="ascend310p")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skipping", result.stdout)
        self.assertEqual(events, [])

    def test_unsupported_arch_retains_skip(self):
        result, events = self.run_install("success", arch="unsupported-test-arch")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("unknown architecture", result.stdout)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
