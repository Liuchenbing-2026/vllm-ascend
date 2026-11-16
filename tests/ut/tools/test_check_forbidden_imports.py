# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestForbiddenImportsCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = Path(__file__).resolve().parents[3] / "tools" / "check_forbidden_imports.py"

    def run_check(self, *paths):
        return subprocess.run(
            [sys.executable, str(self.script), *(str(path) for path in paths)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def write(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        return path

    def test_clean_file_passes(self):
        result = self.run_check(self.write("clean.py", b"import pathlib\n"))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_forbidden_import_still_fails(self):
        result = self.run_check(self.write("forbidden.py", b"import re\n"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("Found forbidden import: re", result.stdout)

    def test_missing_file_reports_error_without_traceback(self):
        path = self.root / "missing.py"
        result = self.run_check(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn(str(path), result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_non_utf8_file_reports_error_without_traceback(self):
        path = self.write("non_utf8.py", b"\xff\n")
        result = self.run_check(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn(str(path), result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_read_error_does_not_skip_later_files(self):
        result = self.run_check(self.root / "missing.py", self.write("forbidden.py", b"import re\n"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("Found forbidden import: re", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
