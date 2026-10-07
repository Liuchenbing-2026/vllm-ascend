# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ownership guard must fail closed on incomplete or contradictory output."""

import importlib.util
import unittest
from pathlib import Path


def load_guard():
    path = Path(__file__).resolve().parents[3] / "benchmarks/diagnostics/guard_idle_npu.py"
    spec = importlib.util.spec_from_file_location("guard_idle_npu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_snapshot


class TestGuardIdleNpu(unittest.TestCase):
    def test_all_requested_cards_must_be_explicitly_idle(self):
        check = load_guard()
        text = "Process id\nNo running processes found in NPU 4\nNo running processes found in NPU 5\n"
        self.assertTrue(check(text, [4, 5])["safe_snapshot"])
        self.assertFalse(check(text, [4, 5, 6])["safe_snapshot"])

    def test_other_cards_do_not_block_the_requested_subset(self):
        check = load_guard()
        text = "Process id\n| 0 0 | 123 | worker | 1000 |\nNo running processes found in NPU 4\n"
        self.assertTrue(check(text, [4])["safe_snapshot"])
        self.assertFalse(check(text, [0, 4])["safe_snapshot"])

    def test_empty_process_name_still_counts_as_occupied(self):
        check = load_guard()
        text = "Process id\n| 4 0 | 456 | | 1000 |\nNo running processes found in NPU 4\n"
        self.assertFalse(check(text, [4])["safe_snapshot"])

    def test_driver_error_is_not_idle(self):
        with self.assertRaises(ValueError):
            load_guard()("dcmi module initialize failed", [4])


if __name__ == "__main__":
    unittest.main()
