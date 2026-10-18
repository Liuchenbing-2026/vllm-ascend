# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU clocks only; no model imports or device commands."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import controller
from task4_bounded_pin import StageAProbe
from test_bounded_pin import lease_fixture, snapshot_fixture
from timing import PhaseTimer


class TestPhaseTimer(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.lease = dict(lease_fixture(), start_epoch=0, end_epoch=1500)
        (self.root / "lease.json").write_text(json.dumps(self.lease))
        self.timer = PhaseTimer(self.lease, 0, 10000)

    def event(self, name, elapsed, file_rank=None, **changes):
        value = dict(monotonic=10000 + elapsed, lease_id=self.lease["lease_id"], rank=file_rank)
        value.update(changes)
        filename = name + ("" if file_rank is None else f".rank-{file_rank}") + ".json"
        (self.output / filename).write_text(json.dumps(value))

    def loading(self, entry=360, end=None):
        self.event("preflight_complete", 50)
        for rank in range(8):
            self.event("model_entered", entry, rank)
            if end is not None:
                self.event("model_loaded", end, rank)

    def test_initialization_does_not_consume_rank_load_budget(self):
        self.loading()
        self.assertFalse(self.timer.check(self.output, 10600))
        for rank in range(8):
            self.event("model_loaded", 620, rank)
            self.event("probe_complete", 650, rank)
        self.assertTrue(self.timer.check(self.output, 10650))
        self.assertEqual(self.timer.pin_deadline, 10800)

    def test_preflight_and_entry_deadlines(self):
        with self.assertRaisesRegex(TimeoutError, "preflight"):
            self.timer.check(self.output, 10180)
        self.event("preflight_complete", 180)
        with self.assertRaisesRegex(TimeoutError, "preflight"):
            self.timer.check(self.output, 10181)
        self.timer = PhaseTimer(self.lease, 0, 10000)
        self.event("preflight_complete", 50)
        with self.assertRaisesRegex(TimeoutError, "entry"):
            self.timer.check(self.output, 10480)

    def test_missing_slow_rank_does_not_reset_other_deadlines(self):
        self.loading(entry=100)
        for rank in range(1, 8):
            self.event("model_entered", 479, rank)
        with self.assertRaisesRegex(TimeoutError, "Rank 0 load"):
            self.timer.check(self.output, 10520)

    def test_late_return_is_not_accepted_even_if_complete(self):
        self.loading(entry=100, end=520)
        with self.assertRaisesRegex(TimeoutError, "Rank 0 load"):
            self.timer.check(self.output, 10521)
        self.assertIsNone(self.timer.pin_deadline)

    def test_missing_entry_does_not_infer_success_from_load_marker(self):
        self.event("preflight_complete", 50)
        self.event("model_loaded", 300, 0)
        with self.assertRaisesRegex(ValueError, "preceding entry"):
            self.timer.check(self.output, 10301)

    def test_changed_and_disappeared_markers_fail(self):
        self.loading()
        self.timer.check(self.output, 10400)
        self.event("model_entered", 361, 0)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.timer.check(self.output, 10401)
        (self.output / "model_entered.rank-0.json").unlink()
        with self.assertRaisesRegex(ValueError, "disappeared"):
            self.timer.check(self.output, 10401)

    def test_wrong_lease_rank_clock_and_nonfinite_marker(self):
        self.event("preflight_complete", 50)
        for changes in (
            dict(lease_id="old"),
            dict(rank=2),
            dict(monotonic=9999),
            dict(monotonic=float("nan")),
        ):
            with self.subTest(changes=changes):
                self.event("model_entered", 360, 0, **changes)
                with self.assertRaisesRegex(ValueError, "Invalid timing"):
                    self.timer.check(self.output, 10400)

    def test_marker_published_during_poll_is_pending_without_new_budget(self):
        self.event("preflight_complete", 50)
        self.event("model_entered", 400.001, 0)
        self.assertFalse(self.timer.check(self.output, 10400))
        self.assertFalse(self.timer.check(self.output, 10401))
        with self.assertRaisesRegex(TimeoutError, "entry"):
            self.timer.check(self.output, 10480)

    def test_partial_marker_does_not_extend_deadline(self):
        self.event("preflight_complete", 50)
        (self.output / "model_entered.rank-0.json").write_text('{"monotonic":')
        self.assertFalse(self.timer.check(self.output, 10479))
        with self.assertRaisesRegex(TimeoutError, "entry"):
            self.timer.check(self.output, 10480)

    def test_pin_budget_and_absolute_stop(self):
        self.loading(end=620)
        self.assertFalse(self.timer.check(self.output, 10799))
        with self.assertRaisesRegex(TimeoutError, "Pin diagnostics"):
            self.timer.check(self.output, 10800)
        with self.assertRaisesRegex(TimeoutError, "Absolute 18"):
            self.timer.check(self.output, 11080)

    def test_completion_needs_every_rank_and_all_loaded_barrier(self):
        self.loading(end=620)
        for rank in range(7):
            self.event("probe_complete", 650, rank)
        self.assertFalse(self.timer.check(self.output, 10651))
        self.event("probe_complete", 619, 7)
        with self.assertRaisesRegex(ValueError, "without load return"):
            self.timer.check(self.output, 10652)

    def test_anchor_does_not_grant_extra_time_for_late_start(self):
        timer = PhaseTimer(self.lease, 50, 10050)
        self.assertEqual(timer.hard_deadline, 11080)
        for wall in (-1, 60, float("nan")):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                PhaseTimer(self.lease, wall, 10000)

    def monitor(self, wall, mono):
        child = SimpleNamespace(poll=lambda: None)
        with (
            patch.object(controller.time, "time", return_value=wall),
            patch.object(controller.time, "monotonic", return_value=mono),
            patch.object(controller, "inspect_owned") as inspect,
        ):
            with self.assertRaises((TimeoutError, RuntimeError, ValueError)) as result:
                controller.monitor(self.root, self.lease, "fake", child, child, self.timer)
            inspect.assert_not_called()
        return str(result.exception)

    def test_wall_clock_backwards_cannot_extend_device_window(self):
        self.assertIn("Absolute 18", self.monitor(wall=500, mono=11080))

    def test_first_error_precedes_secondary_timeout(self):
        failure = self.output / "FAILED.json"
        failure.write_text('{"error":"original"}')
        self.assertIn("first failure", self.monitor(wall=1080, mono=11080))
        self.assertEqual(failure.read_text(), '{"error":"original"}')

    def test_no_pin_permission_after_expired_load(self):
        self.loading(entry=100, end=520)
        self.assertIn("Rank 0 load", self.monitor(wall=521, mono=10521))
        self.assertFalse((self.output / "pin.permit.json").exists())

    def test_success_records_anchor_bound_permission(self):
        self.loading(end=620)
        for rank in range(8):
            self.event("probe_complete", 650, rank)
        child = SimpleNamespace(poll=lambda: None)
        with (
            patch.object(controller.time, "time", return_value=650),
            patch.object(controller.time, "monotonic", return_value=10650),
            patch.object(controller, "inspect_owned", return_value={"State": {"Running": True}}),
        ):
            controller.monitor(self.root, self.lease, "fake", child, child, self.timer)
        permit = json.loads((self.output / "pin.permit.json").read_text())
        self.assertEqual(permit, {"lease_id": self.lease["lease_id"], "deadline_monotonic": 10800})

    def test_watchdog_uses_fixed_monotonic_deadline(self):
        with (
            patch.object(controller.sys, "argv", ["controller", "--watchdog", "a" * 64, "lease_test", "11080"]),
            patch.object(controller.time, "monotonic", return_value=10010),
            patch.object(controller.time, "time", side_effect=AssertionError("wall clock must not reset watchdog")),
            patch.object(controller.time, "sleep") as sleep,
            patch.object(controller, "stop_owned") as stop,
        ):
            self.assertEqual(controller.main(), 0)
            sleep.assert_called_once_with(1070)
            stop.assert_called_once_with("a" * 64, "lease_test")


class TestWorkerTiming(unittest.TestCase):
    def test_no_controller_permission_no_allocations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lease.json").write_text(json.dumps(lease_fixture()))
            for rank in range(1, 8):
                (root / f"model_loaded.rank-{rank}.json").write_text("{}")

            def fail_on_wait(_):
                (root / "FAILED.json").write_text('{"error":"synthetic controller failure"}')

            probe = StageAProbe(
                root / "lease.json", root, 0, 8, snapshot=snapshot_fixture, clock=lambda: 1000, sleep=fail_on_wait
            )
            with self.assertRaisesRegex(RuntimeError, "Peer or controller"):
                probe.run_steps(lambda _: self.fail("allocation without permission"))
            self.assertEqual(probe.claimed, [])

    def test_duplicate_entry_and_expired_pin_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lease.json").write_text(json.dumps(lease_fixture()))
            probe = StageAProbe(
                root / "lease.json", root, 0, 8, snapshot=snapshot_fixture, clock=lambda: 1000, monotonic=lambda: 10000
            )
            probe.mark("model_entered")
            with self.assertRaises(FileExistsError):
                probe.mark("model_entered")
            probe.pin_deadline = 10000
            with self.assertRaisesRegex(TimeoutError, "permission expired"):
                probe.allocate_step(0, 33554432, lambda _: self.fail("must not allocate"))
            self.assertEqual(probe.claimed, [])


if __name__ == "__main__":
    unittest.main()
