# SPDX-License-Identifier: Apache-2.0
"""CPU-only counterexamples. Fake allocators never allocate the requested bytes."""

import ast
import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import controller
from task4_bounded_pin import StageAProbe, check_headroom, validate_config, validate_lease


def lease_fixture():
    return {
        "status": "APPROVED",
        "scope": "task4-rc2-stage-a-full-model-pin",
        "lease_id": "lease_CPU_FIXTURE_ONLY",
        "start_epoch": 900,
        "end_epoch": 2400,
        "physical_devices": list(range(8)),
        "allocator_ranks": [0],
        "steps_bytes": [33554432, 134217728, 268435456, 1073741824],
        "rank_limit_bytes": 1509949440,
        "total_limit_bytes": 1509949440,
        "vllm_sha": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "ascend_base_sha": "8fde6d4e335fe300e2b949aab0e31659cce0811d",
        "ascend_sha": "a" * 40,
    }


def snapshot_fixture():
    return {"meminfo": "MemAvailable: 134217728 kB\n", "cgroups": {}, "allocator": {"active.freed": -2}}


class TestBoundedPin(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "lease.json"
        self.lease = lease_fixture()
        self.path.write_text(json.dumps(self.lease))

    def probe(self, rank=0, **kwargs):
        return StageAProbe(self.path, self.root, rank, 8, snapshot=snapshot_fixture, clock=lambda: 1000, **kwargs)

    def test_all_eight_ranks_only_rank_zero_allocates_and_retains_four_buffers(self):
        probes = [self.probe(rank) for rank in range(8)]
        calls, errors = [], []

        def worker(probe):
            try:

                def allocate(size):
                    token = object()
                    calls.append((probe.rank, size, token))
                    return token

                probe.run_steps(allocate)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(probe,), daemon=True) for probe in probes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            [(rank, size) for rank, size, _ in calls], [(0, 33554432), (0, 134217728), (0, 268435456), (0, 1073741824)]
        )
        self.assertEqual(sum(size for _, size, _ in calls), 1509949440)
        self.assertEqual(probes[0].retained, [token for _, _, token in calls])
        self.assertTrue(all(not probe.retained for probe in probes[1:]))
        for rank in range(8):
            records = [json.loads(line) for line in (self.root / f"rank-{rank}.jsonl").read_text().splitlines()]
            self.assertEqual(sum(record["phase"] == "before_pin" for record in records), 4)
            self.assertEqual(sum(record["phase"] == "after_pin" for record in records), 4)
            self.assertEqual(records[-1]["phase"], "probe_complete")
            self.assertEqual(records[-1]["allocator"]["active.freed"], -2)

    def test_lease_counterexamples(self):
        cases = [
            dict(status="PENDING"),
            dict(scope="task4-minpin"),
            dict(allocator_ranks=list(range(8))),
            dict(total_limit_bytes=224 << 30),
            dict(rank_limit_bytes=28 << 30),
            dict(steps_bytes=[224 << 30]),
            dict(physical_devices=[0]),
            dict(ascend_sha="HEAD"),
            dict(vllm_sha="other"),
            dict(start_epoch=True),
            dict(end_epoch=2500),
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_lease(dict(self.lease, **changes), 1000)
        for now in (899, 1980, 2400):
            with self.subTest(now=now), self.assertRaises(ValueError):
                validate_lease(self.lease, now)

    def test_duplicate_rank_and_no_nonzero_rank_allocation(self):
        self.probe(0)
        with self.assertRaises(FileExistsError):
            self.probe(0)
        probe = self.probe(1)
        with self.assertRaises(ValueError):
            probe.allocate_step(0, 33554432, lambda _: self.fail("allocator must not run"))

    def test_invalid_or_repeated_steps_never_allocate(self):
        probe = self.probe()
        for index, size in [(0, 224 << 30), (0, 33554433), (1, 134217728), (-1, 1), (True, 33554432)]:
            with self.subTest(index=index, size=size), self.assertRaises(ValueError):
                probe.allocate_step(index, size, lambda _: self.fail("allocator must not run"))
        probe.allocate_step(0, 33554432, lambda _: object())
        with self.assertRaises(ValueError):
            probe.allocate_step(0, 33554432, lambda _: self.fail("duplicate allocation"))

    def test_failure_is_preserved_and_next_step_refused(self):
        probe = self.probe()

        def failing(_):
            raise RuntimeError("synthetic driver 207001")

        with self.assertRaisesRegex(RuntimeError, "207001"):
            probe.allocate_step(0, 33554432, failing)
        first = (self.root / "FAILED.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Peer or controller"):
            probe.allocate_step(1, 134217728, lambda _: self.fail("no allocations after first error"))
        probe.fail(RuntimeError("later error"))
        self.assertEqual(first, (self.root / "FAILED.json").read_bytes())

    def test_changed_lease_and_missing_peer_timeout(self):
        probe = self.probe()
        self.path.write_text(json.dumps(dict(self.lease, total_limit_bytes=224 << 30)))
        with self.assertRaisesRegex(RuntimeError, "Lease changed"):
            probe.check()
        self.path.write_bytes(probe.lease_bytes)
        probe.clock = lambda: 1980
        with self.assertRaises(ValueError):
            probe.wait_all("missing")

    def test_host_and_cgroup_headroom_and_missing_sample(self):
        check_headroom(snapshot_fixture(), 1509949440)
        for sample in [
            {"meminfo": "MemAvailable: 1024 kB\n", "cgroups": {}},
            {"meminfo": "", "cgroups": {}},
            dict(snapshot_fixture(), cgroups={"/x/memory.max": "1024", "/x/memory.current": "1000"}),
        ]:
            with self.subTest(sample=sample), self.assertRaises((RuntimeError, KeyError)):
                check_headroom(sample, 1509949440)

    def test_original_pool_dummy_or_wrong_parallel_rejected(self):
        base = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=8, pipeline_parallel_size=1, data_parallel_size=1),
            load_config=SimpleNamespace(load_format="auto"),
            cache_config=SimpleNamespace(kv_offloading_size=None),
            kv_transfer_config=None,
            use_v2_model_runner=True,
        )
        validate_config(base, False)
        for field, name, value in [
            ("cache_config", "kv_offloading_size", 224),
            ("load_config", "load_format", "dummy"),
            ("parallel_config", "tensor_parallel_size", 4),
        ]:
            changed = copy.deepcopy(base)
            setattr(getattr(changed, field), name, value)
            with self.assertRaises(ValueError):
                validate_config(changed, False)
        with self.assertRaises(ValueError):
            validate_config(base, True)
        base.kv_transfer_config = object()
        with self.assertRaises(ValueError):
            validate_config(base, False)


class TestController(unittest.TestCase):
    def test_no_docker_call_without_approved_lease(self):
        with patch.object(controller, "command") as command:
            with self.assertRaises(ValueError):
                controller.execute(Path("/no-files-read"), dict(lease_fixture(), status="PENDING"))
            command.assert_not_called()

    def test_stop_only_owned_id(self):
        state = {"Id": "ours", "Config": {"Labels": {"task4.pin.lease": "wrong"}}, "State": {"Running": True}}
        with patch.object(controller, "command", return_value=json.dumps([state])) as command:
            with self.assertRaises(ValueError):
                controller.stop_owned("ours", "expected")
            self.assertEqual(command.call_count, 1)

    def test_idle_snapshot_requires_all_eight(self):
        controller.validate_idle("\n".join(f"No running processes found in NPU {rank}" for rank in range(8)))
        with self.assertRaises(RuntimeError):
            controller.validate_idle("No running processes found in NPU 0")

    def test_reap_direct_child_no_future_action(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "future"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time,pathlib,sys;time.sleep(0.3);pathlib.Path(sys.argv[1]).touch()",
                    str(marker),
                ]
            )
            controller.reap(child)
            self.assertIsNotNone(child.returncode)
            time.sleep(0.4)
            self.assertFalse(marker.exists())


class TestRunnerIntegration(unittest.TestCase):
    def runner(self, probe, calls):
        # Execute the actual edited method with only its unrelated dependencies
        # isolated. This proves guard/load/probe order without importing torch.
        path = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/model_runner.py"
        source = ast.parse(path.read_text())
        model_class = next(
            node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
        )
        method = next(
            node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "load_model"
        )
        model_class.bases = [ast.Name(id="FakeLoader", ctx=ast.Load())]
        model_class.body = [method]
        model_class.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[model_class], type_ignores=[]))

        class FakeLoader:
            def load_model(self, *args, **kwargs):
                calls.append("complete_model_load")

        class FakeGuard:
            @staticmethod
            def before_model(config, dummy):
                calls.append("guard")
                if config == "reject":
                    raise ValueError("unapproved")
                return probe

        namespace = {"FakeLoader": FakeLoader, "StageAProbe": FakeGuard, "get_pd_dspark_aux_layer_ids": lambda _: []}
        exec(compile(module, str(path), "exec"), namespace)
        runner = namespace["NPUModelRunner"]()
        runner.vllm_config = "accept"
        return runner

    def test_rejected_lease_cannot_load_model(self):
        calls = []
        runner = self.runner(None, calls)
        runner.vllm_config = "reject"
        with self.assertRaisesRegex(ValueError, "unapproved"):
            runner.load_model()
        self.assertEqual(calls, ["guard"])

    def test_unexpected_probe_return_cannot_reach_kv_initialization(self):
        calls, errors = [], []
        probe = SimpleNamespace(after_model=lambda: calls.append("after_complete_model"), fail=errors.append)
        runner = self.runner(probe, calls)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly returned"):
            runner.load_model()
        self.assertEqual(calls, ["guard", "complete_model_load", "after_complete_model"])
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
