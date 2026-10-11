"""CPU unit checks for metadata reuse and invalidation; no NPU claims."""
import argparse
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class ReadPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--integration-dir', required=True)
        args, remaining = parser.parse_known_args()
        sys.argv = [sys.argv[0], *remaining]
        cls.created = []

        def tensor(values, **kwargs):
            value = (tuple(values), kwargs)
            cls.created.append(value)
            return value

        fake_torch = types.SimpleNamespace(tensor=tensor, int32='int32')
        spec = importlib.util.spec_from_file_location('tested_kvtq_store',
                    Path(args.integration_dir) / 'kvtq_store.py')
        cls.store = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': fake_torch}):
            spec.loader.exec_module(cls.store)

    def setUp(self):
        self.created.clear()

    def test_reuses_k_v_and_layer_metadata(self):
        metadata = types.SimpleNamespace()
        first = self.store._read_plan(metadata, 'decode', 'npu:0', [20, 21])
        second = self.store._read_plan(metadata, 'decode', 'npu:0', [20, 21])
        self.assertIs(first, second)
        self.assertEqual(len(self.created), 1)
        self.assertEqual(first[1:], ([0, 20, 41], 41, 21))

    def test_changed_lengths_and_device_invalidate(self):
        metadata = types.SimpleNamespace()
        lengths = [128, 129]
        first = self.store._read_plan(metadata, 'decode', 'npu:0', lengths)
        lengths[1] += 1
        second = self.store._read_plan(metadata, 'decode', 'npu:0', lengths)
        third = self.store._read_plan(metadata, 'decode', 'npu:1', lengths)
        self.assertIsNot(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(second[1], [0, 128, 258])
        self.assertEqual(third[0][1]['device'], 'npu:1')

    def test_mixed_batch_segments_and_bounded_retention(self):
        metadata = types.SimpleNamespace()
        decode = self.store._read_plan(metadata, 'decode', 'npu:0', [7])
        self.store._read_plan(metadata, 'prefill', 'npu:0', [256])
        self.assertIs(decode, self.store._read_plan(metadata, 'decode', 'npu:0', [7]))
        for length in range(30):
            self.store._read_plan(metadata, 'decode', 'npu:0', [length])
        self.assertEqual(set(metadata._kvtq_read_plans), {'decode', 'prefill'})
        self.assertEqual(len(metadata._kvtq_read_plans), 2)

    def test_empty_batch_and_standalone_read(self):
        plan = self.store._read_plan(None, 'decode', 'npu:0', [])
        self.assertEqual(plan[1:], ([0], 0, 0))
        self.assertIsNot(plan, self.store._read_plan(None, 'decode', 'npu:0', []))


if __name__ == '__main__':
    # unittest parses argv before setUpClass, so preserve only its own flags
    # until the test class has consumed the integration path.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
