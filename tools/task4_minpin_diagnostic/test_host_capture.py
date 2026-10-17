import json
import tempfile
import unittest
from pathlib import Path
from task4_host_capture import HostCapture

class FakeStorage:
    def data_ptr(self): return 4096
    def nbytes(self): return 8192

class FakeTensor:
    shape=(4,16)
    dtype='int8'
    device='metadata-only'
    def stride(self): return (16,1)
    def element_size(self): return 1
    def storage_offset(self): return 0
    def untyped_storage(self): return FakeStorage()

class CaptureTest(unittest.TestCase):
    def test_flush_before_each_allocation_and_preserve_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            capture=HostCapture(directory,stats_reader=lambda:{'allocated_bytes.current':123})
            calls=[]
            def zeros(shape,**kwargs):
                path=next(Path(directory).glob('*.jsonl'))
                record=json.loads(path.read_text().splitlines()[-1])
                self.assertEqual(record['phase'],'before_pin')
                self.assertEqual(record['details']['ordinal'],len(calls))
                calls.append((shape,kwargs))
                return object()
            result=capture.allocate_mirrors({'a':FakeTensor(),'b':FakeTensor()},3,True,zeros)
            self.assertEqual(list(result),['a','b'])
            self.assertEqual(calls, [((3,16),dict(dtype='int8',device='cpu',pin_memory=True))]*2)
            records=[json.loads(x) for x in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual([r['phase'] for r in records],['before_mirrors','before_pin','after_pin','before_pin','after_pin','mirrors_complete'])
            self.assertEqual(records[1]['details']['geometry']['rounded_bytes'],64)
            self.assertIn('rlimit_memlock',records[1])
            self.assertEqual(records[1]['host_allocator']['allocated_bytes.current'],123)

    def test_failure_preserves_exception_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            capture=HostCapture(directory,stats_reader=lambda:{})
            original=MemoryError('synthetic207001')
            calls=[]
            def zeros(*args,**kwargs):
                calls.append(1)
                if len(calls)==2: raise original
                return object()
            with self.assertRaises(MemoryError) as raised:
                capture.allocate_mirrors({'a':FakeTensor(),'b':FakeTensor(),'c':FakeTensor()},3,True,zeros)
            self.assertIs(raised.exception,original)
            self.assertEqual(len(calls),2)
            records=[json.loads(x) for x in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual(records[-1]['phase'],'pin_failed')
            self.assertEqual(records[-1]['details']['successful_regions'],1)

if __name__=='__main__': unittest.main()
