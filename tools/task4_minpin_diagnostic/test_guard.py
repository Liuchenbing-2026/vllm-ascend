import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from task4_host_capture import reserve_quota,validate_minimal_geometry,before_model_guard,load_probe_lease

class Storage:
    def data_ptr(self): return 0
    def nbytes(self): return 2**30
class Tensor:
    dtype='int8'; device='metadata'
    def __init__(self,b): self.shape=(4096,b)
    def stride(self): return (self.shape[1],1)
    def element_size(self): return 1
    def storage_offset(self): return 0
    def untyped_storage(self): return Storage()

class TestGuard(unittest.TestCase):
    def test_valid_geometry(self):
        caches={str(i):Tensor(16640 if i<21 else 131072) for i in range(46)}
        self.assertEqual(validate_minimal_geometry(caches,1,0,8,4*2**20),(3626240,3964928))

    def test_capacity_counterexamples(self):
        caches={str(i):Tensor(16640 if i<21 else 131072) for i in range(46)}
        cases={'original8141':(caches,8141,0,8,4*2**20),'original28GiB':(caches,1,0,8,28*2**30),'bad_rank':(caches,1,8,8,4*2**20),'bad_world':(caches,1,0,9,4*2**20),'missing_regions':({},1,0,8,4*2**20),'over_requested':({str(i):Tensor(131072) for i in range(46)},1,0,8,4*2**20),'rounding_excess':({str(i):Tensor(65537) for i in range(46)},1,0,8,4*2**20)}
        for name,args in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError): validate_minimal_geometry(*args)

    def test_global_quota_duplicate_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            for rank in range(8):reserve_quota(d,rank,4*2**20,4*2**20)
            for args in [(0,1,1),(8,1,1),(-1,1,1),(0,-1,1),(0,5*2**20,5*2**20)]:
                with self.subTest(args=args):
                    with self.assertRaises(ValueError):reserve_quota(d,*args)

    def test_old_config_fails_before_loading(self):
        c=SimpleNamespace(parallel_config=SimpleNamespace(world_size=8),cache_config=SimpleNamespace(kv_offloading_size=224))
        with self.assertRaises(ValueError):before_model_guard(c)

    def test_lease_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'lease.json';now=time.time()
            base=dict(status='APPROVED',lease_id='CPU_TEST_ONLY',start_epoch=now-10,end_epoch=now+1190,physical_devices=list(range(8)),rank_limit_bytes=4*2**20,total_limit_bytes=32*2**20,output_dir='/ws/task4_minpin/attempts/cpu_fixture')
            path.write_text(json.dumps(base));load_probe_lease(path,now)
            cases=[dict(status='PENDING'),dict(lease_id='lease_aa1a6ca9-0c57-4c7c-ace4-1b7d6d027344'),dict(end_epoch=now+100),dict(start_epoch=now+60,end_epoch=now+1260),dict(physical_devices=[0,1]),dict(rank_limit_bytes=28*2**30),dict(output_dir='/ws/old_attempt'),dict(start_epoch=now-1200,end_epoch=now+60)]
            for changes in cases:
                with self.subTest(changes=changes):
                    path.write_text(json.dumps(dict(base,**changes)))
                    with self.assertRaises(ValueError):load_probe_lease(path,now)
            path.unlink()
            with self.assertRaises(FileNotFoundError):load_probe_lease(path,now)

if __name__=='__main__':unittest.main()
