"""Check queued-reader codec and dependency/lifetime ordering on real streams."""

from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parent
script = (root / "verify_read_tiles.py").read_text().replace("/ws/read_tiles_20261014", "/ws/queued_read_20261014")
start = script.index("    assert len(records) == 15")
script = script[:start] + '''    assert len(records) == 15
    order_records = []
    for lengths, heads in [([20] * 16, 2), ([0, 33, 0, 257], 3)]:
        import math
        block = 128
        pages = math.ceil(max(lengths) / block)
        cpu_cache = torch.randint(0, 256, (pages * len(lengths) + 2, block, heads, 66), dtype=torch.uint8)
        cpu_norm = (torch.rand(cpu_cache.shape[:-1]) + 0.01).to(torch.bfloat16)
        cpu_cache[..., 64:] = cpu_norm.unsqueeze(-1).view(torch.uint8)
        cpu_table = torch.arange(pages * len(lengths), dtype=torch.int32).view(len(lengths), pages)
        expected_cpu = reference._dequant_dense_reference(None, cpu_cache.view(torch.bfloat16), cpu_table, lengths)
        for use_other_stream in [False, True]:
            stream = torch.npu.Stream() if use_other_stream else torch.npu.current_stream()
            with torch.npu.stream(stream):
                target = cpu_cache.npu()
                table_target = cpu_table.npu()
                cache = target.clone()
                table = table_target.clone()
                lookup = reference._get_byte_lut(cache.device)
                metadata = SimpleNamespace()
                # Prime logical jobs outside the dependency test; otherwise
                # their first H2D allocation can hide a missing queue edge.
                reader(cache, table, lengths, lookup, metadata)
                torch.npu.synchronize()
                for repeat in range(5):
                    cache.fill_(repeat + 1)
                    cache.copy_(target)
                    table.zero_()
                    table.copy_(table_target)
                    # Release temporary inputs immediately. The queued handler
                    # must retain them and execute after their GPU producers,
                    # before the standard PyTorch clone consumer.
                    consumed = reader(cache.clone(), table.clone(), lengths, lookup.clone(), metadata).clone()
                    actual_cpu = consumed.cpu()
                    mismatch = (actual_cpu.view(torch.int16) != expected_cpu.view(torch.int16)).sum().item()
                    record = {"variant": "async_producer_reader_consumer", "lengths": lengths,
                              "heads": heads, "nondefault_stream": use_other_stream,
                              "repeat": repeat, "mismatched_bf16_bits": mismatch}
                    order_records.append(record)
                    print(json.dumps(record), flush=True)
                    assert mismatch == 0, record
    assert len(order_records) == 20
''' + script[start + len("    assert len(records) == 15\n"):]
script = script.replace('"passed": len(records),', '"passed": len(records),\n                "async_passed": len(order_records),\n                "async_records": order_records,')
(root / "verify_queued_read.py").write_text(script)
compile(script, "verify_queued_read.py", "exec")
remote = "/data02/tq-opt-20261011/queued_read_20261014/gate-attempt1"
mapped = "/ws/queued_read_20261014/gate-attempt1"
command = ["docker", "exec", "-e", "ASCEND_RT_VISIBLE_DEVICES=2,5", "-e", "VLLM_ASCEND_KVTQ_BITS=4", "tq-opt-20261011-model", "bash", "-lc",
           "source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && "
           f"export PYTHONPATH=/ws/queued_read_20261014:/ws/source/vllm:/ws/source/vllm-ascend:$PYTHONPATH && "
           f"exec /ws/.venv/bin/python -u {mapped}/verify_queued_read.py --output-dir {mapped}"]
host = f'''import pathlib,json,subprocess
root=pathlib.Path({remote!r})
assert (root.parent/'build.exit').read_text().strip()=='0'
npu=subprocess.check_output(['npu-smi','info'],text=True)
assert 'No running processes found in NPU 2' in npu and 'No running processes found in NPU 5' in npu
(root/'resource-before.log').write_text(npu)
with (root/'stdout.log').open('w') as output:
 result=subprocess.run(json.loads({json.dumps(command)!r}),stdout=output,stderr=subprocess.STDOUT,timeout=1800)
(root/'gate.exit').write_text(str(result.returncode)+'\\n')
result.check_returncode()
report=json.loads((root/'verify-native-read.json').read_text())
assert report['passed']==15 and report['async_passed']==20
'''
deploy = f'''import pathlib,subprocess,json
root=pathlib.Path({remote!r})
assert not root.exists()
root.mkdir()
(root/'verify_queued_read.py').write_text({script!r})
(root/'host.py').write_text({host!r})
(root/'command.json').write_text({json.dumps(command)!r})
with (root/'host.log').open('w') as output:
 p=subprocess.Popen(['python3','-I','-u',str(root/'host.py')],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'host.pid').write_text(str(p.pid))
print(json.dumps({{'pid':p.pid,'root':str(root)}}))
'''
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@112.29.145.3", "python3 -I -"], input=deploy.encode(), capture_output=True)
(root / "queued-gate-deploy.log").write_bytes(result.stdout + result.stderr)
result.check_returncode();print(result.stdout.decode())
