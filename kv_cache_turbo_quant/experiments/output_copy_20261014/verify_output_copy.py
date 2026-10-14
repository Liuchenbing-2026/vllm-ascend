"""Check the actual output-write blocks, caller state and asynchronous order."""

import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def writer(path, name):
    tree = ast.parse(path.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    at = next(i for i, x in enumerate(function.body) if isinstance(x, ast.Assign)
              and isinstance(x.targets[0], ast.Name) and x.targets[0].id == 'out')
    template = ast.parse('def write(attn_out, rot, output, self, batch, n_prefill, num_decode_tokens, num_tokens):\n    pass').body[0]
    template.body = function.body[at:at + 2] + [ast.Return(value=ast.Name(id='output', ctx=ast.Load()))]
    scope = {'torch': torch, '_HEAD_DIM': 128}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[template], type_ignores=[])), str(path), 'exec'), scope)
    return scope['write']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'npu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'npu':
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)
    torch.manual_seed(20261014)
    records = []
    times = []
    for name in ['_tq_decode', '_prefill_from_cache']:
        old, new = [writer(p, name) for p in [args.baseline, args.candidate]]
        for rows in [1, 16, 60, 320, 2000, 4096]:
            begin = 0 if name == '_tq_decode' else 3
            end = begin + rows
            for strided in [False, True]:
                x = torch.randn(rows, 16, 128, dtype=torch.float32, device=args.device)
                rot = torch.randn(128, 128, dtype=torch.float32, device=args.device) / 128 ** .5
                initial = torch.randn(end + 5, 16, 128, device=args.device).to(torch.bfloat16)
                old_out = initial.clone()
                new_out = initial.clone()
                if strided:
                    old_out = old_out.transpose(1, 2).contiguous().transpose(1, 2)
                    new_out = new_out.transpose(1, 2).contiguous().transpose(1, 2)
                with torch.inference_mode():
                    context = SimpleNamespace(num_heads=16)
                    params = (x, rot, context, rows, rows, begin, end)
                    old(x, rot, old_out, *params[2:])
                    new(x, rot, new_out, *params[2:])
                if args.device == 'npu':
                    torch.npu.synchronize()
                mismatch = (old_out.cpu().view(torch.int16) != new_out.cpu().view(torch.int16)).sum().item()
                tail_changed = (new_out[end:].cpu().view(torch.int16) != initial[end:].cpu().view(torch.int16)).sum().item()
                prefix_changed = (new_out[:begin].cpu().view(torch.int16) != initial[:begin].cpu().view(torch.int16)).sum().item()
                record = dict(function=name, rows=rows, strided=strided, mismatch=int(mismatch), tail_changed=int(tail_changed), prefix_changed=int(prefix_changed))
                assert mismatch == tail_changed == prefix_changed == 0, record
                records.append(record)
                if args.device == 'npu' and not strided and rows in [16, 60, 4096]:
                    for repeat in range(3):
                        for mode, fn, target in [('old', old, old_out), ('new', new, new_out)]:
                            with torch.inference_mode():
                                for _ in range(10):
                                    fn(x, rot, target, *params[2:])
                                torch.npu.synchronize()
                                start, stop = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                                start.record()
                                for _ in range(100):
                                    fn(x, rot, target, *params[2:])
                                stop.record()
                                stop.synchronize()
                                times.append(dict(function=name, rows=rows, mode=mode, repeat=repeat,
                                                  event_mean_us=start.elapsed_time(stop) * 10))
    async_passed = 0
    if args.device == 'npu':
        old, new = [writer(p, '_tq_decode') for p in [args.baseline, args.candidate]]
        stream = torch.npu.Stream()
        for i in range(20):
            with torch.npu.stream(stream), torch.inference_mode():
                x = torch.randn(16, 16, 128, device='npu').add_(i)
                rot = torch.eye(128, device='npu')
                a = torch.zeros(16, 16, 128, dtype=torch.bfloat16, device='npu')
                b = torch.zeros_like(a)
                for fn, out in [(old, a), (new, b)]:
                    fn(x, rot, out, SimpleNamespace(num_heads=16), 16, 16, 0, 16)
                a, b = a.clone(), b.clone()
            stream.synchronize()
            assert torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))
            async_passed += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(device=args.device, passed=len(records), cases=records, async_passed=async_passed,
                  timings=times, timing_scope='actual rotation+caller-buffer write; device events include host submission gaps; not whole-model gain')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'timings']}))


if __name__ == '__main__':
    main()
