"""Synthetic bitwise, input preservation, graph replay and latency checks."""
import json
import statistics
import torch
import torch_npu
from rope_copy import ModelNew
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def timing(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(20):
            fn()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 200)
    return times


def main():
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(42)
    model = ModelNew()
    cache = torch.randn(16384,128,dtype=torch.bfloat16,device='npu')
    rows = []
    for t in [1,7,32,255,256,257,512,2048]:
        for seed in [1,42]:
            torch.manual_seed(seed)
            qkv = torch.randn(t,16,128,dtype=torch.bfloat16,device='npu')
            original = qkv.clone()
            q,k = qkv[:,:12],qkv[:,12:14]
            pos = torch.randint(0,16384,(t,),device='npu')
            base = rope_forward_triton(q.clone(), k.clone(),cos_sin_cache=cache,
                                       positions=pos,rope_dim=128,is_neox_style=True)
            candidate = model(q,k,cache,pos)
            for a,b in zip(base,candidate):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert torch.equal(original,qkv)
            for _ in range(4):
                again = model(q,k,cache,pos)
                assert all(torch.equal(a,b) for a,b in zip(candidate,again))
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = model(q,k,cache,pos)
            qkv.normal_()
            pos.copy_(torch.randint(0,16384,(t,),device='npu'))
            graph.replay()
            torch.npu.synchronize()
            base = rope_forward_triton(q.clone(),k.clone(),cos_sin_cache=cache,
                                       positions=pos,rope_dim=128,is_neox_style=True)
            assert all(torch.equal(a,b) for a,b in zip(base,result))
            rows.append(dict(tokens=t,seed=seed,bitwise=True,graph_dynamic_inputs=True))
        if t in [256,2048]:
            def baseline():
                return rope_forward_triton(q,k,cos_sin_cache=cache,positions=pos,
                                           rope_dim=128,is_neox_style=True)
            def candidate():
                return model(q,k,cache,pos)
            a=timing(baseline); b=timing(candidate); a2=timing(baseline)
            print(json.dumps(dict(tokens=t,baseline_us=a,candidate_us=b,
                                  baseline_return_us=a2,
                                  speedup=statistics.median(a+a2)/statistics.median(b))),flush=True)
    print(json.dumps(dict(status='passed',checks=rows)),flush=True)


if __name__ == '__main__':
    main()
