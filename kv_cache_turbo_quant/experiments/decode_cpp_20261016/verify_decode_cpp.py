"""Compare the unchanged Python decode math with C++ orchestration on real NPU."""
from pathlib import Path
from itertools import accumulate
import json, time
import torch, torch_npu

root=Path('/ws/decode_cpp_20261016')
torch.ops.load_library(str(root/'torch-extensions/tq_decode/tq_decode.so'))
device='npu:0'
torch.manual_seed(20261016)
rot=torch.linalg.qr(torch.randn(128,128,dtype=torch.float64))[0].float().to(device)
rot_t=rot.T.contiguous()

def reference(q,k,v,ql,kl,out):
    rq=torch.matmul(q.float(),rot_t).to(torch.bfloat16)
    a,_=torch_npu.npu_fused_infer_attention_score(query=rq,key=k,value=v,
        block_table=None,input_layout='TND',block_size=128,actual_seq_lengths=ql,
        actual_seq_lengths_kv=kl,num_heads=16,num_key_value_heads=2,
        scale=128**-.5,sparse_mode=0)
    out.copy_(torch.matmul(a.view(q.shape[0],16,128).float(),rot))

def candidate(q,k,v,ql,kl,out):
    torch.ops.turboquant.tq_decode(q,k,v,rot,rot_t,ql,kl,16,2,128,128**-.5,out)

records=[]
for inference in [False,True]:
 with torch.inference_mode(inference):
  for index,lengths in enumerate([[1],[1,17,31],[20]*16,[2000]*60,[2199-i for i in range(60)]]):
    b=len(lengths);q=torch.randn(b,16,256,device=device,dtype=torch.bfloat16)[:,:,::2]
    k=torch.randn(sum(lengths),2,128,device=device,dtype=torch.bfloat16);v=torch.randn_like(k)
    ql=list(range(1,b+1));kl=list(accumulate(lengths))
    expected=torch.empty_like(q)
    buffer=torch.full((b+2,16,128,2),123.,device=device,dtype=torch.bfloat16)
    actual=buffer[1:b+1,:,:,0]
    reference(q,k,v,ql,kl,expected);candidate(q,k,v,ql,kl,actual);torch.npu.synchronize()
    assert torch.equal(expected,actual),(inference,index,float((expected-actual).abs().max()))
    assert torch.all(buffer[0]==123) and torch.all(buffer[-1]==123) and torch.all(buffer[:,:,:,1]==123)
    timing={}
    if not inference and index in [2,3]:
     for name,fn in [('python',reference),('cpp',candidate)]:
      for _ in range(5):fn(q,k,v,ql,kl,actual)
      torch.npu.synchronize();begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
      started=time.perf_counter();begin.record()
      for _ in range(30):fn(q,k,v,ql,kl,actual)
      end.record();end.synchronize()
      timing[name]={'device_plus_submission_us':begin.elapsed_time(end)*1000/30,'wall_us':(time.perf_counter()-started)*1e6/30,'repeats':30}
    records.append({'inference':inference,'batch':b,'kv_tokens':sum(lengths),'bitwise_equal':True,'noncontiguous_query_output':True,'sentinels_preserved':True,'timing_screen_only':timing})
    print(json.dumps(records[-1]),flush=True)

stream=torch.npu.Stream();pending=[]
for i in range(20):
 with torch.npu.stream(stream):
  q=torch.randn(3,16,128,device=device,dtype=torch.bfloat16)
  k=torch.randn(39,2,128,device=device,dtype=torch.bfloat16);v=torch.randn_like(k)
  a=torch.empty_like(q);e=torch.empty_like(q)
  candidate(q,k,v,[1,2,3],[1,8,39],a);reference(q,k,v,[1,2,3],[1,8,39],e)
  pending.append((a,e))
stream.synchronize()
assert all(torch.equal(a,e) for a,e in pending)
result={'functional_cases':records,'async_cases':20,'async_bitwise_equal':True,'scope':'rotation/FIA/output only; original BF16 FIA reference, FP32 rotation, zero tolerance; no packed cache/write/model coverage claimed'}
(root/'verify-decode.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS 10 functional and 20 nondefault-stream cases',flush=True)
