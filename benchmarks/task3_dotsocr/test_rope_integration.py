"""Compare the installed wrapper with the unmodified source and alias contract."""
import importlib.util
import json
import torch
import torch_npu
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def main():
    torch.npu.set_device(0)
    init_device_properties_triton()
    spec = importlib.util.spec_from_file_location('original_rope', '/work/rope.base.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    torch.manual_seed(731)
    cases = [(t,qh,kh,128,128,torch.bfloat16,True,False)
             for t in [7,256] for qh,kh in [(1,1),(7,3),(12,2),(16,16)]]
    cases += [(32,12,2,d,r,dt,neo,contiguous) for d,r,dt,neo,contiguous in [
        (128,128,torch.bfloat16,True,True),
        (128,128,torch.float16,True,False),
        (128,64,torch.bfloat16,True,False),
        (128,128,torch.bfloat16,False,False),
        (64,64,torch.bfloat16,True,False)]]
    for t,qh,kh,d,r,dt,neo,contig in cases:
        x=torch.randn(t,qh+kh+1,d,dtype=dt,device='npu')
        y=x.clone()
        q,k=x[:,:qh],x[:,qh:qh+kh]
        q2,k2=y[:,:qh],y[:,qh:qh+kh]
        if contig:
            q,k,q2,k2=[z.contiguous() for z in (q,k,q2,k2)]
        cache=torch.randn(2048,r,dtype=dt,device='npu')
        pos=torch.randint(0,2048,(t,),device='npu')
        kwargs=dict(cos_sin_cache=cache,positions=pos,rope_dim=r,is_neox_style=neo)
        a=base.rope_forward_triton(q,k,**kwargs)
        b=rope_forward_triton(q2,k2,**kwargs)
        for first,second in zip(a,b):
            torch.testing.assert_close(first,second,rtol=0,atol=0)
        assert torch.equal(x,y), 'input mutation contract changed'
        assert (a[0].data_ptr()==q.data_ptr()) == (b[0].data_ptr()==q2.data_ptr())
        assert (a[1].data_ptr()==k.data_ptr()) == (b[1].data_ptr()==k2.data_ptr())
        print(json.dumps(dict(tokens=t,qh=qh,kh=kh,dim=d,rotary=r,dtype=str(dt),
                              neox=neo,contiguous=contig,bitwise=True,alias=True)),flush=True)
    print(json.dumps(dict(status='passed',cases=len(cases))),flush=True)


if __name__ == '__main__':
    main()
