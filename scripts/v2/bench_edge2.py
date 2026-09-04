import os, shutil
import torch, torch_npu  # noqa
import vllm_ascend, vllm_ascend.vllm_ascend_C  # noqa
VA = os.path.dirname(vllm_ascend.__file__)
SRC = os.environ.get("V2_SRC", "/work/v2new")
for f in ("lora_ops_triton.py","lora_ops_triton_kernels.py","lora_cpp_launcher.cpp","lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so"):
    shutil.copy(os.path.join(SRC,f), os.path.join(VA,"lora",f))
os.environ["TRITON_LORA_CPP"]="1"; os.environ["TRITON_LORA_EXACT"]="1"
from vllm_ascend.lora import lora_ops_triton as T
DEV,DT,R,L="npu",torch.bfloat16,16,2
def shrink(H,tag):
    B=8; idx=torch.zeros(1,device=DEV,dtype=torch.int64); seq=torch.full((1,),B,device=DEV,dtype=torch.int64)
    torch.manual_seed(hash(("s",H))%2**31)
    x=torch.randn(B,H,device=DEV,dtype=DT)*0.05; w=torch.randn(L,1,R,H,device=DEV,dtype=DT)*0.05
    y=torch.zeros(B,R,device=DEV,dtype=torch.float32)
    torch.ops._C_ascend.sgmv_shrink(x,w.view(L,R,H),idx,seq,y,0.5); torch.npu.synchronize()
    ref=y.clone(); refmax=ref.abs().max().item()
    y.zero_(); T.sgmv_shrink(x,w,y,None,seq,idx,B,H,1,0.5); torch.npu.synchronize()
    d=(y-ref).abs().max().item(); be=torch.equal(y,ref); rel=d/max(refmax,1e-30)
    print("  shrink %-12s H=%-6d 16aln=%s refmax=%.3e bitexact=%s rel=%.3e %s"%(tag,H,H%16==0,refmax,be,rel,"OK" if (be or rel<1e-3) else "FAIL"),flush=True)
def expand(Ho,YHO,OFF,tag):
    B=8; idx=torch.zeros(1,device=DEV,dtype=torch.int64); seq=torch.full((1,),B,device=DEV,dtype=torch.int64)
    torch.manual_seed(hash(("e",Ho))%2**31)
    xe=torch.randn(B,R,device=DEV,dtype=torch.float32)*0.05; we=torch.randn(L,1,Ho,R,device=DEV,dtype=DT)*0.05
    y0=torch.randn(B,YHO,device=DEV,dtype=DT)*0.05
    y=y0.clone(); torch.ops._C_ascend.sgmv_expand(xe,we.view(L,Ho,R),idx,seq,y,OFF,Ho); torch.npu.synchronize()
    ref=y.clone(); refmax=ref.float().abs().max().item()
    y=y0.clone(); T.sgmv_expand_slice(xe,we,y,None,seq,idx,B,Ho,1,OFF,Ho,True); torch.npu.synchronize()
    d=(y.float()-ref.float()).abs().max().item(); be=torch.equal(y,ref); rel=d/max(refmax,1e-30)
    print("  expand %-12s Ho=%-6d 16aln=%s refmax=%.3e bitexact=%s rel=%.3e %s"%(tag,Ho,Ho%16==0,refmax,be,rel,"OK" if (be or rel<1e-3) else "FAIL"),flush=True)
print("=== 16-aligned but 64-indivisible (real-world edge) ===",flush=True)
shrink(1008,"h1008")   # 63*16, %64=48
shrink(2000,"h2000")   # 125*16, %64=16
shrink(1376,"h1376")   # 86*16, %64=32
expand(1008,1024,0,"ho1008")
expand(2000,2048,0,"ho2000")
expand(112,128,0,"ho112")  # 7*16, %32=16
print("=== NOT 16-aligned (AscendC itself unsupported) ===",flush=True)
shrink(1000,"h1000")
expand(1000,1024,0,"ho1000")
print("DONE",flush=True)
