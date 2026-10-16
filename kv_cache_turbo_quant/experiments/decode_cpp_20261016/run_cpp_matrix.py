"""Same-run BF16, prior Byte TQ and C++-orchestrated Byte TQ comparison."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,signal,subprocess
from run_decode_profiles import Runner as Base
from run_8k1 import ROOT

class Runner(Base):
 def run(self):
  self.run_id='write_ab_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  self.results=ROOT/'results'/self.run_id;self.logs=ROOT/'logs'/self.run_id
  self.results.mkdir(parents=True);self.logs.mkdir(parents=True);self.prerequisites()
  gate_root=Path('/ws/decode_cpp_20261016')
  assert (gate_root/'verify.exit').read_text().strip()=='0'
  gate=json.loads((gate_root/'verify-decode.json').read_text())
  assert len(gate['functional_cases'])==10 and all(x['bitwise_equal'] and x['sentinels_preserved'] for x in gate['functional_cases'])
  assert gate['async_cases']==20 and gate['async_bitwise_equal']
  for name,digest in json.loads((ROOT/'writer-manifest.json').read_text()).items():
   assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
  expected=json.loads((ROOT/'decode-binary.json').read_text())
  assert hashlib.sha256((gate_root/'torch-extensions/tq_decode/tq_decode.so').read_bytes()).hexdigest()==expected['sha256']
  scenes=[('20_20_c16',20,20,16),('2k200_c60',2000,200,60)]
  parameters=dict(modes=['bf16','candidate','optimized'],scenarios=scenes,max_num_seqs=512,
   max_num_batched_tokens=4096,max_model_len=8192,kv_cache_bytes_per_card=25769803776,
   tp=2,physical_cards=[2,5],warmups=1,formal_repeats=3,profiler_active_during_formal=False,
   optimized_reference='newByte abbe29ef; same-run control',source=json.loads((ROOT/'decode-delivery.json').read_text()))
  (self.logs/'run_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
  for mode in ['bf16','candidate','optimized']:
   self.start_server(mode,0,profiling=False)
   self.run_client(mode,1,8000,1,'smoke_8k_'+mode,timeout=300)
   cmd=[self.python,str(ROOT/'scripts/check_model_output.py'),'--mode',mode,'--output',str(self.results/('precision_model_'+mode+'.json'))]
   with (self.logs/('precision_model_'+mode+'.log')).open('w') as f:
    subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
   for scene,inp,out,c in scenes:
    self.run_client(mode,c,inp,out,scene+'_'+mode+'_warmup')
    for repeat in [1,2,3]:self.run_client(mode,c,inp,out,scene+'_'+mode+'_r'+str(repeat))
   self.stop_server()
  a=json.loads((self.results/'precision_model_optimized.json').read_text())['records']
  b=json.loads((self.results/'precision_model_candidate.json').read_text())['records']
  assert len(a)==len(b)==4
  for old,new in zip(a,b):
   assert old['payload_sha256']==new['payload_sha256']
   assert old['response']['choices'][0]['logprobs']['tokens']==new['response']['choices'][0]['logprobs']['tokens']
  (ROOT/'model-token-equality.json').write_text(json.dumps({'passed_samples':4,'equal_token_ids':440,'scope':'Byte TQ Python versus Byte TQ C++; not BF16 semantic quality'})+'\n')
  self.status(stage='cpp_matrix_finished',result_dir=str(self.results))

if __name__=='__main__':
 (ROOT/'controller.pid').write_text(str(os.getpid())+'\n');r=Runner()
 signal.signal(signal.SIGTERM,lambda sig,frame:(_ for _ in ()).throw(KeyboardInterrupt('signal '+str(sig))))
 try:r.run()
 except BaseException as e:
  r.status(stage='controller_failed',error=repr(e));r.issue('cpp-matrix',repr(e),r.logs,'preserve every sample and failure');(ROOT/'background.exit').write_text('1\n');raise
 else:(ROOT/'background.exit').write_text('0\n')
 finally:r.stop_server()
