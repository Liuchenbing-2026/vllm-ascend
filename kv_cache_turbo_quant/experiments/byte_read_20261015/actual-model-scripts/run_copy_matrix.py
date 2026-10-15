import json,os,signal,subprocess
from datetime import datetime,timezone
from pathlib import Path
from run_decode_profiles import Runner as ProfileRunner
from run_8k1 import ROOT

class Runner(ProfileRunner):
    def run(self):
        self.run_id='write_ab_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.results=ROOT/'results'/self.run_id;self.logs=ROOT/'logs'/self.run_id
        self.results.mkdir(parents=True);self.logs.mkdir(parents=True)
        self.prerequisites()
        gate=json.loads(Path('/ws/output_copy_gate_20261014/npu.json').read_text())
        assert gate['passed']==24 and gate['async_passed']==20
        assert json.loads(Path('/ws/output_copy_gate_20261014/cpu.json').read_text())['passed']==24
        import hashlib
        expected=json.loads((ROOT/'writer-manifest.json').read_text())
        for name,digest in expected.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        gate_root=Path('/ws/byte_read_attempt2_20261015')
        g=json.loads((gate_root/'functional/verify-native-read.json').read_text())
        assert g['passed']==15 and g['async_passed']==20 and g['inference_passed']==4 and len(g['lookup_records'])==2
        assert json.loads((gate_root/'memory-review.json').read_text())['no_access_error_calls']==40
        assert Path('/ws/byte_read_micro_20261015/host.exit').read_text().strip()=='0'
        for name,entry in json.loads((ROOT/'byte-native-binaries.json').read_text()).items():
            assert hashlib.sha256((Path('/ws/byte_read_20261015')/name).read_bytes()).hexdigest()==entry['sha256'],name
        scenes=[('20_20_c16',20,20,16),('2k200_c60',2000,200,60)]
        parameters={'modes':['bf16','candidate','optimized'],'scenarios':scenes,'max_num_seqs':512,
            'max_num_batched_tokens':4096,'max_model_len':8192,'kv_cache_bytes_per_card':25769803776,
            'tp':2,'physical_cards':[2,5],'warmups':1,'formal_repeats':3,'profiler_active_during_formal':False,
            'optimized_reference':'output-copy CBE; same-run control, not f80f','source':json.loads((ROOT/'byte-delivery.json').read_text())}
        (self.logs/'run_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
        for mode in ['bf16','candidate','optimized']:
            self.start_server(mode,0,profiling=False)
            self.run_client(mode,1,8000,1,'smoke_8k_'+mode,timeout=300)
            cmd=[self.python,str(ROOT/'scripts/check_model_output.py'),'--mode',mode,'--output',str(self.results/('precision_model_'+mode+'.json'))]
            with (self.logs/('precision_model_'+mode+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
            for scene,inp,out,concurrency in scenes:
                self.run_client(mode,concurrency,inp,out,scene+'_'+mode+'_warmup')
                for repeat in [1,2,3]:self.run_client(mode,concurrency,inp,out,scene+'_'+mode+'_r'+str(repeat))
            self.stop_server()
        a=json.loads((self.results/'precision_model_optimized.json').read_text())
        b=json.loads((self.results/'precision_model_candidate.json').read_text())
        assert len(a['records'])==len(b['records'])==4
        for old,new in zip(a['records'],b['records']):
            assert old['payload_sha256']==new['payload_sha256']
            assert old['response']['choices'][0]['logprobs']['tokens']==new['response']['choices'][0]['logprobs']['tokens']
        (ROOT/'model-token-equality.json').write_text(json.dumps({'passed_samples':4,'equal_token_ids':440,'scope':'output-copy TQ vs paired TQ; not BF16 semantic quality'})+'\n')
        self.status(stage='byte_short_long_finished',result_dir=str(self.results))

if __name__=='__main__':
    (ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    r=Runner()
    signal.signal(signal.SIGTERM,lambda sig,frame:(_ for _ in ()).throw(KeyboardInterrupt('signal '+str(sig))))
    try:r.run()
    except BaseException as e:
        r.status(stage='controller_failed',error=repr(e));r.issue('output-copy-matrix',repr(e),r.logs,'preserve all samples and failures');(ROOT/'background.exit').write_text('1\n');raise
    else:(ROOT/'background.exit').write_text('0\n')
    finally:r.stop_server()
