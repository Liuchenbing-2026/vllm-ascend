"""Durable isolated build, strict precision, memory and alternating micro checks."""
from datetime import datetime,timezone
import hashlib,json,re,subprocess,time,traceback
from pathlib import Path

def main():
 root=Path('/data2/tq-profile-20261014/byte_read_attempt2_20261015');origin=root.parent/'byte_read_20261015';container='tq-profile-20261014-host18'
 def status(stage,**kw):
  row=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kw);(root/'status.json').write_text(json.dumps(row)+'\n')
  with (root/'status.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 def preflight():
  txt=subprocess.check_output(['npu-smi','info'],universal_newlines=True)
  owners=[]
  for line in txt.splitlines():
   m=re.match(r'\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|',line)
   if m and int(m.group(1)) in [2,3]:owners.append({'card':int(m.group(1)),'pid':int(m.group(2))})
  with (root/'resources.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'owners':owners})+'\n')
  return txt,owners
 def run(tag,body):
  txt,owners=preflight();assert not owners,'preserve other card owners: '+str(owners)
  d=root/tag;d.mkdir();cmd=['docker','exec','-e','ASCEND_RT_VISIBLE_DEVICES=2','-e','VLLM_ASCEND_KVTQ_BITS=4','-e','OMP_NUM_THREADS=1',container,'bash','-lc','source /usr/local/Ascend/cann-9.1.0/set_env.sh && source /usr/local/Ascend/cann-9.1.0/opp/vendors/customize/bin/set_env.bash && source /ws/native_read_20261013/opp/vendors/tqread/bin/set_env.bash && source /ws/paired_read_20261014/opp/vendors/tqreadpair/bin/set_env.bash && '+body]
  (d/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');status(tag)
  with (d/'stdout.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=2400)
  (d/'exit').write_text(str(r.returncode)+'\n');assert r.returncode==0,tag+' failed; retain evidence'
 exitcode=1
 try:
  for name,digest in json.loads((root/'source-manifest.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
  while True:
   txt,owners=preflight()
   if not owners:break
   status('waiting_for_free_cards',owners=owners);time.sleep(20)
  (root/'resources-before.log').write_text(txt)
  state=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
  if not state['State']['Running']:subprocess.run(['docker','start',container],check=True)
  assert (origin/'build/exit').read_text().strip()=='0', 'retain original build failure'
  files={};names=set()
  for p in (origin/'opp').rglob('*'):
   if p.is_file():
    blob=p.read_bytes();files[str(p.relative_to(origin))]={'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
    if p.suffix=='.o':names.update(x.decode() for x in re.findall(rb'TqPagedReadByte_[0-9a-f]+_0(?![0-9])',blob))
  assert len(names)==1,names;kernel=names.pop();(root/'build-manifest.json').write_text(json.dumps(files,indent=2)+'\n');(root/'kernel-name.json').write_text(json.dumps({'kernel':kernel})+'\n')
  common='source /ws/byte_read_20261015/opp/vendors/tqreadbyte/bin/set_env.bash && '
  entry='/ws/.venv/bin/python -u /ws/byte_read_attempt2_20261015/scripts/verify_byte_read.py'
  run('functional',common+'exec '+entry+' --precision-only --output-dir /ws/byte_read_attempt2_20261015/functional')
  result=json.loads((root/'functional/verify-native-read.json').read_text());assert result['passed']==15 and result['async_passed']==20 and result['inference_passed']==4 and len(result['lookup_records'])==2
  run('memory',common+'exec mssanitizer --tool=memcheck --kernel-name='+kernel+' --leak-check=yes --log-file=/ws/byte_read_attempt2_20261015/memory/memcheck.log -- '+entry+' --precision-only --output-dir /ws/byte_read_attempt2_20261015/memory')
  log=(root/'memory/memcheck.log').read_text();starts=log.count('Start memcheck sanitizer on kernel '+kernel);clean=log.count('Sanitizer finished on kernel '+kernel+'. No error detected.');leaks=[int(x) for x in re.findall(r'SUMMARY: ([0-9]+) byte\(s\) leaked',log)]
  assert starts==clean==40,(starts,clean);assert all(x==2097152 for x in leaks),leaks;assert not [s for s in log.splitlines() if 'ERROR:' in s and 'LeakCheck' not in s]
  (root/'memory-review.json').write_text(json.dumps({'positive_calls':starts,'no_access_error_calls':clean,'exit_leak_bytes':leaks,'all_memory_issues_closed':not leaks,'scope':'changed byte reader only; original quantizer warnings and exit leak remain open'},indent=2)+'\n')
  run('micro',common+'exec '+entry+' --output-dir /ws/byte_read_attempt2_20261015/micro')
  status('gates_and_micro_complete',kernel=kernel);exitcode=0
 except BaseException:
  (root/'failure.log').write_text(traceback.format_exc());status('failed');raise
 finally:(root/'host.exit').write_text(str(exitcode)+'\n')

if __name__=='__main__':main()
