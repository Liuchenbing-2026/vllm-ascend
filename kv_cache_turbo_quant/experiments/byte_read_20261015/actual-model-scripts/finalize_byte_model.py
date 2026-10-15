"""Wait for the owned matrix and validate complete cohorts, with no exclusions."""
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_short_long import summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 (a.root/'finalizer.pid').write_text(str(os.getpid())+'\n')
 while not all((a.root/n).exists() for n in ['host.exit','background.exit']):
  host_exit=a.root/'host.exit'
  if host_exit.exists() and host_exit.read_text().strip()!='0':
   (a.root/'acceptance.final.json').write_text(json.dumps({'complete':False,'goal_achieved':False,'host_exit':host_exit.read_text().strip(),'controller_exit_available':(a.root/'background.exit').exists(),'scope':'host setup failed before complete controller outputs; original evidence retained'},indent=2)+'\n');return
  time.sleep(20)
 exits={n:(a.root/n).read_text().strip() for n in ['host.exit','background.exit']}
 if any(v!='0' for v in exits.values()):
  (a.root/'acceptance.final.json').write_text(json.dumps({'complete':False,'goal_achieved':False,'exits':exits,'failure_preserved':True},indent=2)+'\n');return
 summary=summarize(a.root);assert summary['matrix_complete'] and summary['successful']==684 and summary['token_equality']['equal_token_ids']==440
 parameters=json.loads((a.root/'logs'/summary['run_id']/'run_parameters.json').read_text());assert parameters['physical_cards']==[2,5] and not parameters['profiler_active_during_formal']
 (a.root/'summary.final.json').write_text(json.dumps(summary,indent=2)+'\n')
 checks=[]
 for scene in ['20_20_c16','2k200_c60']:
  current=next(x for x in summary['rows'] if x['scene']==scene and x['mode']=='candidate');base=next(x for x in summary['rows'] if x['scene']==scene and x['mode']=='bf16')
  checks.append({'scene':scene,'throughput_no_regression':current['output_tps']>=base['output_tps'],'ttft_no_regression':current['ttft_s']<=base['ttft_s'],'tpot_no_regression':current['tpot_ms']<=base['tpot_ms'],'output_gain_percent':current['gain_vs_bf16_percent']})
 achieved=all(x['throughput_no_regression'] and x['ttft_no_regression'] and x['tpot_no_regression'] for x in checks) and checks[1]['output_gain_percent']>0
 (a.root/'acceptance.final.json').write_text(json.dumps({'complete':True,'goal_achieved':achieved,'criteria':'same-run BF16; short throughput/TTFT/TPOT no regression and long positive throughput with latency no regression','checks':checks,'checked_utc':datetime.now(timezone.utc).isoformat(),'source':parameters['source'],'scope':'phase only20/20C16 and2000/200C60; no new8kC256,128K or semantic quality benchmark; device micro gain not whole-model gain'},indent=2)+'\n')
 print(json.dumps({'goal_achieved':achieved,'checks':checks}),flush=True)

if __name__=='__main__':main()
