"""Independent complete-cohort validation; early host failures terminate too."""
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from summarize_matrix import summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 (a.root/'finalizer.pid').write_text(str(os.getpid())+'\n')
 while True:
  exits={n:(a.root/n).read_text().strip() for n in ['host.exit','background.exit'] if (a.root/n).exists()}
  if any(v!='0' for v in exits.values()) or len(exits)==2:break
  time.sleep(20)
 if len(exits)!=2 or any(v!='0' for v in exits.values()):
  (a.root/'acceptance.final.json').write_text(json.dumps({'complete':False,'goal_achieved':False,'exits':exits,'failure_preserved':True},indent=2)+'\n');return
 summary=summarize(a.root)
 assert summary['matrix_complete'] and summary['successful']==684 and summary['token_equality']['equal_token_ids']==440
 parameters=json.loads((a.root/'logs'/summary['run_id']/'run_parameters.json').read_text())
 assert parameters['physical_cards']==[2,5] and not parameters['profiler_active_during_formal']
 (a.root/'summary.final.json').write_text(json.dumps(summary,indent=2)+'\n')
 checks=[]
 for scene in ['20_20_c16','2k200_c60']:
  current=next(x for x in summary['rows'] if x['scene']==scene and x['mode']=='candidate')
  base=next(x for x in summary['rows'] if x['scene']==scene and x['mode']=='bf16')
  checks.append(dict(scene=scene,throughput_no_regression=current['output_tps']>=base['output_tps'],ttft_no_regression=current['ttft_s']<=base['ttft_s'],tpot_no_regression=current['tpot_ms']<=base['tpot_ms'],output_gain_percent=current['gain_vs_bf16_percent']))
 achieved=all(x['throughput_no_regression'] and x['ttft_no_regression'] and x['tpot_no_regression'] for x in checks) and checks[1]['output_gain_percent']>0
 (a.root/'acceptance.final.json').write_text(json.dumps(dict(complete=True,goal_achieved=achieved,criteria='same-run BF16 short no throughput/TTFT/TPOT regression, long positive throughput and no latency regression',checks=checks,source=parameters['source'],checked_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
 print(json.dumps(dict(goal_achieved=achieved,checks=checks)),flush=True)

if __name__=='__main__':main()
