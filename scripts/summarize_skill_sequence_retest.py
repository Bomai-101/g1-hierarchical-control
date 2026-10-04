"""Summarize verified retest without mixing prior seed42 or unconfirmed stops."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
base=json.loads((a.verification/'summary.json').read_text())
if base['partial']:raise RuntimeError('Cannot freeze a partial retest')
rows=list(csv.DictReader((a.verification/'cases.csv').open()))
def yes(x):return x=='True'
def dist(rr,key):
    vals=[float(r[key]) for r in rr if r.get(key) not in ('',None)]
    return dict(n=len(vals),median=float(np.median(vals)),p90=float(np.quantile(vals,.9)),max=max(vals),min=min(vals)) if vals else dict(n=0,median=None,p90=None,max=None,min=None)
groups=[]
for protocol in ['normal','temporary','persistent']:
    rr=[r for r in rows if r['protocol']==protocol];passed=[r for r in rr if yes(r['persistent_hold_passed'] if protocol=='persistent' else r['sequence_passed'])]
    groups.append(dict(protocol=protocol,total=len(rr),passed=len(passed),physical_failures=sum(r['failure_s']!='' for r in rr),confirmed_stops=sum(r['stopping_confirmed_s']!='' for r in rr),completed=sum(yes(r['sequence_completed']) for r in rr),metrics_all_available={key:dist(rr,key) for key in ['drop_to_detection_s','delivery_resumed_to_fresh_s','stopping_latency_s','request_to_stop_net_m','request_to_stop_path_m','hold_net_drift_m','hold_path_m','hold_duration_s','total_stop_net_m','total_stop_path_m','hold_latency_s','resume_speed_confirm_latency_s','resume_tail_vx_rmse_mps']},availability='Distance/timing summaries use every available first-episode measurement, not success-only filtering. Missing confirmation is excluded with explicit n.'))
strata=[]
for seed in sorted(set(r['seed'] for r in rows)):
 for protocol in ['normal','temporary','persistent']:
  for trigger in [3.22,6.22,9.22]:
   rr=[r for r in rows if r['seed']==seed and r['protocol']==protocol and abs(float(r['planned_trigger_s'])-trigger)<.061]
   strata.append(dict(seed=int(seed),protocol=protocol,trigger_base_s=trigger,total=len(rr),passed=sum(yes(r['persistent_hold_passed'] if protocol=='persistent' else r['sequence_passed']) for r in rr)))
result=dict(groups=groups,strata=strata,verification=base['verification'],scope='Prospective held-out reset seeds and entry times. World root pose variability only, no new disturbances. No original skill criteria relaxed. Statistics descriptive; copies within a seed are not independent trials.')
(a.output/'summary.json').write_text(json.dumps(result,indent=2))
with (a.output/'strata.csv').open('w') as f:
 w=csv.DictWriter(f,list(strata[0]));w.writeheader();w.writerows(strata)
print(json.dumps(groups,indent=2))
