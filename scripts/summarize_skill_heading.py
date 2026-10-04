"""Aggregate verified feedback cases; no change to original acceptance."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);a=p.parse_args();rows=[];checks={};trace_hashes={}
for part in ('verification_mujoco','verification_isaac_hold','verification_isaac_march'):
 folder=a.results/part;s=json.loads((folder/'summary.json').read_text())
 for k,v in s['verification'].items():checks[k]=max(checks.get(k,0),v) if k=='max_actor_error' else checks.get(k,0)+v
 trace_hashes.update(json.loads((folder/'trace_hashes.json').read_text()))
 with (folder/'cases.csv').open() as f:
  for r in csv.DictReader(f):
   record=json.loads((folder/f'evidence/{r["engine"]}_{r["skill"]}/case_{int(r["case"]):02d}/complete.json').read_text());record=record['results'][int(r['env'])] if r['engine']=='isaac' else record
   r['final_abs_heading_error_rad']=abs(record.get('signed_heading_change_rad',np.nan));rows.append(r)
groups=[]
for e in ('isaac','mujoco'):
 for skill in ('hold','march'):
  for gain in (0.,.5,1.):
   rr=[r for r in rows if r['engine']==e and r['skill']==skill and float(r['gain'])==gain];full=[r for r in rr if r['full']=='True'];median=lambda key:float(np.median([float(r[key]) for r in full])) if full else None
   groups.append(dict(engine=e,skill=skill,gain=gain,total=len(rr),full=len(full),passed=sum(r['passed']=='True' for r in rr),settled=sum(r.get('settled')=='True' for r in full),cycles_and_lift=sum(r.get('cycles_and_lift')=='True' for r in full),remaining_net_median_m=median('remaining_net_m'),remaining_signed_heading_median_rad=median('remaining_heading_rad'),late2s_abs_heading_change_median_rad=float(np.median([abs(float(r['tail_signed_heading_rad'])) for r in full])) if full else None,final_abs_heading_error_median_rad=median('final_abs_heading_error_rad'),heading_excursion_median_rad=median('heading_excursion_rad'),tail_speed_median_mps=median('tail_speed_rms')))
with (a.results/'cases.csv').open('w') as f:
 fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
(a.results/'summary.json').write_text(json.dumps(dict(groups=groups,verification=checks,total_cases=len(rows),limitations='Lowvx.5 only,seed42; survivors-only medians can be biased by survival changes. Gains fixed prospectively. Nonzero yaw is unseen stationary command training condition. No deployment.'),indent=2));(a.results/'trace_hashes.json').write_text(json.dumps(trace_hashes,indent=2))
for name,h in trace_hashes.items():
 if hashlib.sha256((a.raw/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Changed trace')
for name in ('run_plan.json','status.json'):shutil.copyfile(a.raw/name,a.results/name)
print(json.dumps(dict(groups=groups,checks=checks,traces=len(trace_hashes)),indent=2))
