"""Freeze development-only representatives, then describe heldout branches."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--branches',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.report/'plan.json').read_text());x=np.load(a.source/'responses.npz');rt=json.loads((a.source/'runtime.json').read_text());complete=json.loads((a.branches/'complete.json').read_text());names=rt['joint_names'];legs=[i for i,n in enumerate(names) if any(s in n for s in ('hip_','knee_','ankle_'))];scores={}
for variant in complete['variants']:
 y=np.load(a.branches/f'{variant}.npz');diff=y['joint_pos'][:,plan['development_indices']][:,:,legs]-x['joint_pos'][:,plan['development_indices']][:,:,legs];scores[variant]=float(np.sqrt(np.mean(diff**2)))
selected=['baseline']
for prefix in ('friction','contact_time'):selected.append(min((v for v in scores if v.startswith(prefix)),key=lambda v:(scores[v],v)))
selection=dict(selected=selected,development_leg_rmse_rad=scores,rule=plan['selection'],plan_sha256=hashlib.sha256((a.report/'plan.json').read_bytes()).hexdigest(),branch_complete_sha256=hashlib.sha256((a.branches/'complete.json').read_bytes()).hexdigest())
(a.report/'selection.json').write_text(json.dumps(selection,indent=2))
def yaw(q):return np.unwrap(np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2)),axis=0)
rows=[]
for variant in complete['variants']:
 y=np.load(a.branches/f'{variant}.npz')
 for ci,c in enumerate(rt['cases']):
  err=y['joint_pos'][:,ci][:,legs]-x['joint_pos'][:,ci][:,legs];rows.append(dict(variant=variant,case=c['name'],split='development' if ci in plan['development_indices'] else 'heldout',leg_rmse_rad=float(np.sqrt(np.mean(err**2))),leg_max_rad=float(abs(err).max()),heading_final_error_rad=float(yaw(y['root_link'][:,ci,3:7])[-1]-yaw(x['root_link'][:,ci,3:7])[-1]),com_vx_final_error_mps=float(y['root_com'][-1,ci,7]-x['root_com'][-1,ci,7])))
with (a.report/'branch_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(selection,indent=2))
