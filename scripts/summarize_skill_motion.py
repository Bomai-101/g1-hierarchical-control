"""Additional recorded motion diagnostics; does not redefine acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
for engine in ('mujoco','isaac'):
 for skill in ('hold','march'):
  folder=a.root/f'{engine}_{skill}';plan=json.loads((folder/'plan.json').read_text());native=engine=='isaac'
  for ci,c in enumerate(plan['cases']):
   z=np.load(folder/f'case_{ci:02d}/trace.npz');r=json.loads((folder/f'case_{ci:02d}/complete.json').read_text());results=r['results'] if native else [r]
   for j,result in enumerate(results):
    full=result.get('full_post_horizon',result.get('termination')=='observation_complete')
    if not full:continue
    switch=result['switch_s'];start=round(switch/.02);end=start+501;t=z['time'][start:end];feet=z['feet_contact'][start:end,j] if native else z['feet_contact'][start:end];h=z['feet_height'][start:end,j] if native else z['feet_height'][start:end];late=t>=t[-1]-2.-1e-9;measured=z['signals'][start:end,j] if native else z['signals'][start:end]
    valid_lifts=[]
    for k in range(2):
     airborne=h[:,k]>=.080;changes=np.diff(np.r_[False,airborne,False].astype(int));begins=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1);valid_lifts.append(int(sum(e-b>=4 and t[b]>=switch+2. for b,e in zip(begins,ends))))
    offset=j*2*np.pi/len(results) if native else c['phase_offset'];phase=(t-switch)*2*np.pi/.9+offset;s=np.sin(phase);desired=np.stack((s<=.2,-s<=.2),axis=1)
    rows.append(dict(engine=engine,skill=skill,case=ci,env=j,protocol=c['protocol'],vx=c['vx'],yaw=c['yaw'],sustained_lift_segments_after_2s=valid_lifts,late_both_feet_contact_fraction=float(np.mean(feet[late].all(axis=1))),post2s_phase_contact_agreement=float(np.mean((feet==desired)[t>=switch+2.])) if skill=='march' else None,late_signed_heading_rad=float(measured[-1,20]-measured[np.flatnonzero(late)[0],20])))
a.output.write_text(json.dumps(dict(definitions='Sustained lift: ankle body >=.080m for >=4 sampled states (>=.06s), begins after switch+2s. Exploratory evidence, not changed acceptance; contact definitions differ across engines. Cold-start and single-seed limits remain.',results=rows),indent=2))
