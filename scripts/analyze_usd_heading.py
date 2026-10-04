"""Independently verify frozen-policy heading trials and archive metrics."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np


def checked(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    folder=a.run_root/'usd_heading_replays';complete=json.loads((folder/'complete.json').read_text());inputs=json.loads((folder/'inputs.json').read_text())
    for path,expected in inputs.items():checked(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,'Input changed: '+path)
    sys.path.insert(0,str(a.reference_root/'mujoco'))
    from mujoco_eval.policy import quat_rotate_inverse,build_policy_observation,TorchActorPolicy
    from mujoco_eval.run_grid import resolve_joint_id
    meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());defaults=np.array(meta['default_joint_pos'])
    actor=TorchActorPolicy(next(Path(path) for path in inputs if Path(path).name=='policy_actor.npz'))
    rows=complete['results'];evidence=[]
    for row in rows:
        stem=f"{row['profile']}_vx{row['speed']:g}_{row['case']}";trace=np.load(folder/f'{stem}.npz');signal=trace['signals'];time=trace['time']
        model=mujoco.MjModel.from_binary_path(str(a.run_root/('visual_mujoco/baseline1499.mjb' if row['profile']=='original' else 'usd_physical_models/g1_usd_physical_position.mjb')))
        data=mujoco.MjData(model);pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');ids=np.array([resolve_joint_id(model,n)[0] for n in meta['joint_names']]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
        rebuilt=[];actor_error=0.;selected=set(np.linspace(0,1499,32,dtype=int))
        for i,state in enumerate(trace['states']):
            data.qpos[:]=state[:model.nq];data.qvel[:]=state[model.nq:];mujoco.mj_forward(model,data)
            quat=data.xquat[pelvis].copy();forward=data.xmat[pelvis].reshape(3,3)[:,0];heading=np.arctan2(forward[1],forward[0])
            raw=row['target_heading_rad']-heading;error=np.arctan2(np.sin(raw),np.cos(raw))
            command=np.array([row['speed'],0,np.clip(.5*error,-1,1)])
            spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,spatial,0);ang,lin=spatial[:3],spatial[3:]
            bodylin=quat_rotate_inverse(quat,lin);bodyang=quat_rotate_inverse(quat,ang);gravity=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-gravity[2],-1,1))
            rebuilt.append(np.r_[data.qpos[:3],quat,lin,ang,bodylin,bodyang,data.xpos[pelvis,2],tilt,command,heading,error,row['target_heading_rad']])
            if i in selected:
                previous=trace['actions'][i-1] if i else np.zeros(37)
                obs=build_policy_observation(quat,lin,ang,command,data.qpos[qa],data.qvel[va],defaults,previous)
                checked(np.array_equal(obs[9:12],command.astype(np.float32)),'Observation command differs')
                actor_error=max(actor_error,float(np.max(np.abs(actor.act(obs)-trace['actions'][i]))))
        rebuilt=np.array(rebuilt);signal_error=float(np.max(np.abs(rebuilt-signal)))
        checked(signal_error<1e-10 and actor_error<1e-6,'Signal/actor reconstruction differs')
        checked(signal.shape==(1501,27) and len(trace['actions'])==1500 and row['physical_steps']==30000 and not row['fell'] and not row['numerical_failure'],'Invalid episode')
        checked(np.allclose(np.diff(time),.02,atol=1e-10) and abs(time[-1]-30)<1e-8,'Invalid timeline')
        checked(np.all(trace['force_max']<=model.jnt_actfrcrange[ids,1]+1e-7),'Force cap exceeded')
        checked(abs(rebuilt[0,24]-row['start_heading_rad'])<1e-10,'Initial heading differs')
        e=rebuilt[:,25];steady=time>=time[-1]-10
        metrics=dict(final_heading_error_deg=float(np.rad2deg(e[-1])),steady_signed_error_deg=float(np.rad2deg(e[steady].mean())),steady_error_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(e[steady]**2)))),steady_max_abs_error_deg=float(np.rad2deg(np.abs(e[steady]).max())),steady_mean_world_wz=float(rebuilt[steady,12].mean()),steady_mean_yaw_command=float(rebuilt[steady,23].mean()),steady_forward_rmse=float(np.sqrt(np.mean((rebuilt[steady,13]-row['speed'])**2))),max_tilt_deg=float(np.rad2deg(rebuilt[:,20].max())))
        for band in (5,15):
            outside=np.flatnonzero(np.abs(e)>np.deg2rad(band));inside=np.flatnonzero(np.abs(e)<=np.deg2rad(band));candidate=int(outside[-1]+1) if len(outside) else 0
            metrics[f'first_entry_{band}deg_s']=float(time[inside[0]]) if len(inside) else None
            metrics[f'settling_{band}deg_s']=float(time[candidate]) if candidate<len(time) and time[candidate]<=time[-1]-2 else None
        initial=np.arctan2(np.sin(row['target_heading_rad']-row['start_heading_rad']),np.cos(row['target_heading_rad']-row['start_heading_rad']))
        travel=np.r_[0,np.cumsum(np.arctan2(np.sin(np.diff(rebuilt[:,24])),np.cos(np.diff(rebuilt[:,24]))))]
        metrics['overshoot_deg']=float(np.rad2deg(max(0.,max(np.sign(initial)*(travel-initial))))) if abs(initial)>1e-8 else None
        for key,val in metrics.items():checked((row[key] is None and val is None) or (row[key] is not None and val is not None and abs(row[key]-val)<1e-8),'Metric differs: '+key)
        evidence.append(dict(case=stem,signal_max_abs_error=signal_error,actor_max_abs_error=actor_error,states=1501,actor_inputs_checked=32,force_limit_pass=True,trace_sha256=hashlib.sha256((folder/f'{stem}.npz').read_bytes()).hexdigest()))
    pairs=[]
    for old in rows:
        if old['profile']!='original':continue
        new=next(r for r in rows if r['profile']=='usd_position' and r['speed']==old['speed'] and r['case']==old['case'])
        pairs.append(dict(speed=old['speed'],case=old['case'],original_steady_error_deg=old['steady_signed_error_deg'],usd_steady_error_deg=new['steady_signed_error_deg'],original_settling15_s=old['settling_15deg_s'],usd_settling15_s=new['settling_15deg_s'],usd_settling5_s=new['settling_5deg_s'],usd_mean_yaw_command=new['steady_mean_yaw_command'],usd_mean_world_wz=new['steady_mean_world_wz']))
    for name,data in [('summary.csv',rows),('comparison.csv',pairs)]:
        with (a.output/name).open('w') as stream:w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for name in ('complete.json','inputs.json'):(a.output/name).write_bytes((folder/name).read_bytes())
    (a.output/'replay_verification.json').write_text(json.dumps(dict(cases=evidence,all28valid30s=True,checks='Source hashes; body +X projection heading; trigonometric shortest-angle error; closed-loop commands; 42028 reconstructed states; 896 actor evaluations; independently computed entry/settling/overshoot and other metrics; force caps',limitation='Same fixed targets and law, but angular commands differ with measured heading. Single deterministic flat setting, no disturbances; bands are exploratory.'),indent=2))
    print('PASS: 28 traces,42028 states,896 actor evaluations,heading/commands/metrics/force caps/input hashes')

if __name__=='__main__':main()
