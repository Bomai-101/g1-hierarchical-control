"""Verify isolated probes, response calibration, heading tests, and existing Isaac evidence."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np


def require(ok,message):
    if not ok:raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.reference_root/'mujoco'))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import resolve_joint_id
    meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz')
    model_path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb';fitpath=a.run_root/'usd_yaw_diagnosis/response_fit.json';fits=json.loads(fitpath.read_text())['fits'];evidence=[];allrows=[];filehashes={}
    for foldername in ('usd_yaw_diagnosis','usd_yaw_compensation'):
        folder=a.run_root/foldername;complete=json.loads((folder/'complete.json').read_text());inputs=json.loads((folder/'inputs.json').read_text());headingmode=foldername=='usd_yaw_compensation'
        for path,h in inputs.items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'Input changed: '+path)
        (a.output/f'{foldername}_inputs.json').write_bytes((folder/'inputs.json').read_bytes());(a.output/f'{foldername}_complete.json').write_bytes((folder/'complete.json').read_bytes())
        for row in complete['results']:
            speed=row['speed'];variant='compensated' if headingmode else row['variant'];stem=f"usd_position_vx{speed:g}_{row['case']}" if headingmode else f"{variant}_vx{speed:g}_yaw{row['yaw_command']:+.1f}"
            path=folder/f'{stem}.npz';trace=np.load(path);signal=trace['signals'];time=trace['time'];model=mujoco.MjModel.from_binary_path(str(model_path));original=mujoco.MjModel.from_binary_path(str(model_path))
            if variant=='dt_half':model.opt.timestep=.0005
            if variant=='friction06':model.geom_friction[(model.geom_contype!=0)|(model.geom_conaffinity!=0),0]=.6
            changed=[n for n in dir(model) if isinstance(getattr(model,n),np.ndarray) and not np.array_equal(getattr(model,n),getattr(original,n))]
            require(changed==(['geom_friction'] if variant=='friction06' else []),'Unexpected changed array')
            require(abs(model.opt.timestep-(.0005 if variant=='dt_half' else .001))<1e-12,'Unexpected timestep')
            data=mujoco.MjData(model);pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');ids=np.array([resolve_joint_id(model,n)[0] for n in meta['joint_names']]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
            rebuilt=[];actor_error=0.;selected=set(np.linspace(0,len(trace['actions'])-1,16,dtype=int))
            for i,state in enumerate(trace['states']):
                data.qpos[:]=state[:model.nq];data.qvel[:]=state[model.nq:];mujoco.mj_forward(model,data)
                quat=data.xquat[pelvis].copy();forward=data.xmat[pelvis].reshape(3,3)[:,0];heading=np.arctan2(forward[1],forward[0]);spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,spatial,0);ang,lin=spatial[:3],spatial[3:]
                blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang);gravity=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-gravity[2],-1,1))
                if headingmode:
                    raw=row['target_heading_rad']-heading;err=np.arctan2(np.sin(raw),np.cos(raw));cmd=np.array([speed,0,np.clip(.5*err+fits[str(speed)]['experimental_command_bias'],-1,1)]);tail=[heading,err,row['target_heading_rad']]
                else:cmd=np.array([speed,0,row['yaw_command']]);tail=[heading]
                rebuilt.append(np.r_[data.qpos[:3],quat,lin,ang,blin,bang,data.xpos[pelvis,2],tilt,cmd,tail])
                if i in selected:
                    previous=trace['actions'][i-1] if i else np.zeros(37);obs=build_policy_observation(quat,lin,ang,cmd,data.qpos[qa],data.qvel[va],defaults,previous)
                    require(np.array_equal(obs[9:12],cmd.astype(np.float32)),'Actor command mismatch');actor_error=max(actor_error,float(np.max(np.abs(actor.act(obs)-trace['actions'][i]))))
            rebuilt=np.array(rebuilt);error=float(np.max(np.abs(rebuilt-signal)));require(error<1e-10 and actor_error<1e-6,'Signal/actor mismatch')
            duration=30 if headingmode else 20;require(abs(time[-1]-duration)<1e-8 and np.allclose(np.diff(time),.02,atol=1e-10),'Timeline differs');require(len(time)==duration*50+1 and len(trace['actions'])==duration*50 and row['physical_steps']==round(duration/model.opt.timestep),'Counts differ');require(not row['fell'] and not row['numerical_failure'],'Episode failed');require(np.all(trace['force_max']<=model.jnt_actfrcrange[ids,1]+1e-7),'Force cap exceeded')
            mask=time>=time[-1]-10;h=np.unwrap(rebuilt[:,24]);t=time[mask];center=t-t.mean();slope=float(np.sum(center*(h[mask]-h[mask].mean()))/np.sum(center**2))
            if headingmode:
                e=rebuilt[:,25];metrics=dict(steady_signed_error_deg=float(np.rad2deg(e[mask].mean())),steady_error_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(e[mask]**2)))))
                for band in (5,15):
                    outside=np.flatnonzero(np.abs(e)>np.deg2rad(band));candidate=int(outside[-1]+1) if len(outside) else 0
                    metrics[f'settling_{band}deg_s']=float(time[candidate]) if candidate<len(time) and time[candidate]<=time[-1]-2 else None
            else:metrics=dict(heading_slope=slope,heading_endpoint_rate=float((h[-1]-h[np.flatnonzero(mask)[0]])/(time[-1]-time[np.flatnonzero(mask)[0]])),mean_world_wz=float(rebuilt[mask,12].mean()),mean_body_wz=float(rebuilt[mask,18].mean()))
            metrics['steady_forward_rmse' if headingmode else 'forward_rmse']=float(np.sqrt(np.mean((rebuilt[mask,13]-speed)**2)))
            metrics['max_tilt_deg']=float(np.rad2deg(rebuilt[:,20].max()))
            if headingmode:
                metrics['steady_mean_world_wz']=float(rebuilt[mask,12].mean());metrics['steady_mean_yaw_command']=float(rebuilt[mask,23].mean())
                metrics['steady_max_abs_error_deg']=float(np.rad2deg(np.abs(e[mask]).max()));metrics['final_heading_error_deg']=float(np.rad2deg(e[-1]))
                initial=np.arctan2(np.sin(row['target_heading_rad']-row['start_heading_rad']),np.cos(row['target_heading_rad']-row['start_heading_rad']))
                travel=np.r_[0,np.cumsum(np.arctan2(np.sin(np.diff(rebuilt[:,24])),np.cos(np.diff(rebuilt[:,24]))))]
                metrics['overshoot_deg']=float(np.rad2deg(max(0.,max(np.sign(initial)*(travel-initial))))) if abs(initial)>1e-8 else None
                for band in (5,15):
                    inside=np.flatnonzero(np.abs(e)<=np.deg2rad(band));metrics[f'first_entry_{band}deg_s']=float(time[inside[0]]) if len(inside) else None
            for key,val in metrics.items():require((row[key] is None and val is None) or (row[key] is not None and val is not None and abs(row[key]-val)<1e-8),'Metric differs: '+key)
            digest=hashlib.sha256(path.read_bytes()).hexdigest();filehashes[str(path.resolve())]=digest;evidence.append(dict(case=stem,states=len(time),actor_inputs_checked=16,signal_max_abs_error=error,actor_max_abs_error=actor_error,raw_sha256=digest));allrows.append(dict(experiment=foldername,**row))
    probes=[r for r in allrows if r['experiment']=='usd_yaw_diagnosis']
    for speed in (.5,1.):
        near=sorted([r for r in probes if r['variant']=='baseline' and r['speed']==speed and r['yaw_command'] in (0.,.1)],key=lambda r:r['yaw_command']);gain=(near[1]['heading_slope']-near[0]['heading_slope'])/.1;bias=-near[0]['heading_slope']/gain
        require(abs(bias-fits[str(speed)]['experimental_command_bias'])<1e-12,'Fit changed')
    isaac=a.run_root/'isaac_posttraining';protocol=json.loads((isaac/'protocol.json').read_text());require(protocol['checkpoint_hashes']['baseline1499']['sha256']==hashlib.sha256(Path(protocol['checkpoint_hashes']['baseline1499']['path']).read_bytes()).hexdigest(),'Isaac checkpoint changed')
    isaacrows=[]
    for i,case in enumerate(protocol['cases']):
        path=isaac/f'baseline1499_{i:02d}.npz';trace=np.load(path);t=trace['time'];s=trace['trace'];mask=(t[:,None]>=20)&(s[:,:,0]>0);h=np.unwrap(s[:,:,20],axis=0)
        require(np.all(s[:,:,0]>0),'Isaac data include post-reset samples');require(np.allclose(s[:,:,22],case['speed']) and np.allclose(s[:,:,23],0),'Isaac linear command differs')
        error=np.arctan2(np.sin(case['target']-s[:,:,20]),np.cos(case['target']-s[:,:,20]))
        if case['mode']=='heading':require(np.allclose(s[:,:,24],np.clip(.5*error,-1,1),atol=1e-6),'Isaac heading law differs')
        else:require(np.allclose(s[:,:,24],case['target']),'Isaac rate command differs')
        tt=t[t>=20];hh=h[t>=20];tc=tt-tt.mean();slope=float(np.mean(np.sum(tc[:,None]*(hh-hh.mean(axis=0)),axis=0)/np.sum(tc**2)))
        isaacrows.append(dict(case_id=i,**case,mean_world_wz=float(s[:,:,16][mask].mean()),heading_slope=slope,steady_heading_error_deg=float(np.rad2deg(error[mask]).mean()) if case['mode']=='heading' else None,parallel_envs=s.shape[1],raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));filehashes[str(path.resolve())]=isaacrows[-1]['raw_sha256']
    previous=json.loads((a.run_root/'usd_heading_replays/complete.json').read_text())['results'];comparison=[]
    for row in allrows:
        if row['experiment']!='usd_yaw_compensation':continue
        old=next(r for r in previous if r['profile']=='usd_position' and r['speed']==row['speed'] and r['case']==row['case'])
        comparison.append(dict(speed=row['speed'],case=row['case'],before_error_deg=old['steady_signed_error_deg'],compensated_error_deg=row['steady_signed_error_deg'],before_settling5_s=old['settling_5deg_s'],compensated_settling5_s=row['settling_5deg_s'],before_forward_rmse=old['steady_forward_rmse'],compensated_forward_rmse=row['steady_forward_rmse'],before_overshoot_deg=old['overshoot_deg'],compensated_overshoot_deg=row['overshoot_deg']))
    for name,data in [('response.csv',probes),('isaac_existing.csv',isaacrows),('heading_comparison.csv',comparison)]:
        with (a.output/name).open('w') as stream:w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (a.output/'response_fit.json').write_bytes(fitpath.read_bytes());(a.output/'isaac_protocol.json').write_bytes((isaac/'protocol.json').read_bytes())
    (a.output/'verification.json').write_text(json.dumps(dict(cases=evidence,all22complete=True,reconstructed_states=sum(e['states'] for e in evidence),actor_evaluations=sum(e['actor_inputs_checked'] for e in evidence),isaac_cases_checked=12,isaac_reused_not_rerun=True,raw_sha256=filehashes,limits='Single deterministic flat setting; offset diagnostic does not identify a single physical cause and is not default deployment.'),indent=2))
    print('PASS:',len(evidence),'new traces;',sum(e['states'] for e in evidence),'states;',sum(e['actor_inputs_checked'] for e in evidence),'actor evaluations;12 existingIsaac traces')

if __name__=='__main__':main()
