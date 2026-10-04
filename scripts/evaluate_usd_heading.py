"""Evaluate original training-task heading feedback with frozen baseline1499."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def wrap_pi(angle):
    angle=np.asarray(angle);wrapped=(angle+np.pi)%(2*np.pi)
    return np.where((wrapped==0)&(angle>0),np.pi,wrapped-np.pi)


def heading_from_quat(q):
    w,x,y,z=np.asarray(q).T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def heading_metrics(time,heading,target,start):
    error=wrap_pi(target-heading);steady=time>=time[-1]-10.;initial=float(wrap_pi(target-start))
    result=dict(final_heading_error_deg=float(np.rad2deg(error[-1])),steady_signed_error_deg=float(np.rad2deg(error[steady].mean())),
        steady_error_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(error[steady]**2)))),steady_max_abs_error_deg=float(np.rad2deg(np.abs(error[steady]).max())))
    for degrees in (5,15):
        inside=np.abs(error)<=np.deg2rad(degrees);suffix=np.logical_and.accumulate(inside[::-1])[::-1]
        entered=np.flatnonzero(inside);settled=np.flatnonzero(suffix&(time<=time[-1]-2.))
        result[f'first_entry_{degrees}deg_s']=float(time[entered[0]]) if len(entered) else None
        result[f'settling_{degrees}deg_s']=float(time[settled[0]]) if len(settled) else None
    displacement=np.unwrap(heading)-start
    result['overshoot_deg']=float(np.rad2deg(max(0.,np.max(np.sign(initial)*(displacement-initial))))) if abs(initial)>1e-8 else None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_root/'mujoco'))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import resolve_joint_id,pd_gains,read_base_state
    meta_path=args.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text())
    models=dict(original=args.run_root/'visual_mujoco/baseline1499.mjb',usd_position=args.run_root/'usd_physical_models/g1_usd_physical_position.mjb')
    files=[meta_path,args.policy,Path(__file__).resolve(),*models.values(),args.reference_root/'mujoco/mujoco_eval/policy.py',
        Path('/home/omai/robotics/reference_projects/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py'),
        Path('/home/omai/robotics/reference_projects/IsaacLab/source/isaaclab/isaaclab/utils/math.py'),args.reference_root/'isaac_sim/g1_walk_sim51/g1_env_cfg.py']
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(args.output/'inputs.json').write_text(json.dumps(hashes,indent=2))
    cases=[dict(name='hold0',start=0.,target=0.),dict(name='correct_left',start=.5,target=0.),dict(name='correct_right',start=-.5,target=0.),
        dict(name='left90',start=0.,target=np.pi/2),dict(name='right90',start=0.,target=-np.pi/2),
        dict(name='wrap_left',start=np.deg2rad(179),target=np.deg2rad(-179)),dict(name='wrap_right',start=np.deg2rad(-179),target=np.deg2rad(179))]
    actor=TorchActorPolicy(args.policy);rows=[]
    for profile,path in models.items():
        for speed in (.5,1.):
            for case in cases:
                model=mujoco.MjModel.from_binary_path(str(path));model.opt.timestep=.001;data=mujoco.MjData(model)
                ids=np.array([resolve_joint_id(model,name)[0] for name in meta['joint_names']]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
                ai=np.array([next(i for i in range(model.nu) if model.actuator_trnid[i,0]==jid) for jid in ids]);defaults=np.array(meta['default_joint_pos']);kp,kd=pd_gains(meta['joint_names'])
                data.qpos[:]=model.qpos0;data.qpos[:3]=[0,0,.74];data.qpos[3:7]=[np.cos(case['start']/2),0,0,np.sin(case['start']/2)];data.qpos[qa]=defaults
                last=np.zeros(37);target=defaults.copy();times=[];signals=[];states=[];actions=[];peak_force=np.zeros(37);steps=0;fell=False;numerical=False
                for step in range(30001):
                    if step%20==0:
                        mujoco.mj_forward(model,data);quat,lin,ang,height=read_base_state(model,data,'pelvis');heading=float(heading_from_quat(quat))
                        error=float(wrap_pi(case['target']-heading));command=np.array([speed,0.,np.clip(.5*error,-1,1)])
                        bodylin=quat_rotate_inverse(quat,lin);bodyang=quat_rotate_inverse(quat,ang);gravity=quat_rotate_inverse(quat,[0,0,-1]);tilt=float(np.arccos(np.clip(-gravity[2],-1,1)))
                        times.append(float(data.time));states.append(np.r_[data.qpos,data.qvel].copy());signals.append(np.r_[data.qpos[:3],quat,lin,ang,bodylin,bodyang,height,tilt,command,heading,error,case['target']])
                        if height<.35:fell=True;break
                        if step==30000:break
                        obs=build_policy_observation(quat,lin,ang,command,data.qpos[qa],data.qvel[va],defaults,last)
                        if not np.array_equal(obs[9:12],command.astype(np.float32)):raise RuntimeError('Task command not passed to actor')
                        last=actor.act(obs).astype(float);actions.append(last.copy());target=defaults+meta['action_scale']*last
                    if profile=='usd_position':data.ctrl[ai]=target
                    else:
                        ctrl=kp*(target-data.qpos[qa])-kd*data.qvel[va];limited=model.actuator_ctrllimited[ai]
                        ctrl[limited]=np.clip(ctrl[limited],model.actuator_ctrlrange[ai[limited],0],model.actuator_ctrlrange[ai[limited],1]);data.ctrl[ai]=ctrl
                    mujoco.mj_step(model,data);steps+=1;peak_force=np.maximum(peak_force,np.abs(data.qfrc_actuator[va]))
                    if not np.all(np.isfinite(data.qpos)) or any(data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
                        numerical=True;break
                signal=np.array(signals);time=np.array(times);steady=time>=time[-1]-10
                metrics=heading_metrics(time,signal[:,24],case['target'],case['start'])
                row=dict(profile=profile,speed=speed,case=case['name'],start_heading_rad=case['start'],target_heading_rad=case['target'],elapsed_s=steps*.001,fell=int(fell),numerical_failure=int(numerical),
                    physical_steps=steps,actions=len(actions),steady_mean_world_wz=float(signal[steady,12].mean()),steady_mean_yaw_command=float(signal[steady,23].mean()),
                    steady_forward_rmse=float(np.sqrt(np.mean((signal[steady,13]-speed)**2))),max_tilt_deg=float(np.rad2deg(signal[:,20].max())),**metrics)
                stem=f"{profile}_vx{speed:g}_{case['name']}";np.savez_compressed(args.output/f'{stem}.npz',time=time,states=states,signals=signals,actions=actions,force_max=peak_force)
                (args.output/f'{stem}.json').write_text(json.dumps(row,indent=2));rows.append(row)
                print('HEADING_REPLAY',profile,speed,case['name'],row['elapsed_s'],row['fell'],round(row['steady_signed_error_deg'],3),row['settling_5deg_s'],row['settling_15deg_s'],flush=True)
    for path,expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Source changed')
    (args.output/'complete.json').write_text(json.dumps(dict(results=rows,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,protocol=dict(seed=42,duration=30,physics_dt=.001,policy_dt=.02,
        heading_gain=.5,angular_command_limits=[-1,1],steady_window_s=10,settling_definition='All remaining sampled heading errors within band, at least2s remaining; bands5/15deg are exploratory, not safety limits',cases=cases,
        note='Same heading targets and feedback law; closed-loop angular commands differ across profiles because their measured headings differ. No supervisor or fallback.')),indent=2))


if __name__=='__main__':main()
