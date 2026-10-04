"""Independent fixed1499 command response experiment; no watchdog authority."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import mujoco
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from g1_control.hierarchy.skill_interface import VelocityCommand


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def command_at(t,yaw,response,switch=5.22,ramp=.5):
    if t<2.4-1e-9:return np.array([.5,0.,0.])
    command=np.array([.5,0.,yaw])
    if t<switch-1e-9 or response=='hold':return command
    if response=='zero':return np.zeros(3)
    if response=='ramp':return command*max(0.,1.-(t-switch)/ramp)
    raise ValueError('Unknown response')


def settling_evidence(t,speed,yaw_rate,window=.5,dwell=1.,speed_limit=.05,yaw_limit=.05):
    """Causal trailing windows, full-window samples, distinct start/confirmation."""
    dt=float(t[1]-t[0]);n=round(window/dt)+1;need=round(dwell/dt)+1
    smooth_speed=np.full(len(t),np.nan);smooth_yaw=np.full(len(t),np.nan)
    for i in range(n-1,len(t)):
        smooth_speed[i]=np.mean(speed[i-n+1:i+1]);smooth_yaw[i]=np.sqrt(np.mean(yaw_rate[i-n+1:i+1]**2))
    eligible=(smooth_speed<=speed_limit)&(smooth_yaw<=yaw_limit)
    start=None;confirmation=None;run=0
    for i,ok in enumerate(eligible):
        run=run+1 if ok else 0
        if run>=need:
            start=float(t[i-need+1]);confirmation=float(t[i]);break
    escape=None
    if confirmation is not None:
        after=np.flatnonzero((t>confirmation+1e-9)&~eligible)
        if len(after):escape=float(t[after[0]])
    return dict(settling_start_s=start,settling_confirmed_s=confirmation,first_escape_after_confirmation_s=escape,eligible_through_end_after_confirmation=confirmation is not None and escape is None)


def metrics(trace,switch=5.22):
    t=trace['time'];s=trace['signals'];i=round(switch/.02);elapsed=t[i:]-switch;post=s[i:]
    speed=np.linalg.norm(post[:,13:15],axis=1)
    # Euler heading differences, not body angular z; backward difference ends at each sample.
    rates=np.r_[np.nan,np.diff(s[:,20])/.02][i:]
    quat=post[:,3:7];w,x,y,z=quat.T
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    final=elapsed>=elapsed[-1]-2.-1e-9;path=float(np.sum(np.linalg.norm(np.diff(post[:,:2],axis=0),axis=1)))
    evidence=settling_evidence(elapsed,speed,rates) if len(elapsed)>1 else {}
    return dict(post_duration_s=float(elapsed[-1]),path_after_switch_m=path,net_displacement_m=float(np.linalg.norm(post[-1,:2]-post[0,:2])),signed_heading_change_rad=float(post[-1,20]-post[0,20]),max_heading_excursion_rad=float(np.max(abs(post[:,20]-post[0,20]))),final_2s_horizontal_speed_rms_mps=float(np.sqrt(np.mean(speed[final]**2))),final_2s_euler_yaw_rate_rms_rps=float(np.sqrt(np.mean(rates[final]**2))),minimum_height_m=float(np.min(post[:,19])),max_abs_roll_deg=float(np.rad2deg(np.max(abs(roll)))),max_abs_pitch_deg=float(np.rad2deg(np.max(abs(pitch)))),**evidence)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root','model','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'mujoco'))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import read_base_state
    meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';policy_path=a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz';meta=json.loads(meta_path.read_text());defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(policy_path)
    plan=dict(switch_s=5.22,ramp_s=.5,post_duration_s=10.,physics_dt=.001,policy_dt=.02,prelude_straight_s=2.4,directions={'right':-.2,'straight':0.,'left':.2},responses=['hold','zero','ramp'],forward_mps=.5,settling=dict(window_s=.5,dwell_s=1.,speed_limit_mps=.05,euler_yaw_rms_limit_rps=.05),scope='Independent command response; watchdog/task sequencer not connected. Thresholds are exploratory analysis, not safety. No action/history reset at command change.',termination='height<.35m ends simulation; numeric/physics warning fails run. Ending simulation is not stopping.',metric_definitions='Planar path from root xy sampled20ms; speed from body COM horizontal velocity; Euler heading unwrapped, backward yaw-rate difference20ms; roll/pitch standard wxyz convention. Final2s is last2s before end, potentially shorter than10s on failure.')
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2));files=[Path(__file__).resolve(),a.model,meta_path,policy_path,a.output/'plan.json',REPO/'src/g1_control/hierarchy/skill_interface.py',a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py'];hashes={str(f.resolve()):sha(f) for f in files};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2));results=[]
    for direction,yaw in plan['directions'].items():
        for response in plan['responses']:
            label=direction+'_'+response;out=a.output/label;out.mkdir();m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
            if abs(m.opt.timestep-.001)>1e-12:raise RuntimeError('Wrong dt')
            names=meta['joint_names'];ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names])
            if min(ids)<0 or min(ai)<0:raise RuntimeError('Interface mismatch')
            qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,.74];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=defaults;mujoco.mj_forward(m,d)
            last=np.zeros(37);states=[];signals=[];observations=[];actions=[];targets=[];commands=[];peak=np.zeros(37);digest=hashlib.sha256();prefix=hashlib.sha256();old_yaw=0.;heading=0.;steps=0;termination='observation_complete'
            for tick in range(762):
                now=tick*.02;mujoco.mj_forward(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');euler=float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2)));heading+=(euler-old_yaw+np.pi)%(2*np.pi)-np.pi;old_yaw=euler
                states.append(np.r_[d.qpos,d.qvel].copy());signals.append(np.r_[d.qpos[:3],quat,lin,ang,quat_rotate_inverse(quat,lin),quat_rotate_inverse(quat,ang),height,heading])
                if height<.35:termination='height_termination';break
                if tick==761:break
                command=command_at(now,yaw,response,plan['switch_s'],plan['ramp_s']);VelocityCommand(*command);obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last);last=actor.act(obs).astype(float);target=defaults+meta['action_scale']*last;commands.append(command);observations.append(obs);actions.append(last.copy());targets.append(target.copy());d.ctrl[ai]=target
                for _ in range(20):
                    mujoco.mj_step(m,d);steps+=1;peak=np.maximum(peak,abs(d.qfrc_actuator[va]))
                    for values in (d.qpos,d.qvel,d.ctrl):
                        digest.update(values.tobytes())
                        if tick<261:prefix.update(values.tobytes())
                    if not np.all(np.isfinite(np.r_[d.qpos,d.qvel])) or any(w.number for w in d.warning):raise RuntimeError(label+' nonfinite/physics warning')
            trace=dict(time=np.arange(len(states))*.02,states=np.array(states),signals=np.array(signals),commands=np.array(commands),observations=np.array(observations),actions=np.array(actions),targets=np.array(targets),force_max=peak)
            if len(states)<=262:raise RuntimeError('Failure before response '+label)
            if np.any(peak>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap')
            np.savez_compressed(out/'trace.npz',**trace);result=dict(label=label,direction=direction,yaw_command=yaw,response=response,termination=termination,states=len(states),policy_calls=len(actions),physical_steps=steps,physics_state_control_sha256=digest.hexdigest(),prefix_physics_sha256=prefix.hexdigest(),**metrics(trace));(out/'complete.json').write_text(json.dumps(result,indent=2));results.append(result);print('COMMAND_RESPONSE',label,termination,json.dumps(metrics(trace)),flush=True)
    for path,expected in hashes.items():
        if sha(Path(path))!=expected:raise RuntimeError('Input changed '+path)
    (a.output/'complete.json').write_text(json.dumps(dict(results=results,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,plan=plan),indent=2))
    with (a.output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)

if __name__=='__main__':main()
