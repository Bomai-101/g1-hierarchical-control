"""Independent posture-hold probes with current123D/37D interface."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import mujoco
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from g1_control.hierarchy.skill_interface import VelocityCommand
from g1_control.hierarchy.experimental_hold import ExperimentalPostureHold
from evaluate_command_stop import metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root','model','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'mujoco'))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import read_base_state
    meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';policy_path=a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz';meta=json.loads(meta_path.read_text());defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(policy_path)
    plan=dict(switch_s=5.22,ramp_s=.5,post_duration_s=10.,physics_dt=.001,policy_dt=.02,prelude_straight_s=2.4,directions={'right':-.2,'straight':0.,'left':.2},responses=['zero','capture_pose','nominal_pose'],forward_mps=.5,settling=dict(window_s=.5,dwell_s=1.,speed_limit_mps=.05,euler_yaw_rms_limit_rps=.05),blend='Smoothstep0.50s from previous applied37D action; capture_pose uses entry123D q delta, nominal_pose uses defaults. Final target clipped to model joint ranges. Original position actuator gains/limits.',scope='Independent posture probes, no watchdog registration or runtime authority; scalar timing and123D/37D interface, no legacy64D/6D controller. No balance feedback. No physics reset at handoff.',termination='height<.35m ends simulation; numeric/physics warning fails run. Ending simulation is not stopping.',metric_definitions='Planar path from root xy sampled20ms; speed from body COM horizontal velocity; Euler heading unwrapped, backward yaw-rate difference20ms; roll/pitch standard wxyz convention. Final2s is last2s before end, potentially shorter than10s on failure.')
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2));files=[Path(__file__).resolve(),a.model,meta_path,policy_path,a.output/'plan.json',REPO/'src/g1_control/hierarchy/skill_interface.py',REPO/'src/g1_control/hierarchy/experimental_hold.py',REPO/'scripts/evaluate_command_stop.py',a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py'];hashes={str(f.resolve()):sha(f) for f in files};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2));results=[]
    for direction,yaw in plan['directions'].items():
        for response in plan['responses']:
            label=direction+'_'+response;out=a.output/label;out.mkdir();m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
            if abs(m.opt.timestep-.001)>1e-12:raise RuntimeError('Wrong dt')
            names=meta['joint_names'];ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names])
            if min(ids)<0 or min(ai)<0:raise RuntimeError('Interface mismatch')
            qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];probe=None;controller_modes=[];actor_calls=0;handoff=None;d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,.74];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=defaults;mujoco.mj_forward(m,d)
            last=np.zeros(37);states=[];signals=[];observations=[];actions=[];targets=[];commands=[];peak=np.zeros(37);digest=hashlib.sha256();prefix=hashlib.sha256();old_yaw=0.;heading=0.;steps=0;termination='observation_complete'
            for tick in range(762):
                now=tick*.02;mujoco.mj_forward(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');euler=float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2)));heading+=(euler-old_yaw+np.pi)%(2*np.pi)-np.pi;old_yaw=euler
                states.append(np.r_[d.qpos,d.qvel].copy());signals.append(np.r_[d.qpos[:3],quat,lin,ang,quat_rotate_inverse(quat,lin),quat_rotate_inverse(quat,ang),height,heading])
                if height<.35:termination='height_termination';break
                if tick==761:break
                command=np.array([.5,0.,0. if tick<120 else yaw]) if tick<261 else np.zeros(3);VelocityCommand(*command);obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last)
                if tick>=261 and response!='zero':
                    if probe is None:
                        if not np.all(m.jnt_limited[ids]):raise RuntimeError('Probe requires finite joint limits')
                        probe=ExperimentalPostureHold(defaults,meta['action_scale'],m.jnt_range[ids,0],m.jnt_range[ids,1],mode=response,blend_s=.5);probe.enter(now,obs,last);handoff=dict(time_s=now,previous_action=last.tolist(),goal_action=probe.goal.tolist(),goal_target=(defaults+meta['action_scale']*probe.goal).tolist(),blend_s=.5)
                    last=probe.act(now,obs);controller_modes.append(response)
                else:last=actor.act(obs).astype(float);actor_calls+=1;controller_modes.append('actor1499')
                target=defaults+meta['action_scale']*last;commands.append(command);observations.append(obs);actions.append(last.copy());targets.append(target.copy());d.ctrl[ai]=target
                for _ in range(20):
                    mujoco.mj_step(m,d);steps+=1;peak=np.maximum(peak,abs(d.qfrc_actuator[va]))
                    for values in (d.qpos,d.qvel,d.ctrl):
                        digest.update(values.tobytes())
                        if tick<261:prefix.update(values.tobytes())
                    if not np.all(np.isfinite(np.r_[d.qpos,d.qvel])) or any(w.number for w in d.warning):raise RuntimeError(label+' nonfinite/physics warning')
            trace=dict(time=np.arange(len(states))*.02,states=np.array(states),signals=np.array(signals),commands=np.array(commands),observations=np.array(observations),actions=np.array(actions),targets=np.array(targets),force_max=peak,controller_modes=np.array(controller_modes))
            if len(states)<=262:raise RuntimeError('Failure before response '+label)
            if np.any(peak>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap')
            np.savez_compressed(out/'trace.npz',**trace);result=dict(label=label,direction=direction,yaw_command=yaw,response=response,termination=termination,states=len(states),policy_calls=len(actions),physical_steps=steps,actor_calls=actor_calls,handoff=handoff,physics_state_control_sha256=digest.hexdigest(),prefix_physics_sha256=prefix.hexdigest(),**metrics(trace));(out/'complete.json').write_text(json.dumps(result,indent=2));results.append(result);print('POSTURE_PROBE',label,termination,json.dumps(metrics(trace)),flush=True)
    for path,expected in hashes.items():
        if sha(Path(path))!=expected:raise RuntimeError('Input changed '+path)
    (a.output/'complete.json').write_text(json.dumps(dict(results=results,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,plan=plan),indent=2))
    with (a.output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)

if __name__=='__main__':main()
