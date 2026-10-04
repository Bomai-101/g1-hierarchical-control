"""Independent candidate/1499 handoffs, explicit marching phase input."""
import argparse,hashlib,json,sys,csv
from pathlib import Path
import numpy as np
import mujoco
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from g1_control.hierarchy.skill_adapter import skill_observation,handoff_action
from g1_control.hierarchy.skill_metrics import metrics
from skill_heading_feedback import heading_command

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('reference-root','walk-policy','skill-policy','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--skill',choices=('hold','march'),required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'mujoco'))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import read_base_state
    meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text());defaults=np.array(meta['default_joint_pos']);walker=TorchActorPolicy(a.walk_policy);skill=TorchActorPolicy(a.skill_policy);expected=123 if a.skill=='hold' else 125
    if skill.weights[0].shape[1]!=expected:raise RuntimeError('Candidate dimensions')
    cases=[dict(vx=v,yaw=y,switch_s=t,phase_offset=phase,gain=gain) for v in (.5,) for y in (-.2,0.,.2) for t,phase in ((5.22,0.),(5.44,np.pi/2)) for gain in (0.,.5,1.)]
    if a.smoke:cases=[cases[0]]
    post=1. if a.smoke else 10.;plan=dict(feedback_start_s=2.,feedback_ramp_s=.5,yaw_limit=.2,target="Heading at actualhandoff",skill=a.skill,cases=cases,post_duration_s=post,blend_s=.5,policy_dt=.02,physics_dt=.001,scope='Independent fixed-schedule handoff to candidate; not online watchdog registration. MuJoCo contact means ground contact with ordered ankle-roll body.',pass_criteria=dict(hold='Full10s, settling remains through end, net displacement<=.5m, heading excursion<=.2rad, roll/pitch<=20deg',march='Full10s, >=5ground-contact losses and >=5 landings/foot, liftheight above ankle-floor .055m >=.025m, net displacement<=.2m, heading excursion<=.2rad, tilt<=20deg; provisional metrics, not safety'))
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2));sources=[Path(__file__).resolve(),a.model,a.walk_policy,a.skill_policy,meta_path,REPO/'src/g1_control/hierarchy/skill_adapter.py',REPO/'src/g1_control/hierarchy/skill_metrics.py',Path(__file__).with_name('skill_heading_feedback.py'),a.output/'plan.json'];hashes={str(f.resolve()):sha(f) for f in sources};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2));results=[]
    for case_index,c in enumerate(cases):
        out=a.output/f'case_{case_index:02d}';out.mkdir();m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);names=meta['joint_names'];ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);feet=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')])
        if min(ids)<0 or min(ai)<0 or min(feet)<0:raise RuntimeError('Joint/foot contract')
        d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,.74];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=defaults;mujoco.mj_forward(m,d);last=np.zeros(37);entry=None;anchor=None;states=[];signals=[];feet_height=[];feet_contact=[];base_obs=[];inputs=[];actions=[];proposals=[];targets=[];peak=np.zeros(37);digest=hashlib.sha256();old_yaw=0.;heading=0.;termination='observation_complete';ticks=round((c['switch_s']+post)/.02)
        for tick in range(ticks+1):
            now=tick*.02;mujoco.mj_forward(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');yaw=float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2)));heading+=(yaw-old_yaw+np.pi)%(2*np.pi)-np.pi;old_yaw=yaw
            states.append(np.r_[d.qpos,d.qvel].copy());signals.append(np.r_[d.qpos[:3],quat,lin,ang,quat_rotate_inverse(quat,lin),quat_rotate_inverse(quat,ang),height,heading]);feet_height.append(d.xpos[feet,2].copy());contact=np.zeros(2,dtype=bool)
            for k in range(d.ncon):
                b1,b2=m.geom_bodyid[d.contact[k].geom]
                for j,f in enumerate(feet):
                    if (b1==f and b2==0) or (b2==f and b1==0):contact[j]=True
            feet_contact.append(contact)
            if height<.35:termination='height_termination';break
            if tick==ticks:break
            after=now>=c['switch_s']-1e-9;command=np.zeros(3) if after else np.array([c['vx'],0.,0. if now<2.4-1e-9 else c['yaw']])
            if after:
                if anchor is None:anchor=heading
                command[2]=heading_command(heading,anchor,now-c['switch_s'],c['gain'])
            obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last)
            if after:
                if entry is None:entry=last.copy()
                internal=skill_observation(obs,a.skill,max(0.,now-c['switch_s']),c['phase_offset']);proposal=skill.act(internal).astype(float);last=handoff_action(entry,proposal,max(0.,now-c['switch_s']))
            else:
                internal=np.r_[obs,np.zeros(expected-123)].astype(np.float32);proposal=walker.act(obs).astype(float);last=proposal
            target=defaults+meta['action_scale']*last;base_obs.append(obs);inputs.append(internal);actions.append(last.copy());proposals.append(proposal.copy());targets.append(target);d.ctrl[ai]=target
            for _ in range(20):
                mujoco.mj_step(m,d);peak=np.maximum(peak,abs(d.qfrc_actuator[va]));digest.update(d.qpos.tobytes());digest.update(d.qvel.tobytes());digest.update(d.ctrl.tobytes())
                if not np.all(np.isfinite(np.r_[d.qpos,d.qvel])) or any(w.number for w in d.warning):raise RuntimeError('Numeric/physics warning')
        trace=dict(time=np.arange(len(states))*.02,states=np.array(states),signals=np.array(signals),feet_height=np.array(feet_height),feet_contact=np.array(feet_contact),base_observations=np.array(base_obs),skill_inputs=np.array(inputs),actions=np.array(actions),proposals=np.array(proposals),targets=np.array(targets),force_max=peak);np.savez_compressed(out/'trace.npz',**trace)
        if np.any(peak>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap')
        i=round(c['switch_s']/.02)
        if len(states)<=i:result=dict(case=case_index,**c,termination=termination,passed=False,reason='Failed before handoff')
        else:
            score=metrics(trace,c['switch_s']);contacts=trace['feet_contact'][i:];edges=np.diff(contacts.astype(int),axis=0);losses=(edges<0).sum(axis=0);landings=(edges>0).sum(axis=0);lift=np.max(trace['feet_height'][i:],axis=0)-.055;full=termination=='observation_complete' and not a.smoke;tilt=max(score['max_abs_roll_deg'],score['max_abs_pitch_deg']);passed=full and tilt<=20 and score['max_heading_excursion_rad']<=.2
            if a.skill=='hold':passed=passed and score['eligible_through_end_after_confirmation'] and score['net_displacement_m']<=.5
            else:passed=passed and bool(np.all(losses>=5)) and bool(np.all(landings>=5)) and bool(np.all(lift>=.025)) and score['net_displacement_m']<=.2
            result=dict(case=case_index,**c,termination=termination,passed=bool(passed),smoke=a.smoke,foot_contact_losses=losses.tolist(),foot_landings=landings.tolist(),max_lift_above_ankle_floor_m=lift.tolist(),states=len(states),actor_calls=len(actions),physics_steps=len(actions)*20,physics_state_control_sha256=digest.hexdigest(),**score)
        (out/'complete.json').write_text(json.dumps(result,indent=2));results.append(result);print('MU_SKILL_EVAL',json.dumps(result),flush=True)
    for path,expected_hash in hashes.items():
        if sha(Path(path))!=expected_hash:raise RuntimeError('Input changed '+path)
    (a.output/'complete.json').write_text(json.dumps(dict(skill=a.skill,engine='mujoco',python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,results=results,passed_cases=sum(r['passed'] for r in results),total_cases=len(cases),candidate_only=True,smoke=a.smoke),indent=2))

if __name__=='__main__':main()
