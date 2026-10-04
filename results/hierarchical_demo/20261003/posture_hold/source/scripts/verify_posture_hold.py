"""Reconstruct actor and posture targets independently; validate handoffs."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
from evaluate_command_stop import metrics

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('run','reference-root','model','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.reference_root/'mujoco'))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import read_base_state
    complete=json.loads((a.run/'complete.json').read_text());plan=json.loads((a.run/'plan.json').read_text());hashes=json.loads((a.run/'inputs.json').read_text())
    for path,digest in hashes.items():require(sha(Path(path))==digest,'Changed input '+path)
    require(str(a.model.resolve()) in hashes,'Model identity');require(len(complete['results'])==9,'Grid count');require(plan['switch_s']==5.22 and plan['ramp_s']==.5 and plan['post_duration_s']==10.,'Frozen schedule')
    meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');traces={};states=0;actors=0;posture_outputs=0
    expected={(direction,response) for direction in ('left','right','straight') for response in ('zero','capture_pose','nominal_pose')};require({(r['direction'],r['response']) for r in complete['results']}==expected,'Grid missing case')
    for result in complete['results']:
        folder=a.run/result['label'];require(json.loads((folder/'complete.json').read_text())==result,'Case metadata');x=np.load(folder/'trace.npz');traces[result['label']]=(x,result);count=result['policy_calls'];require(x['states'].shape==(count+1,87) and x['actions'].shape==(count,37) and x['commands'].shape==(count,3),'Shape');require(result['states']==count+1 and result['physical_steps']==20*count,'Step count');require(np.array_equal(x['time'],np.arange(count+1)*.02),'Sample clock')
        m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);names=meta['joint_names'];ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);last=np.zeros(37);previous_before_handoff=None;goal_action=None;previous_yaw=0.;heading=0.
        initial=m.qpos0.copy();initial[:3]=[0,0,.74];initial[3:7]=[1,0,0,0];initial[qa]=defaults;require(np.array_equal(initial,x['states'][0,:m.nq]) and np.all(x['states'][0,m.nq:]==0),'Initial state')
        for tick,state in enumerate(x['states']):
            require(np.all(np.isfinite(state)),'Finite state');d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');yaw=float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2)));heading+=(yaw-previous_yaw+np.pi)%(2*np.pi)-np.pi;previous_yaw=yaw;signal=np.r_[d.qpos[:3],quat,lin,ang,quat_rotate_inverse(quat,lin),quat_rotate_inverse(quat,ang),height,heading];require(np.max(abs(signal-x['signals'][tick]))<1e-11,'State signals');states+=1
            if tick<count:
                require(height>=.35,'Physics continued after height failure');t=tick*.02;yaw_command={'left':.2,'right':-.2,'straight':0.}[result['direction']];command=np.array([.5,0.,0. if tick<120 else yaw_command])
                if tick>=261:command*=0.
                require(np.max(abs(command-x['commands'][tick]))<2e-15,'Command schedule');obs=build_policy_observation(quat,lin,ang,x['commands'][tick],d.qpos[qa],d.qvel[va],defaults,last);require(np.array_equal(obs,x['observations'][tick]),'Actor observation')
                if tick<261 or result['response']=='zero':
                    action=actor.act(obs);require(np.max(abs(action-x['actions'][tick]))<1e-6,'Actor output');require(x['controller_modes'][tick]=='actor1499','Actor mode');actors+=1
                else:
                    if tick==261:
                        previous_before_handoff=last.copy();positions=defaults+obs[12:49] if result['response']=='capture_pose' else defaults;goal_action=(np.clip(positions,m.jnt_range[ids,0],m.jnt_range[ids,1])-defaults)/meta['action_scale'];handoff=result['handoff'];require(handoff['time_s']==5.22 and handoff['blend_s']==.5,'Handoff timing');require(np.array_equal(handoff['previous_action'],last),'Handoff previous action');require(np.array_equal(handoff['goal_action'],goal_action),'Handoff destination');require(np.array_equal(handoff['goal_target'],defaults+meta['action_scale']*goal_action),'Goal target')
                    u=float(np.clip((t-5.22)/.5,0,1));weight=u*u*(3-2*u);action=previous_before_handoff*(1-weight)+goal_action*weight;require(np.array_equal(action,x['actions'][tick]),'Posture target reconstruction');require(x['controller_modes'][tick]==result['response'],'Posture mode');posture_outputs+=1
                last=x['actions'][tick];target=defaults+meta['action_scale']*last;require(np.array_equal(target,x['targets'][tick]),'Target scale')
                if tick%32==0:
                    d.ctrl[ai]=target;mujoco.mj_forward(m,d);require(np.all(abs(d.qfrc_actuator[va])<=m.jnt_actfrcrange[ids,1]+1e-7),'Force sample cap')
        require(np.all(x['force_max']<=m.jnt_actfrcrange[ids,1]+1e-7),'Force peak cap')
        if result['termination']=='observation_complete':require(count==761 and height>=.35,'Completion')
        else:require(result['termination']=='height_termination' and height<.35,'Failure reason')
        require(result['actor_calls']==(count if result['response']=='zero' else 261),'Actor call count')
        if result['response']!='zero':require(np.array_equal(x['actions'][261],x['actions'][260]),'Initial handoff action jump')
        else:require(result['handoff'] is None,'Baseline unexpectedly holds posture')
        for key,value in metrics(x).items():require(value==result[key],'Metric '+key)
    for direction in ('left','right','straight'):
        hold,hr=traces[direction+'_zero']
        for response in ('capture_pose','nominal_pose'):
            other,orr=traces[direction+'_'+response];require(np.array_equal(hold['states'][:262],other['states'][:262]),'Pre-response state mismatch');require(np.array_equal(hold['signals'][:262],other['signals'][:262]),'Pre-response signal mismatch')
            for key in ('commands','observations','actions','targets'):require(np.array_equal(hold[key][:261],other[key][:261]),'Pre-response '+key)
            require(hr['prefix_physics_sha256']==orr['prefix_physics_sha256'],'Pre-response physics mismatch')
    regression=[]
    previous=a.run.parent/'command_response'
    if previous.exists():
        for direction in ('left','right','straight'):
            old=np.load(previous/(direction+'_zero')/'trace.npz');new=traces[direction+'_zero'][0]
            for key in old.files:require(np.array_equal(old[key],new[key]),'Previous zero baseline changed '+key)
            regression.append(direction+'_zero all old NPZ arrays exact')
    proof=dict(status='PASS',sampled_states_rebuilt=states,actor_outputs_recomputed=actors,posture_outputs_reconstructed=posture_outputs,paired_prefixes_exact=6,prefix_physics_digests_exact=6,prior_zero_baseline_regression=regression,metrics_recomputed=True,smooth_handoff_and_current_interface_checked=True,scope='Full sampled state/command/actor reconstruction and recorded peak-force caps; no physical safety or multi-seed robustness claim, no full dynamic trajectory re-simulation.')
    (a.output/'verification.json').write_text(json.dumps(proof,indent=2));(a.output/'raw_hashes.json').write_text(json.dumps({str(f.resolve()):sha(f) for f in a.run.rglob('*') if f.is_file()},indent=2));print(json.dumps(proof))

if __name__=='__main__':main()
