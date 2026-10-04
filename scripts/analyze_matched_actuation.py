"""Verify matched response records and localize finger-limit effects on yaw."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import mujoco
import numpy as np

def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.run_root
 sys.path.insert(0,str(a.reference_root/'mujoco'))
 from mujoco_eval.policy import quat_rotate_inverse,build_policy_observation,TorchActorPolicy
 from mujoco_eval.run_grid import resolve_joint_id
 rt=json.loads((root/'matched_actuation_isaac/runtime.json').read_text());x=np.load(root/'matched_actuation_isaac/responses.npz');names=rt['joint_names'];meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());defaults=np.array(meta['default_joint_pos']);modelpath=root/'usd_physical_models/g1_usd_physical_position.mjb';finger=[i for i,n in enumerate(names) if any(n.endswith('_'+s+'_joint') for s in ('five','six','three','four','zero','one','two'))];legs=[i for i,n in enumerate(names) if any(s in n for s in ('hip','knee','ankle'))];metrics=[];checks=[];inputs={}
 folders=('matched_actuation_isaac','matched_actuation_mujoco','matched_actuation_limits','walking_finger_constraints','finger_limit_localization')
 for folder in folders:
  inp=json.loads((root/folder/'inputs.json').read_text())
  for file,h in inp.items():require(sha(file)==h,'Input changed: '+file)
  inputs[folder]=inp
  for name in ('inputs.json','complete.json'):(a.output/f'{folder}_{name}').write_bytes((root/folder/name).read_bytes())
 require(np.abs(x['contact_force']).max()==0,'ActualIsaac contact present');require(np.max(np.abs(x['joint_pos'][0]-x['requested_joint_pos']))<1e-7 and np.max(np.abs(x['joint_vel'][0]-x['requested_joint_vel']))<1e-7,'Isaac initial joint state mismatch');require(np.allclose(x['time'],np.arange(41)*.005),'Isaac timing differs')
 m=mujoco.MjModel.from_binary_path(str(modelpath));ids=np.array([resolve_joint_id(m,n)[0] for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);bids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in rt['body_names']]);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
 for prop,value in [('stiffness',m.actuator_gainprm[ai,0]),('damping',-m.actuator_biasprm[ai,2]),('armature',m.dof_armature[va]),('effort_limits',m.actuator_forcerange[ai,1])]:require(np.allclose(rt[prop],value,atol=1e-6),'Actuator property mismatch '+prop)
 require(np.allclose(rt['joint_friction_coeff'],0) and np.allclose(m.dof_frictionloss,0),'Friction contract differs')
 for variant in ('matched_actuation_mujoco','matched_actuation_limits'):
  for mode,dt in [('matched_dt005',.005),('substep_dt001',.001)]:
   y=np.load(root/variant/f'{mode}.npz');m=mujoco.MjModel.from_binary_path(str(modelpath));m.opt.gravity[:]=0;m.opt.timestep=dt
   if variant=='matched_actuation_limits':m.jnt_solref[ids[finger],0]=.002
   require(np.array_equal(y['target'],x['target']) and np.array_equal(y['time'],x['time']),'Targets/timestamps changed')
   d=mujoco.MjData(m);signalerr=forceerr=velocityerr=poseerr=0.
   for i,case in enumerate(rt['cases']):
    for k in range(41):
     state=y['states'][k,i];d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[ai]=y['target'][i];mujoco.mj_forward(m,d);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,v,0);linkvel=v[3:]-np.cross(v[:3],d.xipos[pelvis]-d.xpos[pelvis]);link=np.r_[d.xpos[pelvis],d.xquat[pelvis],linkvel,v[:3]]
     require(d.ncon==0 and np.isfinite(state).all(),'Invalidcontact-free state');signalerr=max(signalerr,float(np.max(np.abs(np.r_[d.qpos[qa],d.qvel[va]]-np.r_[y['joint_pos'][k,i],y['joint_vel'][k,i]]))));forceerr=max(forceerr,float(np.max(np.abs(d.qfrc_actuator[va]-y['nominal_torque'][k,i]))));velocityerr=max(velocityerr,float(np.max(np.abs(np.r_[v[3:],v[:3]]-y['root_com_velocity'][k,i]))));poseerr=max(poseerr,float(np.max(np.abs(link-y['root_link'][k,i]))))
     require(np.all(np.abs(d.qfrc_actuator[va])<=m.jnt_actfrcrange[ids,1]+1e-7),'Actuation cap exceeded')
    e=y['joint_pos'][:,i]-x['joint_pos'][:,i];ei=np.max(np.abs(e),axis=0);j=int(ei.argmax());target_violation=np.maximum(m.jnt_range[ids,0]-x['target'][i],0)+np.maximum(x['target'][i]-m.jnt_range[ids,1],0)
    row=dict(variant=variant,mode=mode,case=case['name'],kind='walking_state' if case['source_sample'] is not None else 'small_step',excited_joint=case['joint'],max_joint_error_rad=float(ei.max()),worst_joint=names[j],max_leg_error_rad=float(ei[legs].max()),max_finger_error_rad=float(ei[finger].max()),target_outside_limits=int(np.count_nonzero(target_violation>1e-6)),excited_joint_error_rad=float(ei[names.index(case['joint'])]) if case['joint'] else None)
    metrics.append(row)
   require(max(signalerr,forceerr,velocityerr,poseerr)<1e-10,'StoredMuJoCo signals differ');checks.append(dict(variant=variant,mode=mode,states=len(rt['cases'])*41,signal_max_error=signalerr,torque_max_error=forceerr,root_com_velocity_max_error=velocityerr,root_link_max_error=poseerr,raw_sha256=sha(root/variant/f'{mode}.npz')))
 actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');walkrows=[];walkchecks=[]
 for folder in ('walking_finger_constraints','finger_limit_localization'):
  complete=json.loads((root/folder/'complete.json').read_text())
  for row in complete['results']:
   variant=row['variant'];speed=row['speed'];stem=f'{variant}_vx{speed:g}_yaw+0.0';t=np.load(root/folder/f'{stem}.npz');m=mujoco.MjModel.from_binary_path(str(modelpath));original=mujoco.MjModel.from_binary_path(str(modelpath));selected=names if variant=='finger_limit002' else ['left_four_joint','right_four_joint','left_six_joint','right_six_joint'] if variant=='active4' else [variant]
   selection=[i for i,n in enumerate(names) if i in finger and (variant=='finger_limit002' or n in selected)]
   if variant!='baseline':m.jnt_solref[ids[selection],0]=.002
   changed=[n for n in dir(m) if isinstance(getattr(m,n),np.ndarray) and not np.array_equal(getattr(m,n),getattr(original,n))];require(changed==([] if variant=='baseline' else ['jnt_solref']),'Unexpectedmodel mutation')
   d=mujoco.MjData(m);signalerr=actionerr=0.;sampled=set(np.linspace(0,999,32,dtype=int));rebuilt=[]
   for k,state in enumerate(t['states']):
    d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);q=d.xquat[pelvis];R=d.xmat[pelvis].reshape(3,3);heading=np.arctan2(R[1,0],R[0,0]);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,v,0);ang,lin=v[:3],v[3:];blin=quat_rotate_inverse(q,lin);bang=quat_rotate_inverse(q,ang);g=quat_rotate_inverse(q,[0,0,-1]);tilt=np.arccos(np.clip(-g[2],-1,1));cmd=np.array([speed,0,0]);rebuilt.append(np.r_[d.qpos[:3],q,lin,ang,blin,bang,d.xpos[pelvis,2],tilt,cmd,heading])
    if k in sampled:
     prev=t['actions'][k-1] if k else np.zeros(37);obs=build_policy_observation(q,lin,ang,cmd,d.qpos[qa],d.qvel[va],defaults,prev);actionerr=max(actionerr,float(np.max(np.abs(actor.act(obs)-t['actions'][k]))))
   rebuilt=np.array(rebuilt);signalerr=float(np.max(np.abs(rebuilt-t['signals'])));require(signalerr<1e-10 and actionerr<1e-6,'Walking signals/actions differ');require(len(t['time'])==1001 and len(t['actions'])==1000 and row['physical_steps']==20000 and not row['fell'] and not row['numerical_failure'],'Incompletewalking');require(np.allclose(np.diff(t['time']),.02,atol=1e-10) and abs(t['time'][-1]-20)<1e-8,'Walkingtime mismatch');require(np.all(t['force_max']<=m.jnt_actfrcrange[ids,1]+1e-7),'Walkingcap exceeded')
   mask=t['time']>=t['time'][-1]-10;tt=t['time'][mask];h=np.unwrap(rebuilt[:,24]);center=tt-tt.mean();slope=float(center@(h[mask]-h[mask].mean())/(center@center));require(abs(slope-row['heading_slope'])<1e-10,'Walkingscore mismatch')
   base=next(r for r in complete['results'] if r['variant']=='baseline' and r['speed']==speed);walkrows.append(dict(folder=folder,variant=variant,speed=speed,heading_slope=slope,relative_bias_reduction=1-abs(slope)/abs(base['heading_slope']),forward_rmse=row['forward_rmse'],max_tilt_deg=row['max_tilt_deg'],changed_joint_count=len(selection) if variant!='baseline' else 0));walkchecks.append(dict(case=folder+'/'+stem,states=1001,actor_inputs=32,signal_max_error=signalerr,actor_max_error=actionerr,raw_sha256=sha(root/folder/f'{stem}.npz')))
 consistency=[]
 for speed in (.5,1.):
  for left,right,label in [('baseline','baseline','baseline'),('finger_limit002','finger_limit002','all14'),('finger_limit002','active4','all14_vs_active4')]:
   a1=np.load(root/'walking_finger_constraints'/f'{left}_vx{speed:g}_yaw+0.0.npz');b1=np.load(root/'finger_limit_localization'/f'{right}_vx{speed:g}_yaw+0.0.npz');equal=all(np.array_equal(a1[k],b1[k]) for k in a1.files);require(equal,'Repeated/groupcomparison differs');consistency.append(dict(speed=speed,comparison=label,all_arrays_equal=True))
 writecsv(a.output/'matched_response.csv',metrics);writecsv(a.output/'finger_localization.csv',walkrows);(a.output/'isaac_runtime.json').write_bytes((root/'matched_actuation_isaac/runtime.json').read_bytes());(a.output/'verification.json').write_text(json.dumps(dict(matched_cases=checks,walking_cases=walkchecks,matched_reconstructed_states=42*41*4,walking_reconstructed_states=18*1001,actor_evaluations=18*32,consistency=consistency,source_hashes_verified=True,isaac_contact_force_max=0.,limitations='Matched suspendedresponses and20s zero-rate walking only. Joint-limit mismatch contribution shown, not sole yawcause; no symmetricweightcopy ordefaultpromotion.'),indent=2));print('PASS:6888matchedMu states,18018walking states,576actorinputs,inputhashes,caps,onlyfingerlimits changed; active4/all14 arraysidentical')
if __name__=='__main__':main()
