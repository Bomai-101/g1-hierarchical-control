"""Separate finger-constraint physics from eight policy observation slots."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(a.reference_root/'mujoco'))
 from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
 from mujoco_eval.run_grid import resolve_joint_id,read_base_state
 meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);policy_path=a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz';actor=TorchActorPolicy(policy_path);model_path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb'
 fingers=['left_four_joint','right_four_joint','left_six_joint','right_six_joint'];fi=np.array([names.index(n) for n in fingers]);slots=np.r_[12+fi,49+fi]
 files=[Path(__file__),meta_path,policy_path,model_path,a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py'];hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2));rows=[]
 for physical,observation in [('soft','self'),('tight','self'),('soft','paired'),('tight','paired')]:
  paired='tight' if physical=='soft' else 'soft';variant=f'{physical}_self' if observation=='self' else f'{physical}_paired_{paired}'
  for speed in (.5,1.):
   models=[];datas=[]
   for mode in (physical,paired):
    m=mujoco.MjModel.from_binary_path(str(model_path));original=mujoco.MjModel.from_binary_path(str(model_path));ids=np.array([resolve_joint_id(m,n)[0] for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==jid) for jid in ids])
    if mode=='tight':m.jnt_solref[ids[fi],0]=.002
    changed=[n for n in dir(m) if isinstance(getattr(m,n),np.ndarray) and not np.array_equal(getattr(m,n),getattr(original,n))]
    if changed!=(['jnt_solref'] if mode=='tight' else []):raise RuntimeError('Unexpected physical mutation')
    d=mujoco.MjData(m);d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,.74];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=defaults;models.append(m);datas.append(d)
   last=np.zeros(37);target=defaults.copy();command=np.array([speed,0,0]);time=[];states=[];paired_states=[];signals=[];paired_signals=[];obs_actual=[];obs_used=[];actions=[];targets=[];peaks=np.zeros((2,37));fell=False;paired_fell=False;bad=False;steps=0
   for step in range(20001):
    if step%20==0:
     samples=[]
     for m,d in zip(models,datas):
      mujoco.mj_forward(m,d);q,lin,ang,height=read_base_state(m,d,'pelvis');blin=quat_rotate_inverse(q,lin);bang=quat_rotate_inverse(q,ang);g=quat_rotate_inverse(q,[0,0,-1]);tilt=np.arccos(np.clip(-g[2],-1,1));heading=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));samples.append((q,lin,ang,np.r_[d.qpos[:3],q,lin,ang,blin,bang,height,tilt,command,heading]))
     time.append(float(datas[0].time));states.append(np.r_[datas[0].qpos,datas[0].qvel].copy());paired_states.append(np.r_[datas[1].qpos,datas[1].qvel].copy());signals.append(samples[0][3]);paired_signals.append(samples[1][3]);paired_fell|=samples[1][3][19]<.35
     if samples[0][3][19]<.35:fell=True;break
     if step==20000:break
     q,lin,ang,_=samples[0];raw=build_policy_observation(q,lin,ang,command,datas[0].qpos[qa],datas[0].qvel[va],defaults,last);used=raw.copy()
     if observation=='paired':
      used[12+fi]=(datas[1].qpos[qa[fi]]-defaults[fi]).astype(np.float32);used[49+fi]=datas[1].qvel[va[fi]].astype(np.float32)
     outside=np.ones(123,dtype=bool);outside[slots]=False
     if not np.array_equal(used[outside],raw[outside]) or not np.array_equal(used[9:12],command.astype(np.float32)):raise RuntimeError('Observation intervention changed other fields')
     obs_actual.append(raw.copy());obs_used.append(used.copy());last=actor.act(used).astype(float);actions.append(last.copy());target=defaults+meta['action_scale']*last;targets.append(target.copy())
    for index,(m,d) in enumerate(zip(models,datas)):
     d.ctrl[ai]=target;mujoco.mj_step(m,d);peaks[index]=np.maximum(peaks[index],np.abs(d.qfrc_actuator[va]))
     if not np.isfinite(d.qpos).all() or any(d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):bad=True
    # Conditional paired model: discard drift of all non-finger states.
    # Preserve only the four alternate finger positions and velocities.
    keep_q=datas[1].qpos[qa[fi]].copy();keep_v=datas[1].qvel[va[fi]].copy()
    datas[1].qpos[:]=datas[0].qpos;datas[1].qvel[:]=datas[0].qvel
    datas[1].qpos[qa[fi]]=keep_q;datas[1].qvel[va[fi]]=keep_v
    steps+=1
    if bad:break
   s=np.array(signals);ps=np.array(paired_signals);t=np.array(time);mask=t>=t[-1]-10;h=np.unwrap(s[:,24]);actual=np.array(obs_actual);used=np.array(obs_used)
   row=dict(variant=variant,physical=physical,observation=observation,paired_physical=paired,speed=speed,elapsed_s=steps*.001,physical_steps=steps,actions=len(actions),fell=int(fell),paired_fell=int(paired_fell),numerical_failure=int(bad),heading_slope=float(np.polyfit(t[mask],h[mask],1)[0]),mean_world_wz=float(s[mask,12].mean()),forward_rmse=float(np.sqrt(np.mean((s[mask,13]-speed)**2))),mean_body_vx=float(s[mask,13].mean()),max_tilt_deg=float(np.rad2deg(s[:,20].max())),paired_max_tilt_deg=float(np.rad2deg(ps[:,20].max())),max_paired_root_position_difference_m=float(np.max(np.linalg.norm(s[:,:3]-ps[:,:3],axis=1))),max_finger_observation_intervention=float(np.max(np.abs(actual[:,slots]-used[:,slots]))))
   stem=f'{variant}_vx{speed:g}';np.savez_compressed(a.output/f'{stem}.npz',time=t,states=states,paired_states=paired_states,signals=s,paired_signals=ps,obs_actual=obs_actual,obs_used=obs_used,actions=actions,targets=targets,force_max=peaks);rows.append(row);print('FINGER_PATH',variant,speed,steps,round(row['heading_slope'],6),round(row['forward_rmse'],6),'paired_fell',paired_fell,flush=True)
 for path,h in hashes.items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise RuntimeError('Input changed')
 (a.output/'complete.json').write_text(json.dumps(dict(results=rows,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,protocol=dict(fingers=fingers,observation_slots=slots.tolist(),duration=20,physics_dt=.001,policy_dt=.02,score='last10s',weight='locallytrained1499,unchanged',command='vx0.5/1,vy0,wz0;noheadingfeedbackoroffset',intervention='2x2 physical soft/tight and actual/paired fourfinger position/velocity. Pairedplant uses oppositeconstraints and identical37targets at every physicalstep; after eachstep reset allnon-four-finger qpos/qvel to mainstate, retainingfourfinger states only. Eightobs slots only; mainbase/otherjoints/all37previousactions unchanged. Artificialdiagnostic,notdefaultdeployment;no passive-monitor equivalence claim',limitation='Conditional pairedfingerstate generator with nonfingerstate transplantation; artificial diagnostic, not a physical independent secondrobot or momentum-conserving fullcounterfactual. Noadditivepercentagecausal attribution.')),indent=2))
if __name__=='__main__':main()
