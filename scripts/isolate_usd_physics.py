"""Isolate sagittal asymmetry, contact solver, integrator, and reflected-policy effects."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.reference_root/'mujoco'))
from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
from mujoco_eval.run_grid import resolve_joint_id,read_base_state
meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text());defaults=np.array(meta['default_joint_pos'])
policy_path=a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz';actor=TorchActorPolicy(policy_path)
model_path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb'
files=[meta_path,policy_path,model_path,Path(__file__).resolve(),a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py']
hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2))
audit_path=Path('results/flat_baseline/20261002/usd_physics_isolation/bilateral_audit.json');audit=json.loads(audit_path.read_text());perm=np.array(audit['mirror_permutation']);sign=np.array(audit['mirror_signs'],dtype=np.float32)
hashes[str(audit_path.resolve())]=hashlib.sha256(audit_path.read_bytes()).hexdigest();(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2))
def reflect_obs(obs):
    x=obs.copy();x[:3]*=[1,-1,1];x[3:6]*=[-1,1,-1];x[6:9]*=[1,-1,1];x[9:12]*=[1,-1,-1]
    for start in (12,49,86):x[start:start+37]=sign*obs[start:start+37][perm]
    return x
rng=np.random.default_rng(42)
for _ in range(100):
    x=rng.normal(size=123).astype(np.float32)
    if not np.array_equal(reflect_obs(reflect_obs(x)),x):raise RuntimeError('Reflection is not involutive')
rows=[]
for variant in ('baseline','refresh_constants','central_com_sym','central_inertia_sym','contact_solref01','cone_elliptic','integrator_implicit','mirror_actor'):
 for speed in (.5,1.):
  for yaw in (0.,):
   model=mujoco.MjModel.from_binary_path(str(model_path));original=mujoco.MjModel.from_binary_path(str(model_path));collidable=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
   central=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,r['body']) for r in audit['central_bodies']]
   base_data=mujoco.MjData(model)
   base_ids=np.array([resolve_joint_id(model,n)[0] for n in meta['joint_names']]);base_data.qpos[model.jnt_qposadr[base_ids]]=defaults;mujoco.mj_forward(model,base_data)
   S=np.diag([1.,-1.,1.]);changed_settings={}
   if variant=='central_com_sym':
    for body in central:
     R=base_data.xmat[body].reshape(3,3);local=model.body_ipos[body].copy();world=R@local;world[1]=0.;model.body_ipos[body]=R.T@world
    changed_settings['parameter']='central body COM sagittal offsets removed; mass unchanged'
   if variant=='central_inertia_sym':
    for body in central:
     R=base_data.xmat[body].reshape(3,3);Q=base_data.ximat[body].reshape(3,3);I=Q@np.diag(model.body_inertia[body])@Q.T;I=.5*(I+S@I@S);local=R.T@I@R;values,vectors=np.linalg.eigh(local)
     if np.linalg.det(vectors)<0:vectors[:,0]*=-1
     quat=np.zeros(4);mujoco.mju_mat2Quat(quat,vectors.flatten());model.body_inertia[body]=values;model.body_iquat[body]=quat
    changed_settings['parameter']='central inertia antisymmetric components removed; COM unchanged'
   if variant in ('refresh_constants','central_com_sym','central_inertia_sym'):mujoco.mj_setConst(model,mujoco.MjData(model))
   if variant=='contact_solref01':model.geom_solref[collidable,0]=.01;changed_settings['parameter']='contact solref time constant .02 -> .01'
   if variant=='cone_elliptic':model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;changed_settings['parameter']='pyramidal -> elliptic friction cone'
   if variant=='integrator_implicit':model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT;changed_settings['parameter']='implicitfast -> implicit'
   if variant=='mirror_actor':changed_settings['parameter']='sagittal reflection wrapper around same actor weights; no model change'
   changed=[name for name in dir(model) if isinstance(getattr(model,name),np.ndarray) and not np.array_equal(getattr(model,name),getattr(original,name))]
   allowed={'baseline':set(),'refresh_constants':{'actuator_acc0','body_invweight0','dof_invweight0','dof_M0','dof_length','light_poscom0'},'mirror_actor':set(),'central_com_sym':{'body_ipos','actuator_acc0','dof_length','light_poscom0','body_invweight0','dof_invweight0','dof_M0'},'central_inertia_sym':{'body_inertia','body_iquat','actuator_acc0','dof_length','light_poscom0','body_invweight0','dof_invweight0','dof_M0'},'contact_solref01':{'geom_solref'},'cone_elliptic':set(),'integrator_implicit':set()}
   if not set(changed)<=allowed[variant]:raise RuntimeError(f'Unexpected model change {variant}: {changed}')
   start=0.;dt=float(model.opt.timestep);decimation=round(.02/dt);total=round(20/dt)
   data=mujoco.MjData(model);ids=np.array([resolve_joint_id(model,n)[0] for n in meta['joint_names']]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids];ai=np.array([next(i for i in range(model.nu) if model.actuator_trnid[i,0]==jid) for jid in ids])
   data.qpos[:]=model.qpos0;data.qpos[:3]=[0,0,.74];data.qpos[3:7]=[np.cos(start/2),0,0,np.sin(start/2)];data.qpos[qa]=defaults
   command=np.array([speed,0,yaw]);last=np.zeros(37);target=defaults.copy();times=[];states=[];signals=[];actions=[];peak=np.zeros(37);fell=False;bad=False;steps=0
   for step in range(total+1):
    if step%decimation==0:
     mujoco.mj_forward(model,data);quat,lin,ang,height=read_base_state(model,data,'pelvis');blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang);gravity=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-gravity[2],-1,1))
     heading=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
     times.append(float(data.time));states.append(np.r_[data.qpos,data.qvel].copy());signals.append(np.r_[data.qpos[:3],quat,lin,ang,blin,bang,height,tilt,command,heading])
     if height<.35:fell=True;break
     if step==total:break
     obs=build_policy_observation(quat,lin,ang,command,data.qpos[qa],data.qvel[va],defaults,last)
     if not np.array_equal(obs[9:12],command.astype(np.float32)):raise RuntimeError('Command mismatch')
     last=(sign*actor.act(reflect_obs(obs))[perm] if variant=='mirror_actor' else actor.act(obs)).astype(float);actions.append(last.copy());target=defaults+meta['action_scale']*last
    data.ctrl[ai]=target;mujoco.mj_step(model,data);steps+=1;peak=np.maximum(peak,np.abs(data.qfrc_actuator[va]))
    if not np.all(np.isfinite(data.qpos)) or any(data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):bad=True;break
   time=np.array(times);s=np.array(signals);h=np.unwrap(s[:,24]);mask=time>=time[-1]-10
   row=dict(variant=variant,speed=speed,yaw_command=yaw,start_heading_rad=start,physics_dt=dt,changed_model_arrays=changed,changed_settings=changed_settings,model_options=dict(cone=int(model.opt.cone),integrator=int(model.opt.integrator)),elapsed_s=steps*dt,physical_steps=steps,actions=len(actions),fell=int(fell),numerical_failure=int(bad),mean_world_wz=float(s[mask,12].mean()),mean_body_wz=float(s[mask,18].mean()),heading_endpoint_rate=float((h[-1]-h[np.flatnonzero(mask)[0]])/(time[-1]-time[np.flatnonzero(mask)[0]])),heading_slope=float(np.polyfit(time[mask],h[mask],1)[0]),forward_rmse=float(np.sqrt(np.mean((s[mask,13]-speed)**2))),max_tilt_deg=float(np.rad2deg(s[:,20].max())))
   stem=f'{variant}_vx{speed:g}_yaw{yaw:+.1f}';np.savez_compressed(a.output/f'{stem}.npz',time=time,states=states,signals=signals,actions=actions,force_max=peak);rows.append(row)
   print('YAW_PROBE',variant,speed,yaw,row['elapsed_s'],round(row['heading_slope'],5),round(row['mean_world_wz'],5),flush=True)
for path,expected in hashes.items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Input changed')
(a.output/'complete.json').write_text(json.dumps(dict(results=rows,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,protocol=dict(duration=20,scoring='final10s',policy_dt=.02,actor='locally trained1499; unchanged',task='held angular-rate commands; no heading feedback',configuration='independent in-memory variants; original model and defaults unchanged; symmetry interventions are artificial diagnostics, mirror actor changes controller behavior without changing weights',replication='deterministic fixed initial conditions, no stochastic seed coverage')),indent=2))
