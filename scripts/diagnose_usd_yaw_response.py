"""Fixed-policy yaw-response and isolated physical-configuration probes."""
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
rows=[]
for variant in ('baseline','dt_half','friction06','start90'):
 for speed in (.5,1.):
  for yaw in ((-.2,-.1,0.,.1,.2) if variant=='baseline' else (0.,)):
   model=mujoco.MjModel.from_binary_path(str(model_path));original=mujoco.MjModel.from_binary_path(str(model_path));collidable=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
   if variant=='dt_half':model.opt.timestep=.0005
   if variant=='friction06':model.geom_friction[collidable,0]=.6
   # Fail if a single-variable probe unexpectedly changes any other compiled array.
   changed=[]
   for name in dir(model):
    left=getattr(model,name);right=getattr(original,name)
    if isinstance(left,np.ndarray) and not np.array_equal(left,right):changed.append(name)
   expected=['geom_friction'] if variant=='friction06' else []
   if changed!=expected:raise RuntimeError(f'Unexpected model array mutation {changed}')
   start=np.pi/2 if variant=='start90' else 0.;dt=float(model.opt.timestep);decimation=round(.02/dt);total=round(20/dt)
   if abs(decimation*dt-.02)>1e-12:raise RuntimeError('Policy rate differs')
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
     last=actor.act(obs).astype(float);actions.append(last.copy());target=defaults+meta['action_scale']*last
    data.ctrl[ai]=target;mujoco.mj_step(model,data);steps+=1;peak=np.maximum(peak,np.abs(data.qfrc_actuator[va]))
    if not np.all(np.isfinite(data.qpos)) or any(data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):bad=True;break
   time=np.array(times);s=np.array(signals);h=np.unwrap(s[:,24]);mask=time>=time[-1]-10
   row=dict(variant=variant,speed=speed,yaw_command=yaw,start_heading_rad=start,physics_dt=dt,changed_model_arrays=changed,elapsed_s=steps*dt,physical_steps=steps,actions=len(actions),fell=int(fell),numerical_failure=int(bad),mean_world_wz=float(s[mask,12].mean()),mean_body_wz=float(s[mask,18].mean()),heading_endpoint_rate=float((h[-1]-h[np.flatnonzero(mask)[0]])/(time[-1]-time[np.flatnonzero(mask)[0]])),heading_slope=float(np.polyfit(time[mask],h[mask],1)[0]),forward_rmse=float(np.sqrt(np.mean((s[mask,13]-speed)**2))),max_tilt_deg=float(np.rad2deg(s[:,20].max())))
   stem=f'{variant}_vx{speed:g}_yaw{yaw:+.1f}';np.savez_compressed(a.output/f'{stem}.npz',time=time,states=states,signals=signals,actions=actions,force_max=peak);rows.append(row)
   print('YAW_PROBE',variant,speed,yaw,row['elapsed_s'],round(row['heading_slope'],5),round(row['mean_world_wz'],5),flush=True)
for path,expected in hashes.items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Input changed')
(a.output/'complete.json').write_text(json.dumps(dict(results=rows,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,protocol=dict(duration=20,scoring='final10s',policy_dt=.02,actor='locally trained1499; unchanged',task='held angular-rate commands; no heading feedback',configuration='independent in-memory variants; original model and defaults unchanged',replication='deterministic fixed initial conditions, no stochastic seed coverage')),indent=2))
