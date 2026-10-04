"""Held-target replay of actual PhysX support and large-leg branches."""
import argparse,hashlib,json,sys
from pathlib import Path
import mujoco
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for k in ('source','model','reference-root','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.reference_root/'mujoco'))
from mujoco_eval.policy import quat_rotate_inverse
from mujoco_eval.run_grid import resolve_joint_id
x=np.load(a.source/'responses.npz');r=json.loads((a.source/'runtime.json').read_text());checks={};peaks={}
for mode,dt in [('dt005',.005),('dt001',.001)]:
 records={k:[] for k in ('states','joint_pos','joint_vel','root_link','root_com','body_pose','foot_force','nominal_torque','ncon')};check=[];peak=[]
 for ci,case in enumerate(r['cases']):
  m=mujoco.MjModel.from_binary_path(str(a.model));m.opt.timestep=dt
  if r['contact_free']:m.opt.gravity[:]=0
  d=mujoco.MjData(m);ids=np.array([resolve_joint_id(m,n)[0] for n in r['joint_names']]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in r['joint_names']]);bids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in r['body_names']]);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
  root=x['root_link'][0,ci];d.qpos[:]=m.qpos0;d.qpos[:7]=root[:7];d.qpos[qa]=x['joint_pos'][0,ci];d.qvel[:3]=root[7:10];d.qvel[3:6]=quat_rotate_inverse(root[3:7],root[10:]);d.qvel[va]=x['joint_vel'][0,ci];d.ctrl[ai]=x['target'][ci];mujoco.mj_forward(m,d)
  sp=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,sp,0)
  c=dict(case=case['name'],com_pos=float(np.max(np.abs(d.xipos[pelvis]-x['root_com'][0,ci,:3]))),com_velocity=float(np.max(np.abs(np.r_[sp[3:],sp[:3]]-x['root_com'][0,ci,7:]))),body_pos=float(np.max(np.abs(d.xpos[bids]-x['body_pose'][0,ci,:,:3]))),kp=float(np.max(np.abs(m.actuator_gainprm[ai,0]-r['stiffness']))),kd=float(np.max(np.abs(-m.actuator_biasprm[ai,2]-r['damping']))),armature=float(np.max(np.abs(m.dof_armature[va]-r['armature']))))
  if c['com_pos']>2e-6 or c['com_velocity']>2e-6 or c['body_pos']>1e-5 or max(c['kp'],c['kd'],c['armature'])>1e-6:raise RuntimeError(str(c))
  check.append(c);rec={k:[] for k in records};pk=np.zeros(37);decimation=round(.005/dt)
  for step in range(round(.2/dt)+1):
   if step%decimation==0:
    mujoco.mj_forward(m,d);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,sp,0);offset=d.xipos[pelvis]-d.xpos[pelvis];fv=np.zeros((2,3))
    for j in range(d.ncon):
     contact=d.contact[j];f=np.zeros(6);mujoco.mj_contactForce(m,d,j,f);fw=contact.frame.reshape(3,3).T@f[:3]
     # Contact-frame normal points geom1 -> geom2; force acts on geom2.
     for k,b in enumerate(feet):
      if m.geom_bodyid[contact.geom[1]]==b:fv[k]+=fw
      if m.geom_bodyid[contact.geom[0]]==b:fv[k]-=fw
    rec['states'].append(np.r_[d.qpos,d.qvel].copy());rec['joint_pos'].append(d.qpos[qa].copy());rec['joint_vel'].append(d.qvel[va].copy());rec['root_link'].append(np.r_[d.xpos[pelvis],d.xquat[pelvis],sp[3:]-np.cross(sp[:3],offset),sp[:3]]);rec['root_com'].append(np.r_[d.xipos[pelvis],d.xquat[pelvis],sp[3:],sp[:3]]);rec['body_pose'].append(np.c_[d.xpos[bids],d.xquat[bids]]);rec['foot_force'].append(fv);rec['nominal_torque'].append(d.qfrc_actuator[va].copy());rec['ncon'].append(d.ncon)
   if not np.all(np.isfinite(d.qpos)) or any(w.number for w in d.warning) or (r['contact_free'] and d.ncon):raise RuntimeError('Invalid branch')
   if step<round(.2/dt):mujoco.mj_step(m,d);pk=np.maximum(pk,np.abs(d.qfrc_actuator[va]))
  if np.any(pk>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap')
  peak.append(pk.tolist())
  for k in records:records[k].append(rec[k])
 output={k:np.swapaxes(np.array(v),0,1) for k,v in records.items()};output.update(time=x['time'],target=x['target']);np.savez_compressed(a.output/f'{mode}.npz',**output);checks[mode]=check;peaks[mode]=peak;print('LEG_CONTACT_MUJOCO',mode,len(r['cases']),flush=True)
files=[Path(__file__).resolve(),a.model]+[a.source/f for f in ('responses.npz','runtime.json','inputs.json')]+[a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py']
(a.output/'inputs.json').write_text(json.dumps({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2));(a.output/'complete.json').write_text(json.dumps(dict(initial_checks=checks,peak_forces=peaks,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,contact_free=r['contact_free'],note='Held-target 200ms reset branches. Mu contact force instantaneous post-forward; PhysX contact sensor step-averaged impulse/dt. Compare integrated forces, not pointwise exact equivalence. Friction Mu .8 vs PhysX static .8/dynamic .6.'),indent=2))
