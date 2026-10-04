"""Replay contact-free actual PhysX initial states and held targets in MuJoCo."""
import argparse,json,hashlib,sys
from pathlib import Path
import mujoco
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
folder=a.run_root/'matched_actuation_isaac';trace_path=folder/'responses.npz';x=np.load(trace_path);runtime=json.loads((folder/'runtime.json').read_text());names=runtime['joint_names'];model_path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb';sys.path.insert(0,str(a.reference_root/'mujoco'))
from mujoco_eval.policy import quat_rotate_inverse
from mujoco_eval.run_grid import resolve_joint_id
initial_checks={};max_caps={};outputs={}
for mode,dt in [('matched_dt005',.005),('substep_dt001',.001)]:
 records={key:[] for key in ('states','joint_pos','joint_vel','root_link','root_com_pos','root_com_velocity','body_pose','nominal_torque')};checks=[];peaks=[]
 for case in range(len(runtime['cases'])):
  m=mujoco.MjModel.from_binary_path(str(model_path));m.opt.gravity[:]=0;m.opt.timestep=dt;d=mujoco.MjData(m);ids=np.array([resolve_joint_id(m,n)[0] for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');bodies=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in runtime['body_names']]);root=x['root_link'][0,case];d.qpos[:]=m.qpos0;d.qpos[:7]=root[:7];d.qpos[qa]=x['joint_pos'][0,case];d.qvel[:3]=root[7:10];d.qvel[3:6]=quat_rotate_inverse(root[3:7],root[10:13]);d.qvel[va]=x['joint_vel'][0,case];d.ctrl[ai]=x['target'][case];mujoco.mj_forward(m,d)
  sp=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,sp,0)
  checks.append(dict(case=runtime['cases'][case]['name'],base_com_pos_error_m=float(np.max(np.abs(d.xipos[pelvis]-x['root_com'][0,case,:3]))),base_com_velocity_error=float(np.max(np.abs(np.r_[sp[3:],sp[:3]]-x['root_com'][0,case,7:]))),body_position_error_m=float(np.max(np.abs(d.xpos[bodies]-x['body_pose'][0,case,:,:3])))))
  if checks[-1]['base_com_velocity_error']>2e-6 or checks[-1]['base_com_pos_error_m']>2e-6 or checks[-1]['body_position_error_m']>1e-5:raise RuntimeError('Initial state conversion differs')
  record={key:[] for key in records};peak=np.zeros(37);decimation=round(.005/dt)
  for step in range(round(.2/dt)+1):
   if step%decimation==0:
    mujoco.mj_forward(m,d);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,sp,0);offset=d.xipos[pelvis]-d.xpos[pelvis];linkvel=sp[3:]-np.cross(sp[:3],offset)
    record['states'].append(np.r_[d.qpos,d.qvel].copy());record['joint_pos'].append(d.qpos[qa].copy());record['joint_vel'].append(d.qvel[va].copy());record['root_link'].append(np.r_[d.xpos[pelvis],d.xquat[pelvis],linkvel,sp[:3]]);record['root_com_pos'].append(d.xipos[pelvis].copy());record['root_com_velocity'].append(np.r_[sp[3:],sp[:3]]);record['body_pose'].append(np.c_[d.xpos[bodies],d.xquat[bodies]]);record['nominal_torque'].append(d.qfrc_actuator[va].copy())
   if d.ncon or not np.all(np.isfinite(d.qpos)) or any(w.number for w in d.warning):raise RuntimeError('Invalid contact-free response')
   if step<round(.2/dt):mujoco.mj_step(m,d);peak=np.maximum(peak,np.abs(d.qfrc_actuator[va]))
  if np.any(peak>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap exceeded')
  peaks.append(peak.tolist())
  for key in records:records[key].append(record[key])
 output={key:np.swapaxes(np.array(values),0,1) for key,values in records.items()};output.update(time=x['time'],target=x['target']);np.savez_compressed(a.output/f'{mode}.npz',**output);initial_checks[mode]=checks;max_caps[mode]=peaks;outputs[mode]=str((a.output/f'{mode}.npz').resolve());print('MATCHED_MUJOCO',mode,len(runtime['cases']),41,flush=True)
files=[Path(__file__),trace_path,folder/'runtime.json',folder/'inputs.json',model_path,a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py'];(a.output/'inputs.json').write_text(json.dumps({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2));(a.output/'complete.json').write_text(json.dumps(dict(initial_checks=initial_checks,peak_forces=max_caps,outputs=outputs,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,scope='Contact-free,gravityoff; identicalactualIsaac initial joint/velocity/target. Mu freejoint angularqvel islocal; initial rootlink linearvelocity isworld. COM orientations are not compared due differing inertial frame conventions. Bothmatching5ms and1ms substeps sampled5ms.'),indent=2))
