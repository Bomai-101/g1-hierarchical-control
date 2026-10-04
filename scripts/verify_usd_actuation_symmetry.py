"""Reconstruct saved contact-free joint responses and torque measurements."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=json.loads(a.probe.read_text())
for file,sha in report['inputs'].items():
 if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:raise RuntimeError('Probe input changed')
paths=list(report['inputs']);modelpath=next(p for p in paths if p.endswith('.mjb'));metapath=next(p for p in paths if p.endswith('isaac_metadata.json'));auditpath=next(p for p in paths if p.endswith('bilateral_audit.json'));meta=json.loads(Path(metapath).read_text());audit=json.loads(Path(auditpath).read_text());perm=np.array(audit['mirror_permutation']);sign=np.array(audit['mirror_signs']);evidence=[]
for row in report['results']:
 rebuilt=[]
 for file,sha in row['trace_hashes'].items():
  if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:raise RuntimeError('Probe trace changed')
  trace=np.load(file);m=mujoco.MjModel.from_binary_path(modelpath);m.opt.gravity[:]=0
  if row['integrator']=='implicit':m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT
  ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in meta['joint_names']]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in meta['joint_names']]);d=mujoco.MjData(m);signal=[];force=[]
  for state in trace['states']:
   d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[ai]=trace['target'];mujoco.mj_forward(m,d)
   if d.ncon or not np.all(np.isfinite(state)):raise RuntimeError('Invalid contact-free state')
   signal.append(np.r_[d.qpos[qa]-meta['default_joint_pos'],d.qvel[va]]);force.append(d.qfrc_actuator[va].copy())
  signal=np.array(signal);force=np.array(force);signalerr=float(np.max(np.abs(signal-trace['signals'])));forceerr=float(np.max(np.abs(force-trace['forces'])))
  if signalerr>1e-12 or forceerr>1e-10 or len(signal)!=201 or not np.allclose(trace['time'],np.arange(201)*.001):raise RuntimeError('Response signal/time mismatch')
  if np.any(np.abs(force)>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Torque cap exceeded')
  rebuilt.append((signal,force));evidence.append(dict(trace=file,states=201,signal_max_error=signalerr,force_max_error=forceerr,sha256=sha))
 left,right=rebuilt
 errors=dict(max_mirrored_joint_position_difference_rad=float(np.max(np.abs(right[0][:,:37]-left[0][:,:37][:,perm]*sign))),max_mirrored_joint_velocity_difference_rad_s=float(np.max(np.abs(right[0][:,37:]-left[0][:,37:][:,perm]*sign))),max_mirrored_torque_difference_nm=float(np.max(np.abs(right[1]-left[1][:,perm]*sign))))
 for key,val in errors.items():
  if abs(row[key]-val)>1e-10:raise RuntimeError('Response metric differs')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(cases=evidence,reconstructed_states=4824,all24valid=True,scope='Reconstruction only; no fresh physics stepping; no PhysX equivalence claim'),indent=2));print('PASS:24responses,4824states,torques/caps/time/symmetrymetrics/source hashes')
