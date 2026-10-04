"""Compare left/right reflected small target steps without policy or contacts."""
import argparse,json,hashlib
from pathlib import Path
import mujoco
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb';meta=json.loads(a.metadata.read_text());audit=json.loads(a.audit.read_text());perm=np.array(audit['mirror_permutation']);sign=np.array(audit['mirror_signs']);names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);rows=[]
for integrator in ('implicitfast','implicit'):
 for joint in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll'):
  outputs=[]
  for side in ('left','right'):
   m=mujoco.MjModel.from_binary_path(str(path));m.opt.gravity[:]=0
   if integrator=='implicit':m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT
   d=mujoco.MjData(m);ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,2];d.qpos[qa]=defaults
   index=names.index(f'{side}_{joint}_joint');leftindex=names.index(f'left_{joint}_joint');delta=.01 if side=='left' else .01*sign[leftindex];target=defaults.copy();target[index]+=delta;states=[];signals=[];forces=[]
   for step in range(201):
    d.ctrl[ai]=target;mujoco.mj_forward(m,d)
    if d.ncon:raise RuntimeError('Suspended test unexpectedly has contact')
    states.append(np.r_[d.qpos,d.qvel].copy());signals.append(np.r_[d.qpos[qa]-defaults,d.qvel[va]]);forces.append(d.qfrc_actuator[va].copy())
    if not np.all(np.isfinite(d.qpos)) or any(w.number for w in d.warning):raise RuntimeError('Invalid response')
    if step<200:mujoco.mj_step(m,d)
   signals=np.array(signals);forces=np.array(forces);initial=m.actuator_gainprm[ai,0]*(target-defaults)
   if np.max(np.abs(forces[0]-initial))>1e-8:raise RuntimeError('Initial torque differs from configured gain')
   if np.any(np.max(np.abs(forces),axis=0)>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap exceeded')
   stem=f'{integrator}_{side}_{joint}';trace=a.output/f'{stem}.npz';np.savez_compressed(trace,time=np.arange(201)*.001,states=states,signals=signals,forces=forces,target=target)
   outputs.append((signals,forces,index,trace))
  left,right=outputs;deltaerr=right[0][:,:37]-left[0][:,:37][:,perm]*sign;velerr=right[0][:,37:]-left[0][:,37:][:,perm]*sign;forceerr=right[1]-left[1][:,perm]*sign
  rows.append(dict(integrator=integrator,joint=joint,initial_target_delta_rad=.01,left_final_step_rad=float(left[0][-1,left[2]]),right_final_step_mirrored_rad=float(right[0][-1,right[2]]*sign[left[2]]),max_mirrored_joint_position_difference_rad=float(np.abs(deltaerr).max()),max_mirrored_joint_velocity_difference_rad_s=float(np.abs(velerr).max()),max_mirrored_torque_difference_nm=float(np.abs(forceerr).max()),trace_hashes={str(x[3].resolve()):hashlib.sha256(x[3].read_bytes()).hexdigest() for x in outputs}))
(a.output/'actuation_probe.json').write_text(json.dumps(dict(results=rows,protocol='6paired joints x2integrators; each left/right .01rad mirroredstep,200ms; same37positionservos; free-floating gravityoff,2m height,no contacts,no policy; actualcentralasymmetry retained',inputs={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in (path,a.metadata,a.audit,Path(__file__))},limitation='Short suspended response symmetry is not a matched-state PhysX comparison or proof of walking contact dynamics equivalence.'),indent=2));print('PASS:24 suspendedresponses; initialtorques/caps/no contacts; maxjointpositionasymmetry',max(r['max_mirrored_joint_position_difference_rad'] for r in rows))
