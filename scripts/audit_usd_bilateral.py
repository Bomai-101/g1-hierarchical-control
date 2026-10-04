"""Audit sagittal symmetry of the frozen derived physical model."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb';m=mujoco.MjModel.from_binary_path(str(path));d=mujoco.MjData(m);meta=json.loads(a.metadata.read_text());names=meta['joint_names'];ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];d.qpos[qa]=meta['default_joint_pos'];mujoco.mj_forward(m,d);S=np.diag([1.,-1.,1.])
def inertia(i):
 R=d.ximat[i].reshape(3,3);return R@np.diag(m.body_inertia[i])@R.T
bodyrows=[];central=[]
for i in range(1,m.nbody):
 n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i)
 if n.startswith('left_'):
  r=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n.replace('left_','right_',1))
  if r<0:raise RuntimeError('Unpaired body')
  bodyrows.append(dict(left=n,mass_difference_kg=float(m.body_mass[i]-m.body_mass[r]),position_mirror_error_m=float(np.linalg.norm(S@d.xpos[i]-d.xpos[r])),com_mirror_error_m=float(np.linalg.norm(S@d.xipos[i]-d.xipos[r])),inertia_mirror_error_kg_m2=float(np.max(np.abs(S@inertia(i)@S-inertia(r))))))
 elif not n.startswith('right_'):
  central.append(dict(body=n,mass_kg=float(m.body_mass[i]),world_com_y_m=float(d.xipos[i,1]),body_com_y_m=float(m.body_ipos[i,1]),world_inertia_mirror_error=float(np.max(np.abs(S@inertia(i)@S-inertia(i))))))
perm=[];sign=[];jointrows=[]
for k,n in enumerate(names):
 other=n.replace('left_','right_',1) if n.startswith('left_') else n.replace('right_','left_',1) if n.startswith('right_') else n
 j=names.index(other);dot=float((-S@d.xaxis[ids[k]])@d.xaxis[ids[j]]);s=1 if dot>0 else -1;perm.append(j);sign.append(s)
 jointrows.append(dict(joint=n,paired=other,mirror_sign=s,axis_error=float(np.linalg.norm(s*(-S@d.xaxis[ids[k]])-d.xaxis[ids[j]])),default_position_error=float(d.qpos[qa[j]]-s*d.qpos[qa[k]]),range_error=float(np.max(np.abs(m.jnt_range[ids[j]]-(m.jnt_range[ids[k]] if s==1 else -m.jnt_range[ids[k]][::-1])))),armature_error=float(m.dof_armature[va[k]]-m.dof_armature[va[j]]),gain_error=float(np.max(np.abs(m.actuator_gainprm[ai[k]]-m.actuator_gainprm[ai[j]]))),bias_error=float(np.max(np.abs(m.actuator_biasprm[ai[k]]-m.actuator_biasprm[ai[j]]))),force_limit_error=float(np.max(np.abs(m.actuator_forcerange[ai[k]]-m.actuator_forcerange[ai[j]])))))
perm=np.array(perm);sign=np.array(sign)
if not np.array_equal(perm[perm],np.arange(37)) or not np.array_equal(sign*sign[perm],np.ones(37)):raise RuntimeError('Mirror map is not involutive')
footrows=[]
for side in ('left','right'):
 gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'contact_{side}_ankle_roll_link');mid=m.geom_dataid[gid];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];world=v@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];footrows.append((gid,world))
l,r=footrows;dist=np.linalg.norm((l[1]@S)[:,None,:]-r[1][None,:,:],axis=2);footerror=float(max(dist.min(axis=0).max(),dist.min(axis=1).max()))
contacterrors={field:float(np.max(np.abs(getattr(m,field)[l[0]]-getattr(m,field)[r[0]]))) for field in ('geom_friction','geom_solref','geom_solimp','geom_condim','geom_contype','geom_conaffinity')}
if footerror>1e-7 or any(contacterrors.values()):raise RuntimeError('Foot contacts are not mirrored')
for row in bodyrows:
 if any(abs(row[key])>1e-10 for key in row if key!='left'):raise RuntimeError('Bilateral body mismatch')
for row in jointrows:
 if any(abs(row[key])>1e-10 for key in row if key not in ('joint','paired','mirror_sign')):raise RuntimeError('Joint symmetry mismatch')
result=dict(bilateral_bodies=bodyrows,joints=jointrows,mirror_permutation=perm.tolist(),mirror_signs=sign.tolist(),central_bodies=central,foot_contact_hausdorff_error_m=footerror,foot_contact_parameter_errors=contacterrors,whole_body_com_y_m=float(d.subtree_com[1,1]),model_options=dict(integrator=int(m.opt.integrator),solver=int(m.opt.solver),cone=int(m.opt.cone),iterations=int(m.opt.iterations),tolerance=float(m.opt.tolerance)),input_hashes={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in (path,a.metadata,Path(__file__))},limitation='Bilateral limbs and contacts pass; central torso/head/logo COM and inertia are not exactly sagittally symmetric. These were copied from actual PhysX runtime; not evidence of a conversion error.')
(a.output/'bilateral_audit.json').write_text(json.dumps(result,indent=2));print('PASS:19 bilateralbody pairs,37joint mirror maps,foot contacts; central asymmetries recorded',footerror)
