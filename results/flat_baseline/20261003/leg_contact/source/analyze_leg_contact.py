"""Independently check saved branches and summarize cross-engine response errors."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import mujoco
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def require(v,msg):
 if not v:raise RuntimeError(msg)
def yaw(q):return np.unwrap(np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2)),axis=0)
sys.path.insert(0,str(a.reference_root/'mujoco'))
from mujoco_eval.policy import TorchActorPolicy
walk=np.load(a.run_root/'leg_contact_isaac/walking.npz');actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text())
for ti in range(300):
 require(np.max(np.abs(actor.act(walk['obs'][ti])-walk['action'][ti]))<1e-5,'Source actor')
 require(np.max(np.abs(np.array(meta['default_joint_pos'],np.float32)+float(meta['action_scale'])*walk['action'][ti]-walk['target'][ti]))<1e-7,'Source target')
rows=[];verified=0;max_force_rebuild_error=0.;source_proofs={};raw_hashes={}
for condition in ('contact','free','lifted'):
 isa=a.run_root/f'leg_{condition}_isaac';mu=a.run_root/f'leg_{condition}_mujoco';r=json.loads((isa/'runtime.json').read_text());x=np.load(isa/'responses.npz');require((isa/'complete.json').exists() and (mu/'complete.json').exists(),'Missing completion')
 for folder in (isa,mu):
  h=json.loads((folder/'inputs.json').read_text())
  for f,digest in h.items():
   source_file=Path(f)
   if source_file.name=='capture_leg_contact_isaac.py' and hashlib.sha256(source_file.read_bytes()).hexdigest()!=digest:source_file=a.run_root/'leg_contact_isaac/source.py'
   if source_file.name=='inputs.json' and hashlib.sha256(source_file.read_bytes()).hexdigest()!=digest:
    source_file=a.run_root/'leg_lifted_isaac/source_manifest_at_capture.json'
   require(hashlib.sha256(source_file.read_bytes()).hexdigest()==digest,'Changed input '+f)
  source_proofs[str(folder)]=True
  for f in folder.glob('*'):
   if f.is_file():raw_hashes[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
 names=r['joint_names'];leg=np.array([i for i,n in enumerate(names) if any(s in n for s in ('hip_','knee_','ankle_'))]);feet=[r['sensor_body_names'].index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
 require(len(leg)==12,'Leg count');require(np.array_equal(x['time'],np.arange(41)*.005),'Isaac timing')
 if condition in ('free','lifted'):require(np.max(np.abs(x['contact_force'][1:]))<1e-6,'Lifted contacts')
 if condition=='contact':
  for ci,c in enumerate(r['cases']):
   ti=c['source_sample'];support=walk['contact_force'][ti,feet,2]>20;expected={'left_only':[True,False],'right_only':[False,True],'double':[True,True]}[c['support']]
   require(support.tolist()==expected,'Source support class')
   require(np.array_equal(x['target'][ci],walk['target'][ti]),'Source branch target')
 for mode in ('dt005','dt001'):
  y=np.load(mu/f'{mode}.npz');require(y['states'].shape[:2]==(41,len(r['cases'])),'Count');require(np.array_equal(x['time'],y['time']) and np.array_equal(x['target'],y['target']),'Timing/targets')
  m=mujoco.MjModel.from_binary_path(str(a.model));m.opt.timestep=.005 if mode=='dt005' else .001
  if condition=='free':m.opt.gravity[:]=0
  ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);bids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in r['body_names']]);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');foot_bids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
  require(np.max(np.abs(m.actuator_gainprm[ai,0]-r['stiffness']))<1e-6,'Kp')
  require(np.max(np.abs(-m.actuator_biasprm[ai,2]-r['damping']))<1e-6,'Kd')
  require(np.max(np.abs(m.dof_armature[va]-r['armature']))<1e-6,'Armature')
  require(np.max(np.abs(m.jnt_actfrcrange[ids,1]-r['effort_limits']))<1e-6,'Effort limits')
  for ci,case in enumerate(r['cases']):
   require(np.max(np.abs(y['joint_pos'][0,ci]-x['joint_pos'][0,ci]))<1e-12,'Initial q')
   require(np.max(np.abs(y['joint_vel'][0,ci]-x['joint_vel'][0,ci]))<1e-12,'Initial qv')
   d=mujoco.MjData(m)
   for ti in range(41):
    state=y['states'][ti,ci];require(np.all(np.isfinite(state)),'Finite');d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[ai]=x['target'][ci];mujoco.mj_forward(m,d)
    require(np.array_equal(d.qpos[qa],y['joint_pos'][ti,ci]) and np.array_equal(d.qvel[va],y['joint_vel'][ti,ci]),'Joint reconstruction')
    require(np.max(np.abs(d.xpos[bids]-y['body_pose'][ti,ci,:,:3]))<1e-12,'Body reconstruction')
    require(np.max(np.abs(d.xipos[pelvis]-y['root_com'][ti,ci,:3]))<1e-12,'COM reconstruction')
    sp=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,sp,0)
    require(np.max(np.abs(np.r_[sp[3:],sp[:3]]-y['root_com'][ti,ci,7:]))<1e-12,'COM velocity')
    if ti==0:
     require(np.max(np.abs(d.xipos[pelvis]-x['root_com'][0,ci,:3]))<2e-6,'Initial COM position')
     require(np.max(np.abs(np.r_[sp[3:],sp[:3]]-x['root_com'][0,ci,7:]))<2e-6,'Initial COM velocity')
    fv=np.zeros((2,3))
    for cj in range(d.ncon):
     con=d.contact[cj];force=np.zeros(6);mujoco.mj_contactForce(m,d,cj,force);fw=con.frame.reshape(3,3).T@force[:3]
     for fi,bid in enumerate(foot_bids):
      if m.geom_bodyid[con.geom[1]]==bid:fv[fi]+=fw
      if m.geom_bodyid[con.geom[0]]==bid:fv[fi]-=fw
    max_force_rebuild_error=max(max_force_rebuild_error,float(np.abs(fv-y['foot_force'][ti,ci]).max()))
    require(d.ncon==y['ncon'][ti,ci] and np.max(np.abs(fv-y['foot_force'][ti,ci]))<1e-3,'Contact force reconstruction')
    require(np.max(np.abs(d.qfrc_actuator[va]-y['nominal_torque'][ti,ci]))<1e-8,'Mu force reconstruction')
    require(np.all(np.abs(d.qfrc_actuator[va])<=m.jnt_actfrcrange[ids,1]+1e-7),'Mu force limits');verified+=1
   for horizon in (.02,.05,.2):
    end=round(horizon/.005);err=y['joint_pos'][:end+1,ci][:,leg]-x['joint_pos'][:end+1,ci][:,leg];idx=np.unravel_index(np.abs(err).argmax(),err.shape);heading_delta=float((yaw(y['root_link'][:end+1,ci,3:7])-yaw(x['root_link'][:end+1,ci,3:7]))[-1]);v=y['root_com'][end,ci,7:10]-x['root_com'][end,ci,7:10]
    # Endpoint rectangle quadrature, not equal solver impulse histories.
    phys_imp=x['contact_force'][1:end+1,ci][:,feet].sum(axis=0)*.005;mu_imp=y['foot_force'][1:end+1,ci].sum(axis=0)*.005
    rows.append(dict(condition=condition,mode=mode,case=case['name'],support=case.get('support','large_step'),horizon_s=horizon,leg_q_max_rad=float(np.abs(err).max()),leg_q_rmse_rad=float(np.sqrt(np.mean(err**2))),largest_joint=names[leg[idx[1]]],heading_delta_rad=heading_delta,com_vx_delta_mps=float(v[0]),com_vy_delta_mps=float(v[1]),com_vz_delta_mps=float(v[2]),left_fz_impulse_isaac_Ns=float(phys_imp[0,2]),left_fz_impulse_mu_Ns=float(mu_imp[0,2]),right_fz_impulse_isaac_Ns=float(phys_imp[1,2]),right_fz_impulse_mu_Ns=float(mu_imp[1,2])))
with (a.output/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
proof=dict(status='PASS',max_contact_force_rebuild_error_N=max_force_rebuild_error,rebuilt_mujoco_states=verified,input_hashes=source_proofs,summary_rows=len(rows),note='No passive monitor equivalence claim. PhysX force sensor vs Mu endpoint force quadrature not same instantaneous definition. Reset branch contact warm starts unavailable.')
(a.output/'verification.json').write_text(json.dumps(proof,indent=2));(a.output/'raw_hashes.json').write_text(json.dumps(raw_hashes,indent=2))
for condition in ('contact','free','lifted'):
 for mode in ('dt005','dt001'):
  rr=[v for v in rows if v['condition']==condition and v['mode']==mode and v['horizon_s']==.2];print(condition,mode,'maxleg',max(v['leg_q_max_rad'] for v in rr),'maxyaw',max(abs(v['heading_delta_rad']) for v in rr),'maxvx',max(abs(v['com_vx_delta_mps']) for v in rr),flush=True)
print(json.dumps(proof),flush=True)
