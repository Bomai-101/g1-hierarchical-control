"""Rebuild physical probes and independently verify reflected-policy trajectories."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import mujoco
import numpy as np

def require(ok,message):
 if not ok:raise RuntimeError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(a.reference_root/'mujoco'))
 from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
 from mujoco_eval.run_grid import resolve_joint_id
 folder=a.run_root/'usd_physics_isolation';complete=json.loads((folder/'complete.json').read_text());inputs=json.loads((folder/'inputs.json').read_text())
 for path,h in inputs.items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'Input changed')
 meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');audit=json.loads((a.output/'bilateral_audit.json').read_text());perm=np.array(audit['mirror_permutation']);sign=np.array(audit['mirror_signs'],dtype=np.float32);S=np.diag([1.,-1.,1.]);evidence=[]
 for row in complete['results']:
  variant=row['variant'];speed=row['speed'];stem=f'{variant}_vx{speed:g}_yaw+0.0';path=folder/f'{stem}.npz';trace=np.load(path);m=mujoco.MjModel.from_binary_path(str(a.run_root/'usd_physical_models/g1_usd_physical_position.mjb'));original=mujoco.MjModel.from_binary_path(str(a.run_root/'usd_physical_models/g1_usd_physical_position.mjb'));ids=np.array([resolve_joint_id(m,n)[0] for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
  base=mujoco.MjData(m);base.qpos[qa]=defaults;mujoco.mj_forward(m,base)
  for item in audit['central_bodies']:
   body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,item['body']);R=base.xmat[body].reshape(3,3)
   if variant=='central_com_sym':
    com=R@m.body_ipos[body];com-=np.array([0,com[1],0]);m.body_ipos[body]=R.T@com
   if variant=='central_inertia_sym':
    Q=base.ximat[body].reshape(3,3);I=Q@np.diag(m.body_inertia[body])@Q.T;I[0,1]=I[1,0]=I[1,2]=I[2,1]=0;values,vectors=np.linalg.eigh(R.T@I@R)
    if np.linalg.det(vectors)<0:vectors[:,0]*=-1
    q=np.zeros(4);mujoco.mju_mat2Quat(q,vectors.flatten());m.body_inertia[body]=values;m.body_iquat[body]=q
  if variant in ('refresh_constants','central_com_sym','central_inertia_sym'):mujoco.mj_setConst(m,mujoco.MjData(m))
  if variant=='contact_solref01':m.geom_solref[(m.geom_contype!=0)|(m.geom_conaffinity!=0),0]=.01
  if variant=='cone_elliptic':m.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
  if variant=='integrator_implicit':m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT
  changed=[n for n in dir(m) if isinstance(getattr(m,n),np.ndarray) and not np.array_equal(getattr(m,n),getattr(original,n))]
  require(changed==row['changed_model_arrays'] and int(m.opt.cone)==row['model_options']['cone'] and int(m.opt.integrator)==row['model_options']['integrator'],'Model variant differs')
  if variant=='central_com_sym':require(np.array_equal(m.body_mass,original.body_mass) and np.array_equal(m.body_inertia,original.body_inertia),'COM test changes mass/inertia')
  if variant=='central_inertia_sym':require(np.array_equal(m.body_mass,original.body_mass) and np.array_equal(m.body_ipos,original.body_ipos),'Inertia test changes mass/COM')
  d=mujoco.MjData(m);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');rebuilt=[];actor_error=0;selected=set(np.linspace(0,999,32,dtype=int))
  for i,state in enumerate(trace['states']):
   d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);quat=d.xquat[pelvis];R=d.xmat[pelvis].reshape(3,3);heading=np.arctan2(R[1,0],R[0,0]);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,v,0);ang,lin=v[:3],v[3:];blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang);gravity=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-gravity[2],-1,1));cmd=np.array([speed,0,0]);rebuilt.append(np.r_[d.qpos[:3],quat,lin,ang,blin,bang,d.xpos[pelvis,2],tilt,cmd,heading])
   if i in selected:
    previous=trace['actions'][i-1] if i else np.zeros(37);obs=build_policy_observation(quat,lin,ang,cmd,d.qpos[qa],d.qvel[va],defaults,previous)
    if variant=='mirror_actor':
     reflected=np.r_[obs[:3]*np.array([1,-1,1],dtype=np.float32),obs[3:6]*np.array([-1,1,-1],dtype=np.float32),obs[6:9]*np.array([1,-1,1],dtype=np.float32),obs[9:12]*np.array([1,-1,-1],dtype=np.float32),*[obs[start:start+37][perm]*sign for start in (12,49,86)]]
     action=actor.act(reflected)[perm]*sign
    else:action=actor.act(obs)
    actor_error=max(actor_error,float(np.max(np.abs(action-trace['actions'][i]))))
  rebuilt=np.array(rebuilt);err=float(np.max(np.abs(rebuilt-trace['signals'])));require(err<1e-10 and actor_error<1e-6,'Signal/actor mismatch');time=trace['time'];require(len(time)==1001 and len(trace['actions'])==1000 and row['physical_steps']==20000 and not row['fell'] and not row['numerical_failure'],'Invalid episode');require(np.allclose(np.diff(time),.02,atol=1e-10) and abs(time[-1]-20)<1e-8,'Invalid time');require(np.all(trace['force_max']<=m.jnt_actfrcrange[ids,1]+1e-7),'Force cap exceeded')
  mask=time>=time[-1]-10;t=time[mask];h=np.unwrap(rebuilt[:,24]);center=t-t.mean();slope=float(np.dot(center,h[mask]-h[mask].mean())/np.dot(center,center));metrics=dict(heading_slope=slope,mean_world_wz=float(rebuilt[mask,12].mean()),mean_body_wz=float(rebuilt[mask,18].mean()),heading_endpoint_rate=float((h[-1]-h[np.flatnonzero(mask)[0]])/(time[-1]-time[np.flatnonzero(mask)[0]])),forward_rmse=float(np.sqrt(np.mean((rebuilt[mask,13]-speed)**2))),max_tilt_deg=float(np.rad2deg(rebuilt[:,20].max())))
  for name,val in metrics.items():require(abs(row[name]-val)<1e-10,'Metric differs')
  evidence.append(dict(case=stem,states=1001,actor_inputs=32,signal_max_error=err,actor_max_error=actor_error,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 consistency=[]
 for speed in (.5,1.):
  base=np.load(folder/f'baseline_vx{speed:g}_yaw+0.0.npz');old=np.load(a.run_root/'usd_yaw_diagnosis'/f'baseline_vx{speed:g}_yaw+0.0.npz');refresh=np.load(folder/f'refresh_constants_vx{speed:g}_yaw+0.0.npz')
  for label,x in [('previous20sbaseline',old),('setConstonly',refresh)]:
   equal=all(np.array_equal(base[key],x[key]) for key in base.files);require(equal,'Baseline/control is not bytewise-array identical');consistency.append(dict(speed=speed,comparison=label,all_trace_arrays_equal=True))
 with (a.output/'comparison.csv').open('w') as stream:w=csv.DictWriter(stream,fieldnames=list(complete['results'][0]));w.writeheader();w.writerows(complete['results'])
 for name in ('inputs.json','complete.json'):(a.output/name).write_bytes((folder/name).read_bytes())
 (a.output/'verification.json').write_text(json.dumps(dict(cases=evidence,all16valid20s=True,reconstructed_states=16016,actor_inputs=512,consistency=consistency,scope='Isolated in-memory interventions and diagnostic mirror controller; source model/weights unchanged. No PhysX dynamics equivalence or robustness claim.'),indent=2));print('PASS:16x1001states,512actorinputs;baseline andsetConst-only control identical')
if __name__=='__main__':main()
