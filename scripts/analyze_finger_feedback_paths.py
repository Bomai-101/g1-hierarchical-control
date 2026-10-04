"""Independently verify physics/observation path interventions and policy sensitivity."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import mujoco
import numpy as np

def require(ok,message):
 if not ok:raise RuntimeError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);folder=a.run_root/'finger_feedback_paths';complete=json.loads((folder/'complete.json').read_text());inputs=json.loads((folder/'inputs.json').read_text())
 for path,h in inputs.items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'Input changed: '+path)
 sys.path.insert(0,str(a.reference_root/'mujoco'))
 from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
 from mujoco_eval.run_grid import resolve_joint_id
 meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);fi=np.array([names.index(n) for n in complete['protocol']['fingers']]);slots=np.r_[12+fi,49+fi];outside=np.ones(123,dtype=bool);outside[slots]=False;leg=[i for i,n in enumerate(names) if any(s in n for s in ('hip','knee','ankle'))];nonfinger=[i for i in range(37) if i not in fi];actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');model_path=a.run_root/'usd_physical_models/g1_usd_physical_position.mjb';evidence=[];sensitivity=[];consistency=[];summary=[]
 for row in complete['results']:
  stem=f"{row['variant']}_vx{row['speed']:g}";path=folder/f'{stem}.npz';t=np.load(path);models=[];datas=[]
  for mode in (row['physical'],row['paired_physical']):
   m=mujoco.MjModel.from_binary_path(str(model_path));original=mujoco.MjModel.from_binary_path(str(model_path));ids=np.array([resolve_joint_id(m,n)[0] for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
   if mode=='tight':m.jnt_solref[ids[fi],0]=.002
   changed=[n for n in dir(m) if isinstance(getattr(m,n),np.ndarray) and not np.array_equal(getattr(m,n),getattr(original,n))];require(changed==(['jnt_solref'] if mode=='tight' else []),'Unexpectedphysical change');require(m.opt.timestep==.001,'Unexpectedphysics rate');models.append(m);datas.append(mujoco.MjData(m))
  qmask=np.ones(models[0].nq,dtype=bool);qmask[qa[fi]]=False;vmask=np.ones(models[0].nv,dtype=bool);vmask[va[fi]]=False
  require(np.array_equal(t['states'][:,:models[0].nq][:,qmask],t['paired_states'][:,:models[0].nq][:,qmask]) and np.array_equal(t['states'][:,models[0].nq:][:,vmask],t['paired_states'][:,models[0].nq:][:,vmask]),'Paired non-finger state not synchronized')
  require(t['obs_actual'].shape==(1000,123) and t['obs_used'].shape==(1000,123),'Observation shape differs');require(np.array_equal(t['obs_actual'][:,outside],t['obs_used'][:,outside]),'Intervention leaked beyond8slots');require(np.array_equal(t['targets'],defaults+meta['action_scale']*t['actions']),'Target rule changed')
  require(len(t['time'])==1001 and len(t['actions'])==1000 and row['physical_steps']==20000 and not row['fell'] and not row['paired_fell'] and not row['numerical_failure'],'Incomplete/invalid episode');require(abs(t['time'][-1]-20)<1e-8 and np.allclose(np.diff(t['time']),.02,atol=1e-10),'Timeline changed')
  max_signal=max_actor=max_obs=0.;leg_delta=[];nonfinger_delta=[];all_delta=[]
  for k in range(1001):
   rebuilt=[]
   for index,(m,d,key) in enumerate(zip(models,datas,('states','paired_states'))):
    state=t[key][k];d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');q=d.xquat[pelvis].copy();R=d.xmat[pelvis].reshape(3,3);h=np.arctan2(R[1,0],R[0,0]);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pelvis,v,0);ang,lin=v[:3],v[3:];blin=quat_rotate_inverse(q,lin);bang=quat_rotate_inverse(q,ang);g=quat_rotate_inverse(q,[0,0,-1]);tilt=np.arccos(np.clip(-g[2],-1,1));cmd=np.array([row['speed'],0,0]);signal=np.r_[d.qpos[:3],q,lin,ang,blin,bang,d.xpos[pelvis,2],tilt,cmd,h];recorded=t['signals' if index==0 else 'paired_signals'][k];max_signal=max(max_signal,float(np.max(np.abs(signal-recorded))));rebuilt.append((q,lin,ang,signal))
   if k<1000:
    q,lin,ang,_=rebuilt[0];prev=t['actions'][k-1] if k else np.zeros(37);raw=build_policy_observation(q,lin,ang,np.array([row['speed'],0,0]),datas[0].qpos[qa],datas[0].qvel[va],defaults,prev);used=raw.copy()
    if row['observation']=='paired':used[12+fi]=(datas[1].qpos[qa[fi]]-defaults[fi]).astype(np.float32);used[49+fi]=datas[1].qvel[va[fi]].astype(np.float32)
    max_obs=max(max_obs,float(np.max(np.abs(raw-t['obs_actual'][k]))),float(np.max(np.abs(used-t['obs_used'][k]))));action=actor.act(used);actual_only=actor.act(raw);max_actor=max(max_actor,float(np.max(np.abs(action-t['actions'][k]))));delta=action-actual_only;leg_delta.append(delta[leg]);nonfinger_delta.append(delta[nonfinger]);all_delta.append(delta)
  require(max_signal<1e-10 and max_obs<1e-6 and max_actor<1e-6,'Independent signal/observation/action differs')
  for index,m in enumerate(models):require(np.all(t['force_max'][index]<=m.jnt_actfrcrange[ids,1]+1e-7),'Force cap exceeded')
  mask=t['time']>=t['time'][-1]-10;tt=t['time'][mask];h=np.unwrap(t['signals'][:,24]);center=tt-tt.mean();metrics=dict(heading_slope=float(center@(h[mask]-h[mask].mean())/(center@center)),mean_world_wz=float(t['signals'][mask,12].mean()),forward_rmse=float(np.sqrt(np.mean((t['signals'][mask,13]-row['speed'])**2))),mean_body_vx=float(t['signals'][mask,13].mean()),max_tilt_deg=float(np.rad2deg(t['signals'][:,20].max())),paired_max_tilt_deg=float(np.rad2deg(t['paired_signals'][:,20].max())),max_paired_root_position_difference_m=float(np.max(np.linalg.norm(t['signals'][:,:3]-t['paired_signals'][:,:3],axis=1))),max_finger_observation_intervention=float(np.max(np.abs(t['obs_actual'][:,slots]-t['obs_used'][:,slots]))))
  for name,val in metrics.items():require(abs(row[name]-val)<1e-10,'Metric differs: '+name)
  if row['observation']=='self':
   previous='baseline' if row['physical']=='soft' else 'active4';old=np.load(a.run_root/'finger_limit_localization'/f"{previous}_vx{row['speed']:g}_yaw+0.0.npz");same=all(np.array_equal(t[key],old[key]) for key in ('time','states','signals','actions')) and np.array_equal(t['force_max'][0],old['force_max']);require(same,'Paired calculation perturbed control');consistency.append(dict(case=stem,previous=previous,all_control_arrays_equal=True))
  sensitivity.append(dict(case=stem,leg_action_change_rms=float(np.sqrt(np.mean(np.array(leg_delta)**2))),leg_action_change_max=float(np.max(np.abs(leg_delta))),nonfinger_action_change_rms=float(np.sqrt(np.mean(np.array(nonfinger_delta)**2))),nonfinger_action_change_max=float(np.max(np.abs(nonfinger_delta))),all_action_change_rms=float(np.sqrt(np.mean(np.array(all_delta)**2))),interpretation='Instantaneous same-mainstate: actor(usedobs)-actor(actualobs); only8slots differ, same37previousactions. Doesnotestimatewholeclosedloop causalpercentage.'))
  evidence.append(dict(case=stem,main_states=1001,paired_states=1001,actor_used_evaluations=1000,actor_actual_evaluations=1000,max_signal_error=max_signal,max_observation_error=max_obs,max_action_error=max_actor,nonfinger_states_exactly_synced=True,only8slots_changed=True,force_caps_pass=True,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));summary.append(row)
 for name,rows in [('summary.csv',summary),('policy_sensitivity.csv',sensitivity)]:
  with (a.output/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for name in ('inputs.json','complete.json'):(a.output/name).write_bytes((folder/name).read_bytes())
 (a.output/'verification.json').write_text(json.dumps(dict(cases=evidence,consistency=consistency,reconstructed_states=16016,actor_evaluations=16000,all8main_and_conditional_cases_complete20s=True,scope='Conditional pairedmodel with per-step non-fingerstate transplantation; artificial activepolicyinput diagnostic,notpassivemonitor ordefaultdeployment. Noevidenceofsolecause oradditivepathpercentage.'),indent=2));print('PASS:16016states,16000actorcalls,8slots-only,synchronizedstates,metrics,forcecaps,inputhashes;4priorcontrolsidentical')
if __name__=='__main__':main()
