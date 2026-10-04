"""Verify contact-only changes, branch selection, full-state signals and actor outputs."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import mujoco
import numpy as np
from contact_parameter_variants import VARIANTS,apply_variant
p=argparse.ArgumentParser()
for name in ('run-root','old-root','reference-root','report','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.reference_root/'mujoco'))
from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
from mujoco_eval.run_grid import read_base_state

def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name,rows):
 with (a.output/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
model_path=a.old_root/'usd_physical_models/g1_usd_physical_position.mjb';meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');legs=[i for i,n in enumerate(names) if any(s in n for s in ('hip_','knee_','ankle_'))]
branch=a.run_root/'contact_parameter_branches';walk=a.run_root/'contact_parameter_walking';source=a.run_root/'leg_contact_isaac';x=np.load(source/'responses.npz');rt=json.loads((source/'runtime.json').read_text());selection=json.loads((a.report/'selection.json').read_text());plan=json.loads((a.report/'plan.json').read_text());require(sha(a.report/'plan.json')==selection['plan_sha256'],'Plan changed');require(sha(branch/'complete.json')==selection['branch_complete_sha256'],'Branch result changed');raw_hashes={}
for folder in (branch,walk):
 require((folder/'complete.json').exists(),'Incomplete run')
 for path,digest in json.loads((folder/'inputs.json').read_text()).items():require(sha(Path(path))==digest,'Changed input '+path)
 for file in folder.iterdir():
  if file.is_file():raw_hashes[str(file.resolve())]=sha(file)
branch_rows=[];dev={};states_checked=0;actors_checked=0
for variant,spec in VARIANTS.items():
 m=mujoco.MjModel.from_binary_path(str(model_path));original=mujoco.MjModel.from_binary_path(str(model_path));apply_variant(m,variant);m.opt.timestep=.001
 # Check every model array against the explicitly constructed expected array.
 for name in dir(m):
  value=getattr(m,name)
  if isinstance(value,np.ndarray):
   expected=getattr(original,name).copy()
   if spec and name==spec[0]:expected[(original.geom_contype!=0)|(original.geom_conaffinity!=0),spec[1]]=spec[2]
   require(np.array_equal(value,expected),'Unexpected model mutation '+name)
 ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);bids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in rt['body_names']]);y=np.load(branch/f'{variant}.npz');require(np.array_equal(y['time'],x['time']) and np.array_equal(y['target'],x['target']),'Branch time/target')
 if variant=='baseline':
  previous=np.load(a.run_root/'leg_contact_mujoco/dt001.npz');require(all(np.array_equal(y[k],previous[k]) for k in y.files),'Previous branch baseline differs')
 errors=y['joint_pos'][:,:,legs]-x['joint_pos'][:,:,legs];dev[variant]=float(np.sqrt(np.mean(errors[:,plan['development_indices']]**2)))
 for ci,c in enumerate(rt['cases']):
  require(np.array_equal(y['joint_pos'][0,ci],x['joint_pos'][0,ci]) and np.array_equal(y['joint_vel'][0,ci],x['joint_vel'][0,ci]),'Initial branch joint state')
  d=mujoco.MjData(m)
  for ti,state in enumerate(y['states'][:,ci]):
   d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
   require(np.max(abs(d.xpos[bids]-y['body_pose'][ti,ci,:,:3]))<1e-12,'Branch body positions');require(np.array_equal(d.qpos[qa],y['joint_pos'][ti,ci]),'Branch qpos');quat,lin,ang,height=read_base_state(m,d,'pelvis');require(np.max(abs(np.r_[lin,ang]-y['root_com'][ti,ci,7:]))<1e-12,'Branch COM velocity');states_checked+=1
  branch_rows.append(dict(variant=variant,case=c['name'],split='development' if ci in plan['development_indices'] else 'heldout',leg_rmse_rad=float(np.sqrt(np.mean(errors[:,ci]**2))),leg_max_rad=float(abs(errors[:,ci]).max())))
expected=['baseline']+[min((v for v in dev if v.startswith(prefix)),key=lambda v:(dev[v],v)) for prefix in ('friction','contact_time')];require(expected==selection['selected'],'Selection used wrong development score');require(all(abs(dev[v]-selection['development_leg_rmse_rad'][v])<1e-14 for v in dev),'Development scores')
complete=json.loads((walk/'complete.json').read_text());require(len(complete['results'])==18,'Walking grid');rows=[]
for recorded in complete['results']:
 variant=recorded['variant'];speed=recorded['speed'];yaw=recorded['yaw_command'];require(variant in expected and speed in plan['walking']['speeds'] and yaw in plan['walking']['yaw_commands'],'Grid entry');stem=f'{variant}_vx{speed:g}_yaw{yaw:+.1f}';y=np.load(walk/f'{stem}.npz');m=mujoco.MjModel.from_binary_path(str(model_path));apply_variant(m,variant);d=mujoco.MjData(m);ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);require(y['states'].shape==(1001,87) and y['actions'].shape==(1000,37),'Walking count');require(np.max(abs(y['time']-np.arange(1001)*.02))<1e-10,'Walking time');require(not recorded['fell'] and not recorded['numerical_failure'] and recorded['physical_steps']==20000,'Unsuccessful walking');command=np.array([speed,0,yaw]);last=np.zeros(37)
 for ti,state in enumerate(y['states']):
  require(np.all(np.isfinite(state)),'Nonfinite state');d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang);g=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-g[2],-1,1));heading=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2));signal=np.r_[d.qpos[:3],quat,lin,ang,blin,bang,height,tilt,command,heading];require(np.max(abs(signal-y['signals'][ti]))<1e-11,'Rebuilt walking signal');require(height>=.35,'Height termination');states_checked+=1
  if ti<1000:
   obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last);action=actor.act(obs);require(np.max(abs(action-y['actions'][ti]))<1e-6,'Actor output differs');last=y['actions'][ti];actors_checked+=1
   if ti%32==0:
    d.ctrl[ai]=defaults+meta['action_scale']*last;mujoco.mj_forward(m,d);require(np.all(np.abs(d.qfrc_actuator[va])<=m.jnt_actfrcrange[ids,1]+1e-7),'Force cap')
 require(np.all(y['force_max']<=m.jnt_actfrcrange[ids,1]+1e-7),'Recorded peak force')
 if variant=='baseline' and yaw==0:
  prior=np.load(a.old_root/f'finger_limit_localization/{stem}.npz');require(all(np.array_equal(y[k],prior[k]) for k in y.files),'Previous walking baseline differs')
 s=y['signals'];t=y['time'];mask=t>=t[-1]-10;h=np.unwrap(s[:,24]);metric=dict(forward_rmse=float(np.sqrt(np.mean((s[mask,13]-speed)**2))),mean_world_wz=float(s[mask,12].mean()),mean_body_wz=float(s[mask,18].mean()),world_yaw_rmse=float(np.sqrt(np.mean((s[mask,12]-yaw)**2))),mean_world_yaw_error=float(s[mask,12].mean()-yaw),heading_slope=float(np.polyfit(t[mask],h[mask],1)[0]),max_tilt_deg=float(np.rad2deg(s[:,20].max())))
 for key,value in metric.items():require(abs(value-recorded[key])<1e-12,'Metric mismatch '+key)
 rows.append(dict(variant=variant,speed=speed,yaw_command=yaw,**metric))
require(len({(r['variant'],r['speed'],r['yaw_command']) for r in rows})==18,'Duplicate grid')
write_csv('walking_metrics.csv',rows);write_csv('branch_verified.csv',branch_rows)
proof=dict(status='PASS',states_checked=states_checked,actor_outputs_checked=actors_checked,branch_baseline_all_arrays_exact=True,walking_zero_yaw_baselines_all_arrays_exact=True,selected=expected,scope='All recorded states/signals/actor outputs; model-array isolation, initial joint state, force-cap samples/recorded peaks, input hashes and metrics. Not full per-physics-step replay or acceptance of detector/recovery.')
(a.output/'verification.json').write_text(json.dumps(proof,indent=2));(a.output/'raw_hashes.json').write_text(json.dumps(raw_hashes,indent=2));print(json.dumps(proof),flush=True)
