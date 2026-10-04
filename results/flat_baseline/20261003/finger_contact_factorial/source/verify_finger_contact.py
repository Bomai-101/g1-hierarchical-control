"""Verify all 24 factorial walking traces and unchanged control reproductions."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import mujoco
import numpy as np
from finger_contact_variants import apply_variant,FINGERS,VARIANTS
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
walk=a.run_root/'finger_contact_factorial';plan=json.loads((a.report/'plan.json').read_text());expected=plan['selected'];require(tuple(expected)==VARIANTS,'Plan variants')
model_path=a.old_root/'usd_physical_models/g1_usd_physical_position.mjb';meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz')
for path,digest in json.loads((walk/'inputs.json').read_text()).items():require(sha(Path(path))==digest,'Changed input '+path)
raw_hashes={str(p.resolve()):sha(p) for p in walk.iterdir() if p.is_file()};states_checked=0;actors_checked=0;controls_checked=0
for variant in expected:
 original=mujoco.MjModel.from_binary_path(str(model_path));m=mujoco.MjModel.from_binary_path(str(model_path));apply_variant(m,variant)
 for name in dir(m):
  value=getattr(m,name)
  if isinstance(value,np.ndarray):
   wanted=getattr(original,name).copy()
   if name=='jnt_solref' and variant in ('finger_only','combined'):
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in FINGERS];wanted[ids,0]=.002
   if name=='geom_solref' and variant in ('contact_only','combined'):wanted[(m.geom_contype!=0)|(m.geom_conaffinity!=0),0]=.005
   require(np.array_equal(value,wanted),'Model isolation '+name)
complete=json.loads((walk/'complete.json').read_text());require(len(complete['results'])==24,'Walking grid');rows=[]
for recorded in complete['results']:
 variant=recorded['variant'];speed=recorded['speed'];yaw=recorded['yaw_command'];require(variant in expected and speed in plan['speeds'] and yaw in plan['yaw_commands'],'Grid entry');stem=f'{variant}_vx{speed:g}_yaw{yaw:+.1f}';y=np.load(walk/f'{stem}.npz');m=mujoco.MjModel.from_binary_path(str(model_path));apply_variant(m,variant);d=mujoco.MjData(m);ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);require(y['states'].shape==(1001,87) and y['actions'].shape==(1000,37),'Walking count');require(np.max(abs(y['time']-np.arange(1001)*.02))<1e-10,'Walking time');require(not recorded['fell'] and not recorded['numerical_failure'] and recorded['physical_steps']==20000,'Unsuccessful walking');command=np.array([speed,0,yaw]);last=np.zeros(37)
 for ti,state in enumerate(y['states']):
  require(np.all(np.isfinite(state)),'Nonfinite state');d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang);g=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-g[2],-1,1));heading=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2));signal=np.r_[d.qpos[:3],quat,lin,ang,blin,bang,height,tilt,command,heading];require(np.max(abs(signal-y['signals'][ti]))<1e-11,'Rebuilt walking signal');require(height>=.35,'Height termination');states_checked+=1
  if ti<1000:
   obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last);action=actor.act(obs);require(np.max(abs(action-y['actions'][ti]))<1e-6,'Actor output differs');last=y['actions'][ti];actors_checked+=1
   if ti%32==0:
    d.ctrl[ai]=defaults+meta['action_scale']*last;mujoco.mj_forward(m,d);require(np.all(np.abs(d.qfrc_actuator[va])<=m.jnt_actfrcrange[ids,1]+1e-7),'Force cap')
 require(np.all(y['force_max']<=m.jnt_actfrcrange[ids,1]+1e-7),'Recorded peak force')
 if variant in ('baseline','contact_only'):
  old_variant='baseline' if variant=='baseline' else 'contact_time005';prior=np.load(a.run_root/'contact_parameter_walking'/f'{old_variant}_vx{speed:g}_yaw{yaw:+.1f}.npz');require(all(np.array_equal(y[k],prior[k]) for k in y.files),'Previous control differs');controls_checked+=1
 if variant=='finger_only' and yaw==0:
  prior=np.load(a.old_root/'finger_limit_localization'/f'active4_vx{speed:g}_yaw+0.0.npz');require(all(np.array_equal(y[k],prior[k]) for k in y.files),'Previous finger control differs');controls_checked+=1
 s=y['signals'];t=y['time'];mask=t>=t[-1]-10;h=np.unwrap(s[:,24]);metric=dict(forward_rmse=float(np.sqrt(np.mean((s[mask,13]-speed)**2))),mean_world_wz=float(s[mask,12].mean()),mean_body_wz=float(s[mask,18].mean()),world_yaw_rmse=float(np.sqrt(np.mean((s[mask,12]-yaw)**2))),mean_world_yaw_error=float(s[mask,12].mean()-yaw),heading_slope=float(np.polyfit(t[mask],h[mask],1)[0]),max_tilt_deg=float(np.rad2deg(s[:,20].max())))
 for key,value in metric.items():require(abs(value-recorded[key])<1e-12,'Metric mismatch '+key)
 rows.append(dict(variant=variant,speed=speed,yaw_command=yaw,**metric))
require(len({(r['variant'],r['speed'],r['yaw_command']) for r in rows})==24,'Duplicate grid')
write_csv('walking_metrics.csv',rows)
interactions=[]
for speed in plan['speeds']:
 for yaw in plan['yaw_commands']:
  cells={r['variant']:r for r in rows if r['speed']==speed and r['yaw_command']==yaw}
  for metric in ('forward_rmse','mean_world_yaw_error','world_yaw_rmse','heading_slope'):
   effects={v:cells[v][metric] for v in expected};interaction=effects['combined']-effects['finger_only']-effects['contact_only']+effects['baseline'];interactions.append(dict(speed=speed,yaw_command=yaw,metric=metric,interaction=interaction,**effects))
write_csv('interactions.csv',interactions)
proof=dict(status='PASS',states_checked=states_checked,actor_outputs_checked=actors_checked,previous_control_arrays_exact=controls_checked,model_array_isolation=True,scope='All sampled states/signals/actor outputs, time/grid/metrics, force-cap samples and peaks, original-input hashes, factorial array changes. Not per-physics-step replay, multi-seed or heading-task acceptance.')
(a.output/'verification.json').write_text(json.dumps(proof,indent=2));(a.output/'raw_hashes.json').write_text(json.dumps(raw_hashes,indent=2));print(json.dumps(proof),flush=True)
