"""Fixed-1499 walking snapshots and held-target PhysX branch diagnostics."""
import argparse, hashlib, json, sys
from pathlib import Path


def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('reference-root','run-root','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--source',type=Path);p.add_argument('--contact-free',action='store_true');p.add_argument('--lifted',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 sys.path[:0]=[str(a.reference_root/'isaac_sim'),str(a.reference_root/'mujoco')];app=AppLauncher(a).app;env=None
 try:
  import gymnasium as gym
  import numpy as np
  import torch
  import g1_walk_sim51
  from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
  from isaaclab.utils.io import dump_yaml
  from mujoco_eval.policy import TorchActorPolicy
  meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos'],np.float32)
  policy_path=a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz'
  cfg=load_cfg_from_registry('G1-Walk-Flat-Sim51-Play-v0','env_cfg_entry_point');cfg.scene.num_envs=32 if a.contact_free else 16;cfg.sim.device=a.device;cfg.seed=42;cfg.episode_length_s=60;cfg.observations.policy.enable_corruption=False
  cfg.commands.base_velocity.heading_command=False;cfg.commands.base_velocity.rel_standing_envs=0;cfg.commands.base_velocity.resampling_time_range=(10000,10000)
  cfg.commands.base_velocity.ranges.lin_vel_x=(1,1);cfg.commands.base_velocity.ranges.lin_vel_y=(0,0);cfg.commands.base_velocity.ranges.ang_vel_z=(0,0)
  cfg.events.physics_material.params['static_friction_range']=(.8,.8);cfg.events.physics_material.params['dynamic_friction_range']=(.6,.6)
  cfg.events.reset_base.params['pose_range']={'x':(0,0),'y':(0,0),'yaw':(0,0)}
  if a.contact_free:cfg.scene.robot.spawn.rigid_props.disable_gravity=True
  if a.contact_free or a.lifted:cfg.scene.robot.init_state.pos=(0,0,2)
  env=gym.make('G1-Walk-Flat-Sim51-Play-v0',cfg=cfg);env.reset(seed=42);base=env.unwrapped;robot=base.scene['robot'];sensor=base.scene['contact_forces'];dt=float(cfg.sim.dt)
  if list(robot.joint_names)!=names or dt!=.005:raise RuntimeError('Contract changed')
  def arr(v):return v.detach().cpu().numpy().copy()
  origin=arr(base.scene.env_origins);body_names=list(robot.body_names);sensor_names=list(sensor.body_names);feet=[sensor_names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link')]
  if a.source is None:
   policy=TorchActorPolicy(policy_path);walk={k:[] for k in ('root_hybrid','root_link','root_com','joint_pos','joint_vel','target','body_pose','contact_force','obs','action')}
   for tick in range(300):
    base.command_manager.get_term('base_velocity')._update_command();obs=base.observation_manager.compute()['policy'];action=np.stack([policy.act(v) for v in arr(obs)]);at=torch.tensor(action,device=robot.device)
    for k in ('root_hybrid','root_link','root_com'):
     attribute={'root_hybrid':'root_state_w','root_link':'root_link_state_w','root_com':'root_com_state_w'}[k];v=arr(getattr(robot.data,attribute))[0];v[:3]-=origin[0];walk[k].append(v)
    bp=arr(robot.data.body_link_pose_w)[0];bp[:,:3]-=origin[0];walk['body_pose'].append(bp)
    for k,v in [('joint_pos',robot.data.joint_pos),('joint_vel',robot.data.joint_vel),('contact_force',sensor.data.net_forces_w),('obs',obs)]:walk[k].append(arr(v)[0])
    walk['action'].append(action[0]);walk['target'].append(defaults+float(meta['action_scale'])*action[0])
    _,_,terminated,truncated,_=env.step(at)
    if bool(terminated[0] or truncated[0]):raise RuntimeError('Walking source terminated')
   w={k:np.array(v) for k,v in walk.items()};w['time']=np.arange(300)*.02;np.savez_compressed(a.output/'walking.npz',**w)
   cases=[];indices=[]
   fz=w['contact_force'][:,feet,2];support=fz>20
   for label,mask in [('left_only',support[:,0]&~support[:,1]),('right_only',~support[:,0]&support[:,1]),('double',support[:,0]&support[:,1])]:
    choices=np.flatnonzero(mask&(w['time']>=1))
    if len(choices)<2:raise RuntimeError('Missing support class '+label)
    chosen=[int(choices[len(choices)//3]),int(choices[2*len(choices)//3])]
    for sample in chosen:cases.append(dict(name=f'{label}_{sample}',support=label,source_sample=sample,source_time=float(w['time'][sample]),source_foot_fz=fz[sample].tolist(),target_kind='recorded_hold'));indices.append(sample)
   pos=w['joint_pos'][indices];vel=w['joint_vel'][indices];roots=w['root_hybrid'][indices];targets=w['target'][indices]
   np.savez_compressed(a.output/'initial.npz',joint_pos=pos,joint_vel=vel,root_hybrid=roots,target=targets)
   (a.output/'cases.json').write_text(json.dumps(cases,indent=2))
  else:
   w=np.load(a.source/'initial.npz');cases=json.loads((a.source/'cases.json').read_text());pos=list(w['joint_pos']);vel=list(w['joint_vel']);roots=list(w['root_hybrid']);targets=list(w['target'])
   if a.contact_free:
    for i,n in enumerate(names):
     if any(s in n for s in ('hip_','knee_','ankle_')):
      for delta in (-.2,.2):
       tar=defaults.copy();tar[i]+=delta;pos.append(defaults.copy());vel.append(np.zeros(37,np.float32));roots.append(np.r_[0,0,.74,1,0,0,0,np.zeros(6)].astype(np.float32));targets.append(tar);cases.append(dict(name=f'large_{n}_{delta:+g}',joint=n,delta=delta,target_kind='large_step'))
   pos=np.array(pos);vel=np.array(vel);roots=np.array(roots);targets=np.array(targets)
  count=len(cases);roots=roots.copy();roots[:,:2]=0
  if a.contact_free or a.lifted:roots[:,2]=2
  root=arr(robot.data.root_state_w);jp=arr(robot.data.joint_pos);jv=arr(robot.data.joint_vel);tar=arr(robot.data.default_joint_pos)
  root[:count]=roots;root[:count,:3]+=origin[:count];jp[:count]=pos;jv[:count]=vel;tar[:count]=targets
  def tensor(v):return torch.tensor(v,device=robot.device,dtype=torch.float32)
  robot.reset();sensor.reset();robot.write_root_state_to_sim(tensor(root));robot.write_joint_state_to_sim(tensor(jp),tensor(jv));robot.set_joint_position_target(tensor(tar));robot.set_joint_velocity_target(torch.zeros_like(tensor(tar)));robot.set_joint_effort_target(torch.zeros_like(tensor(tar)));base.scene.write_data_to_sim();base.sim.forward();base.scene.update(dt)
  rec={k:[] for k in ('root_link','root_com','root_hybrid','joint_pos','joint_vel','body_pose','contact_force')}
  for step in range(41):
   for k,attribute in [('root_link','root_link_state_w'),('root_com','root_com_state_w'),('root_hybrid','root_state_w')]:
    v=arr(getattr(robot.data,attribute))[:count];v[:,:3]-=origin[:count];rec[k].append(v)
   bp=arr(robot.data.body_link_pose_w)[:count];bp[:,:,:3]-=origin[:count,None,:];rec['body_pose'].append(bp)
   rec['joint_pos'].append(arr(robot.data.joint_pos)[:count]);rec['joint_vel'].append(arr(robot.data.joint_vel)[:count]);rec['contact_force'].append(arr(sensor.data.net_forces_w)[:count])
   if step<40:robot.set_joint_position_target(tensor(tar));base.scene.write_data_to_sim();base.sim.step(render=False);base.scene.update(dt)
  values={k:np.array(v) for k,v in rec.items()};values.update(time=np.arange(41)*dt,target=targets)
  if not all(np.all(np.isfinite(v)) for v in values.values()):raise RuntimeError('Nonfinite branch')
  if (a.contact_free or a.lifted) and np.max(np.abs(values['contact_force'][1:]))>1e-6:raise RuntimeError('Lifted contact')
  np.savez_compressed(a.output/'responses.npz',**values)
  runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,physics_dt=dt,joint_names=names,body_names=body_names,sensor_body_names=sensor_names,cases=cases,contact_free=a.contact_free,no_ground_contact=a.contact_free or a.lifted,stiffness=arr(robot.data.joint_stiffness[0]).tolist(),damping=arr(robot.data.joint_damping[0]).tolist(),armature=arr(robot.data.joint_armature[0]).tolist(),effort_limits=arr(robot.data.joint_effort_limits[0]).tolist(),joint_friction_coeff=arr(robot.data.joint_friction_coeff[0]).tolist(),note='Branch states written fresh; solver/contact history not transferred. t0 contact reading excluded. Held targets, no branch policy feedback; no measured PhysX actuator force claim.')
  (a.output/'runtime.json').write_text(json.dumps(runtime,indent=2));dump_yaml(str(a.output/'env.yaml'),cfg)
  files=[Path(__file__).resolve(),meta_path,policy_path,a.reference_root/'isaac_sim/assets/g1_minimal.usd',a.reference_root/'isaac_sim/g1_walk_sim51/g1_env_cfg.py',a.reference_root/'isaac_sim/g1_walk_sim51/g1_asset_cfg.py']
  if a.source:files += [a.source/f for f in ('initial.npz','cases.json','inputs.json')]
  (a.output/'inputs.json').write_text(json.dumps({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2));(a.output/'complete.json').write_text(json.dumps(dict(cases=count,samples=41,duration=.2,contact_free=a.contact_free,max_contact_force=float(np.abs(values['contact_force'][1:]).max())),indent=2));print('LEG_CONTACT_ISAAC_COMPLETE',count,flush=True)
 except BaseException:
  import traceback
  (a.output/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
