"""Capture PhysX contact-free responses to matched joint targets; no policy loop."""
import argparse,json,hashlib,sys
from pathlib import Path

def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(a.reference_root/'isaac_sim'));app=AppLauncher(a).app;env=None
 try:
  import gymnasium as gym
  import numpy as np
  import torch
  import g1_walk_sim51
  from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
  from isaaclab.utils.io import dump_yaml
  meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos'],dtype=np.float32)
  audit_path=Path('results/flat_baseline/20261002/usd_physics_isolation/bilateral_audit.json');audit=json.loads(audit_path.read_text());sign=np.array(audit['mirror_signs']);real_path=a.run_root/'interface_live/baseline1499.npz';real=np.load(real_path)
  cases=[dict(name='hold_default',joint=None,delta=0.,source_sample=None)];poses=[defaults.copy()];velocities=[np.zeros(37,dtype=np.float32)];roots=[np.r_[0,0,2,1,0,0,0,np.zeros(6)].astype(np.float32)];targets=[defaults.copy()]
  props=json.loads((a.run_root/'usd_matched_properties/properties.json').read_text())
  # Use each left joint's feasible small offset and mirror it for the right joint.
  for i,n in enumerate(names):
   if not n.startswith(('left_','right_')):continue
   left=n.replace('right_','left_',1);li=names.index(left);delta=-.01 if left in ('left_six_joint','left_four_joint') else .01
   if n.startswith('right_'):delta*=sign[li]
   target=defaults.copy();target[i]+=delta;cases.append(dict(name='step_'+n,joint=n,delta=delta,source_sample=None));poses.append(defaults.copy());velocities.append(np.zeros(37,dtype=np.float32));roots.append(roots[0].copy());targets.append(target)
  for i,n in enumerate(names):
   if n.startswith(('left_','right_')):continue
   for delta in (-.01,.01):
    target=defaults.copy();target[i]+=delta;cases.append(dict(name=f'step_{n}_{delta:+g}',joint=n,delta=delta,source_sample=None));poses.append(defaults.copy());velocities.append(np.zeros(37,dtype=np.float32));roots.append(roots[0].copy());targets.append(target)
  for sample in (50,100,150):
   root=real['root'][sample].copy();root[:3]=[0,0,2];cases.append(dict(name=f'walking_state_{sample}',joint=None,delta=None,source_sample=sample));roots.append(root);poses.append(real['isaac_obs'][sample,12:49]+defaults);velocities.append(real['joint_vel'][sample]);targets.append(real['expected_target'][sample])
  cfg=load_cfg_from_registry('G1-Walk-Flat-Sim51-Play-v0','env_cfg_entry_point');cfg.scene.num_envs=len(cases);cfg.sim.device=a.device;cfg.seed=42;cfg.scene.robot.spawn.rigid_props.disable_gravity=True;cfg.scene.robot.init_state.pos=(0,0,2);cfg.episode_length_s=60;cfg.observations.policy.enable_corruption=False
  env=gym.make('G1-Walk-Flat-Sim51-Play-v0',cfg=cfg);env.reset(seed=42);base=env.unwrapped;robot=base.scene['robot'];dt=float(cfg.sim.dt)
  if list(robot.joint_names)!=names or abs(dt-.005)>1e-12:raise RuntimeError('Joint order/physics rate changed')
  device=robot.device;root=torch.tensor(np.array(roots),device=device);root[:,:3]+=base.scene.env_origins;jpos=torch.tensor(np.array(poses),device=device);jvel=torch.tensor(np.array(velocities),device=device);target=torch.tensor(np.array(targets),device=device)
  robot.reset();robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(jpos,jvel);robot.set_joint_position_target(target);robot.set_joint_velocity_target(torch.zeros_like(target));robot.set_joint_effort_target(torch.zeros_like(target));base.scene.write_data_to_sim();base.sim.forward();base.scene.update(dt)
  def arr(x):return x.detach().cpu().numpy().copy()
  records={k:[] for k in ('root_link','root_com','root_hybrid','joint_pos','joint_vel','body_pose','contact_force')}
  for step in range(41):
   # Root writer uses link pose + COM world velocities; save both explicit definitions.
   link=arr(robot.data.root_link_state_w);com=arr(robot.data.root_com_state_w);hybrid=arr(robot.data.root_state_w);origin=arr(base.scene.env_origins);link[:,:3]-=origin;com[:,:3]-=origin;hybrid[:,:3]-=origin
   records['root_link'].append(link);records['root_com'].append(com);records['root_hybrid'].append(hybrid);records['joint_pos'].append(arr(robot.data.joint_pos));records['joint_vel'].append(arr(robot.data.joint_vel))
   bp=arr(robot.data.body_link_pose_w);bp[:,:,:3]-=origin[:,None,:];records['body_pose'].append(bp);records['contact_force'].append(arr(base.scene['contact_forces'].data.net_forces_w))
   if not np.all(np.isfinite(records['joint_pos'][-1])):raise RuntimeError('Nonfinite response')
   if step<40:robot.set_joint_position_target(target);base.scene.write_data_to_sim();base.sim.step(render=False);base.scene.update(dt)
  values={k:np.array(v) for k,v in records.items()};values.update(time=np.arange(41)*dt,target=arr(target),requested_joint_pos=arr(jpos),requested_joint_vel=arr(jvel),requested_root_hybrid=arr(root)-np.c_[arr(base.scene.env_origins),np.zeros((len(cases),10))])
  np.savez_compressed(a.output/'responses.npz',**values)
  if np.max(np.abs(values['contact_force']))>1e-6:raise RuntimeError('Suspended capture has ground contact')
  runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,physics_dt=dt,policy_steps=0,joint_names=names,body_names=list(robot.body_names),stiffness=arr(robot.data.joint_stiffness[0]).tolist(),damping=arr(robot.data.joint_damping[0]).tolist(),armature=arr(robot.data.joint_armature[0]).tolist(),effort_limits=arr(robot.data.joint_effort_limits[0]).tolist(),joint_friction_coeff=arr(robot.data.joint_friction_coeff[0]).tolist(),cases=cases,note='Free-floating,gravitydisabled,2mheight,nocontacts. Static joint steps and three lifted actualIsaac walkingstates; held recorded targets, no policy feedback. root_hybrid is linkpose+COMworldvelocity, root_link/root_com saved explicitly. ImplicitActuator computed/applied torque is an approximation and not logged as measured drive force.')
  (a.output/'runtime.json').write_text(json.dumps(runtime,indent=2));dump_yaml(str(a.output/'env.yaml'),cfg)
  files=[Path(__file__).resolve(),meta_path,audit_path,real_path,a.run_root/'usd_matched_properties/properties.json',a.reference_root/'isaac_sim/assets/g1_minimal.usd',a.reference_root/'isaac_sim/g1_walk_sim51/g1_asset_cfg.py',a.reference_root/'isaac_sim/g1_walk_sim51/g1_env_cfg.py',Path('/home/omai/robotics/reference_projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py'),Path('/home/omai/robotics/reference_projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py')]
  (a.output/'inputs.json').write_text(json.dumps({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2));(a.output/'complete.json').write_text(json.dumps(dict(cases=len(cases),samples=41,duration=.2,max_contact_force=float(np.abs(values['contact_force']).max()),source_policy='locallytrained1499 previoustrace; no newlyloadedpolicy'),indent=2));print('MATCHED_ISAAC_COMPLETE',len(cases),41,flush=True)
 except BaseException:
  import traceback
  (a.output/'failure.txt').write_text(traceback.format_exc())
  raise
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
