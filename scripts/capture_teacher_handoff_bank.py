"""Frozen1499 first-episode physical states, independent training seed43."""
import argparse,json,hashlib,sys
from pathlib import Path

def main():
 from isaaclab.app import AppLauncher
 p=argparse.ArgumentParser()
 for n in ('reference-root','checkpoint','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--num-envs',type=int,default=512);AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'isaac_sim'));app=AppLauncher(a).app;env=None
 try:
  import numpy as np,torch,gymnasium as gym,g1_walk_sim51
  from skills_sprint_task import make_cfg
  cfg=make_cfg('hold',a.num_envs,43,a.device);cfg.episode_length_s=30.;cfg.observations.policy.enable_corruption=False
  env=gym.make('G1-Walk-Flat-Sim51-v0',cfg=cfg).unwrapped;robot=env.scene['robot'];cmd=env.command_manager.get_term('base_velocity');env.reset(seed=43);cmd.warm_s[:]=999.;cmd.incoming[:,0]=torch.empty(a.num_envs,device=a.device).uniform_(.3,1.);cmd.incoming[:,1]=0.;yaw=torch.empty(a.num_envs,device=a.device).uniform_(-.2,.2);cmd.incoming[:,2]=0.
  state=torch.load(a.checkpoint,map_location=a.device,weights_only=False)['model_state_dict'];keys=sorted([k for k in state if k.startswith('actor.') and k.endswith('.weight')],key=lambda k:int(k.split('.')[1]));layers=[(state[k],state[k.replace('weight','bias')]) for k in keys];last=torch.zeros(a.num_envs,37,device=a.device);alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device);rows={k:[] for k in ('root','joint_pos','joint_vel','action','prev_action','command','time')}
  for tick in range(301):
   now=tick*.02;cmd.incoming[:,2]=yaw if now>=2.4 else 0.;cmd._update_command();alive&=robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]>=.35
   if tick>=150 and tick%5==0:
    ids=alive.nonzero().flatten();root=robot.data.root_state_w[ids].clone();root[:,:3]-=env.scene.env_origins[ids];values=dict(root=root,joint_pos=robot.data.joint_pos[ids],joint_vel=robot.data.joint_vel[ids],action=last[ids],prev_action=env.action_manager.prev_action[ids],command=cmd.command[ids],time=torch.full((len(ids),),now,device=a.device))
    for k,v in values.items():rows[k].append(v.cpu().numpy().copy())
   if tick==300:break
   obs=torch.cat((robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,robot.data.projected_gravity_b,cmd.command,robot.data.joint_pos-robot.data.default_joint_pos,robot.data.joint_vel-robot.data.default_joint_vel,last),dim=1)
   with torch.inference_mode():
    x=obs
    for i,(w,b) in enumerate(layers):
     x=torch.nn.functional.linear(x,w,b)
     if i<len(layers)-1:x=torch.nn.functional.elu(x)
    last=x
   _,_,done,truncated,_=env.step(last);alive&=~(done|truncated)
  arrays={k:np.concatenate(v) for k,v in rows.items()};count=len(arrays['root'])
  if count<1000 or any(not np.isfinite(v).all() for v in arrays.values()):raise RuntimeError('Invalid teacher bank')
  np.savez_compressed(a.output/'bank.npz',**arrays)
  result=dict(samples=count,seed=43,teacher_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),bank_sha256=hashlib.sha256((a.output/'bank.npz').read_bytes()).hexdigest(),vx_range=[.3,1.],yaw_range=[-.2,.2],sample_times_s=[3.,6.],stride_s=.1,joint_names=robot.joint_names,first_episode_only=True,physics_dt=.005,policy_dt=.02,scope='Root/joint states and two previous actions only; reset transplantation does not preserve PhysX contact solver/sensor history; evaluation seed42 not used.')
  (a.output/'complete.json').write_text(json.dumps(result,indent=2));print('TEACHER_BANK_COMPLETE',result,flush=True)
 finally:
  if env is not None:env.close()
  app.close()
if __name__=='__main__':main()
