"""On-policy candidate actions in a state-reset and deterministic blend environment."""
import numpy as np,torch
from isaaclab.envs import ManagerBasedRLEnv
from skills_sprint_task import make_cfg


def handoff_cfg(skill,n,seed,device):
 cfg=make_cfg(skill,n,seed,device);cfg.commands.base_velocity.ranges.lin_vel_x=(0.,0.);cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.);return cfg


class TeacherHandoffEnv(ManagerBasedRLEnv):
 def __init__(self,cfg,bank_path,**kwargs):
  self._bank_path=bank_path;self._bank=None;super().__init__(cfg,**kwargs)
  with np.load(bank_path) as z:self._bank={k:torch.as_tensor(z[k],device=self.device) for k in ('root','joint_pos','joint_vel','action','prev_action')}
  self.entry_actions=torch.zeros(self.num_envs,37,device=self.device);self.bank_entry=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);self.reset_counts=[0,0];self.restore_max_error=0.

 def _reset_idx(self,env_ids):
  super()._reset_idx(env_ids)
  if self._bank is None:return
  ids=torch.as_tensor(env_ids,device=self.device);cmd=self.command_manager.get_term('base_velocity');cmd.warm_s[ids]=0.;cmd.incoming[ids]=0.;cmd.active[ids]=False
  mask=torch.rand(len(ids),device=self.device)<.7;selected=ids[mask];self.bank_entry[ids]=mask;self.entry_actions[ids]=0.;self.reset_counts[0]+=int(mask.sum());self.reset_counts[1]+=int((~mask).sum())
  if len(selected):
   ix=torch.randint(len(self._bank['root']),(len(selected),),device=self.device);robot=self.scene['robot'];root=self._bank['root'][ix].clone();root[:,:3]+=self.scene.env_origins[selected];q=self._bank['joint_pos'][ix];qv=self._bank['joint_vel'][ix];action=self._bank['action'][ix]
   robot.write_root_state_to_sim(root,env_ids=selected);robot.write_joint_state_to_sim(q,qv,env_ids=selected)
   self.action_manager._action[selected]=action;self.action_manager._prev_action[selected]=self._bank['prev_action'][ix];self.entry_actions[selected]=action
   robot.set_joint_position_target(robot.data.default_joint_pos[selected]+.5*action,env_ids=selected)
   for actual,expected in ((robot.data.root_state_w[selected],root),(robot.data.joint_pos[selected],q),(robot.data.joint_vel[selected],qv),(self.action_manager.action[selected],action)):
    error=float((actual-expected).abs().max());self.restore_max_error=max(self.restore_max_error,error)
    if error>1e-5:raise RuntimeError('Teacher state restoration mismatch')
  cmd._update_command()

 def step(self,action):
  elapsed=self.episode_length_buf*self.step_dt;u=(elapsed/.5).clamp(0,1);w=u*u*(3-2*u);blend=self.entry_actions*(1-w[:,None])+action*w[:,None];applied=torch.where(self.bank_entry[:,None],blend,action)
  return super().step(applied)
