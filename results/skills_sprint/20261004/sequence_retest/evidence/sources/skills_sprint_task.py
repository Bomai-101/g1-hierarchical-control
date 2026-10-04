"""Isolated Isaac tasks for stop/hold and phase-conditioned marching."""
import math
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import RewardTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi
from g1_walk_sim51.g1_env_cfg import G1FlatEnvCfg


class SprintCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.warm_s=torch.zeros(self.num_envs,device=self.device)
        self.incoming=torch.zeros(self.num_envs,3,device=self.device)
        self.active=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.anchor_xy=torch.zeros(self.num_envs,2,device=self.device)
        self.anchor_heading=torch.zeros(self.num_envs,device=self.device)
        self.phase_offset=torch.zeros(self.num_envs,device=self.device)
        self.foot_ids,_=self.robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)
        self.sensor_ids,_=env.scene['contact_forces'].find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)
        if len(self.foot_ids)!=2 or len(self.sensor_ids)!=2:raise RuntimeError('Two ordered feet required')

    def _resample_command(self,env_ids):
        n=len(env_ids);r=torch.empty(n,device=self.device)
        self.warm_s[env_ids]=r.uniform_(1.,3.)
        immediate=torch.rand(n,device=self.device)<.3
        self.warm_s[torch.as_tensor(env_ids,device=self.device)[immediate]]=0.
        self.incoming[env_ids,0]=r.uniform_(.3,.8);self.incoming[env_ids,1]=0.
        self.incoming[env_ids,2]=r.uniform_(-.2,.2)
        self.active[env_ids]=False;self.phase_offset[env_ids]=r.uniform_(0,2*math.pi)
        self.vel_command_b[env_ids]=self.incoming[env_ids]

    def _update_command(self):
        elapsed=self._env.episode_length_buf*self._env.step_dt
        stopped=elapsed>=self.warm_s
        entered=stopped&~self.active
        self.anchor_xy[entered]=self.robot.data.root_pos_w[entered,:2]
        self.anchor_heading[entered]=self.robot.data.heading_w[entered]
        self.active=stopped
        self.vel_command_b[:]=torch.where(stopped[:,None],torch.zeros_like(self.incoming),self.incoming)

    def phase(self):
        elapsed=self._env.episode_length_buf*self._env.step_dt
        return (elapsed-self.warm_s)*2*math.pi/.9+self.phase_offset


@configclass
class SprintCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SprintCommand


def term(env):return env.command_manager.get_term('base_velocity')
def active(env):return term(env).active.float()

def phase_observation(env):
    cmd=term(env);p=cmd.phase()
    return torch.stack((torch.sin(p),torch.cos(p)),dim=-1)*cmd.active[:,None]

def speed_tracking(env):
    c=term(env);error=torch.sum((c.command[:,:2]-c.robot.data.root_lin_vel_b[:,:2])**2,dim=-1)
    return torch.where(c.active,3.*torch.exp(-error/.15**2),torch.exp(-error/.5**2))

def yaw_tracking(env):
    c=term(env);error=(c.command[:,2]-c.robot.data.root_ang_vel_w[:,2])**2
    return torch.where(c.active,2.*torch.exp(-error/.2**2),torch.exp(-error/.5**2))

def stay_near_anchor(env):
    c=term(env);error=torch.sum((c.robot.data.root_pos_w[:,:2]-c.anchor_xy)**2,dim=-1)
    return torch.exp(-error/.15**2)*c.active

def retain_heading(env):
    c=term(env);error=wrap_to_pi(c.robot.data.heading_w-c.anchor_heading)**2
    return torch.exp(-error/.2**2)*c.active

def upright_height(env):
    c=term(env);h=c.robot.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    return torch.exp(-((h-.68)/.08)**2)*c.active

def both_feet_contact(env):
    c=term(env);forces=env.scene['contact_forces'].data.net_forces_w[:,c.sensor_ids].norm(dim=-1)
    return (forces>1.).all(dim=-1).float()*c.active

def hold_joint_motion(env,asset_cfg):
    robot=env.scene['robot'];return torch.sum(robot.data.joint_vel[:,asset_cfg.joint_ids]**2,dim=-1)*active(env)

def march_contact(env):
    c=term(env);s=torch.sin(c.phase());desired=torch.stack((s<=.2,-s<=.2),dim=-1)
    actual=env.scene['contact_forces'].data.net_forces_w[:,c.sensor_ids].norm(dim=-1)>1.
    return (actual==desired).float().mean(dim=-1)*c.active

def march_clearance(env):
    c=term(env);s=torch.sin(c.phase());lift=torch.stack((s.clamp(min=0),(-s).clamp(min=0)),dim=-1)*.06
    actual=c.robot.data.body_pos_w[:,c.foot_ids,2]-env.scene.env_origins[:,2,None]
    # Ankle-roll body origin is above the sole; explicit provisional target.
    target=.055+lift
    return torch.exp(-torch.sum((actual-target)**2,dim=-1)/.035**2)*c.active


def make_cfg(skill,num_envs,seed,device):
    cfg=G1FlatEnvCfg();cfg.scene.num_envs=num_envs;cfg.seed=seed;cfg.sim.device=device;cfg.episode_length_s=12.
    cfg.commands.base_velocity=SprintCommandCfg(asset_name='robot',resampling_time_range=(1000.,1000.),heading_command=False,rel_heading_envs=0.,rel_standing_envs=0.,ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(.3,.8),lin_vel_y=(0.,0.),ang_vel_z=(-.2,.2),heading=(-math.pi,math.pi)))
    cfg.rewards.track_lin_vel_xy_exp=RewardTermCfg(func=speed_tracking,weight=1.)
    cfg.rewards.track_ang_vel_z_exp=RewardTermCfg(func=yaw_tracking,weight=1.)
    cfg.rewards.flat_orientation_l2.weight=-2.
    cfg.rewards.stay_near_anchor=RewardTermCfg(func=stay_near_anchor,weight=1.)
    cfg.rewards.retain_heading=RewardTermCfg(func=retain_heading,weight=1.)
    cfg.rewards.upright_height=RewardTermCfg(func=upright_height,weight=.5)
    if skill=='hold':
        cfg.rewards.both_feet_contact=RewardTermCfg(func=both_feet_contact,weight=.5)
        cfg.rewards.hold_joint_motion=RewardTermCfg(func=hold_joint_motion,weight=-.01,params={'asset_cfg':SceneEntityCfg('robot',joint_names=['.*_hip_.*','.*_knee_joint','.*_ankle_.*'])})
    elif skill=='march':
        cfg.observations.policy.skill_phase=ObservationTermCfg(func=phase_observation)
        cfg.rewards.march_contact=RewardTermCfg(func=march_contact,weight=2.)
        cfg.rewards.march_clearance=RewardTermCfg(func=march_clearance,weight=2.)
    else:raise ValueError('Unknown skill')
    return cfg
