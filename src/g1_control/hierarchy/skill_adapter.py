"""Explicit internal phase adapter; does not register skills or switch authority."""
import numpy as np

def skill_observation(base_observation,skill,elapsed_s,phase_offset=0.):
    base=np.asarray(base_observation,dtype=np.float32)
    if base.shape!=(123,) or not np.all(np.isfinite(base)):raise ValueError('Finite123D base observation required')
    if skill=='hold':return base.copy()
    if skill!='march' or not np.isfinite(elapsed_s) or elapsed_s<0 or not np.isfinite(phase_offset):raise ValueError('Valid skill/time/phase required')
    phase=elapsed_s*2*np.pi/.9+phase_offset
    return np.r_[base,np.float32(np.sin(phase)),np.float32(np.cos(phase))].astype(np.float32)

def handoff_action(previous_action,proposal,elapsed_s,blend_s=.5):
    previous=np.asarray(previous_action,dtype=float);target=np.asarray(proposal,dtype=float)
    if previous.shape!=(37,) or target.shape!=(37,) or not np.all(np.isfinite(np.r_[previous,target])):raise ValueError('Finite37D actions required')
    if not np.isfinite(elapsed_s) or elapsed_s<0 or not np.isfinite(blend_s) or blend_s<=0:raise ValueError('Valid time required')
    u=np.clip(elapsed_s/blend_s,0,1);w=u*u*(3-2*u)
    return previous*(1-w)+target*w
