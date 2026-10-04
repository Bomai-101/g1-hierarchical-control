"""Experimental bounded yaw command; scalar/vector diagnostic helper."""
import numpy as np


def heading_command(heading,target,elapsed,gain,limit=.2):
    heading=np.asarray(heading);target=np.asarray(target);elapsed=np.asarray(elapsed)
    if not np.all(np.isfinite(heading)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(elapsed)) or not np.isfinite(gain) or gain<0 or limit<=0:raise ValueError('Finite heading/time and nonnegative gain required')
    error=(target-heading+np.pi)%(2*np.pi)-np.pi
    u=np.clip((elapsed-2.)/.5,0,1);w=u*u*(3-2*u)
    return np.clip(gain*error,-limit,limit)*w
