"""Offline yaw windows with policy-rate, zero-order-held commands."""
import numpy as np
from g1_control.monitoring.yaw_window import _series, _windows


def scheduled_yaw_stats(times, body_rates, headings, commands, window_s):
    """Integrate measured motion against the command actually held per interval.

    A command logged at t_i applies to [t_i,t_(i+1)); a new command cannot
    retroactively change the preceding interval. Body-rate samples interpolate
    linearly; unwrapped heading gives constant backward interval rates.
    """
    t,body=_series(times,body_rates)
    _,heading=_series(t,headings)
    _,command=_series(t,commands)
    starts,valid=_windows(t,window_s)
    dt=np.diff(t)
    slope=np.diff(body)/dt
    left=body[:-1]-command[:-1]
    right=body[1:]-command[:-1]
    integral=np.r_[0.,np.cumsum(dt*(left+right)/2)]
    squared=np.r_[0.,np.cumsum(dt*(left**2+left*right+right**2)/3)]
    i=np.clip(np.searchsorted(t,starts,side='right')-1,0,len(t)-2)
    x=starts-t[i]
    at_start=integral[i]+left[i]*x+slope[i]*x*x/2
    square_start=squared[i]+left[i]**2*x+left[i]*slope[i]*x*x+slope[i]**2*x**3/3
    heading_error_rate=np.diff(heading)/dt-command[:-1]
    hi=np.r_[0.,np.cumsum(heading_error_rate*dt)]
    hs=np.r_[0.,np.cumsum(heading_error_rate**2*dt)]
    reference_heading=np.r_[heading[0],heading[0]+np.cumsum(command[:-1]*dt)]
    return dict(body_mean=np.where(valid,(integral-at_start)/window_s,np.nan),
        body_rmse=np.where(valid,np.sqrt(np.maximum(0.,(squared-square_start)/window_s)),np.nan),
        heading_mean=np.where(valid,(hi-np.interp(starts,t,hi))/window_s,np.nan),
        heading_rmse=np.where(valid,np.sqrt(np.maximum(0.,(hs-np.interp(starts,t,hs))/window_s)),np.nan),
        reference_heading=reference_heading)
