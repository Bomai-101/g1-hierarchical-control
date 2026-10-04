"""Pure NumPy trajectory metrics; no simulator import required."""
import numpy as np

def settling_evidence(t,speed,yaw_rate,window=.5,dwell=1.,speed_limit=.05,yaw_limit=.05):
    """Causal trailing windows, full-window samples, distinct start/confirmation."""
    dt=float(t[1]-t[0]);n=round(window/dt)+1;need=round(dwell/dt)+1
    smooth_speed=np.full(len(t),np.nan);smooth_yaw=np.full(len(t),np.nan)
    for i in range(n-1,len(t)):
        smooth_speed[i]=np.mean(speed[i-n+1:i+1]);smooth_yaw[i]=np.sqrt(np.mean(yaw_rate[i-n+1:i+1]**2))
    eligible=(smooth_speed<=speed_limit)&(smooth_yaw<=yaw_limit)
    start=None;confirmation=None;run=0
    for i,ok in enumerate(eligible):
        run=run+1 if ok else 0
        if run>=need:
            start=float(t[i-need+1]);confirmation=float(t[i]);break
    escape=None
    if confirmation is not None:
        after=np.flatnonzero((t>confirmation+1e-9)&~eligible)
        if len(after):escape=float(t[after[0]])
    return dict(settling_start_s=start,settling_confirmed_s=confirmation,first_escape_after_confirmation_s=escape,eligible_through_end_after_confirmation=confirmation is not None and escape is None)

def metrics(trace,switch=5.22):
    t=trace['time'];s=trace['signals'];i=round(switch/.02);elapsed=t[i:]-switch;post=s[i:]
    speed=np.linalg.norm(post[:,13:15],axis=1)
    # Euler heading differences, not body angular z; backward difference ends at each sample.
    rates=np.r_[np.nan,np.diff(s[:,20])/.02][i:]
    quat=post[:,3:7];w,x,y,z=quat.T
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    final=elapsed>=elapsed[-1]-2.-1e-9;path=float(np.sum(np.linalg.norm(np.diff(post[:,:2],axis=0),axis=1)))
    evidence=settling_evidence(elapsed,speed,rates) if len(elapsed)>1 else {}
    return dict(post_duration_s=float(elapsed[-1]),path_after_switch_m=path,net_displacement_m=float(np.linalg.norm(post[-1,:2]-post[0,:2])),signed_heading_change_rad=float(post[-1,20]-post[0,20]),max_heading_excursion_rad=float(np.max(abs(post[:,20]-post[0,20]))),final_2s_horizontal_speed_rms_mps=float(np.sqrt(np.mean(speed[final]**2))),final_2s_euler_yaw_rate_rms_rps=float(np.sqrt(np.mean(rates[final]**2))),minimum_height_m=float(np.min(post[:,19])),max_abs_roll_deg=float(np.rad2deg(np.max(abs(roll)))),max_abs_pitch_deg=float(np.rad2deg(np.max(abs(pitch)))),**evidence)
