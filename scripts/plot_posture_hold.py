"""Early posture handoff failures; each trace ends at its measured termination."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(3,3,figsize=(13,9),layout='constrained');colors={'zero':'#2563eb','capture_pose':'#f97316','nominal_pose':'#dc2626'}
for row,direction in enumerate(('right','straight','left')):
 for response,color in colors.items():
  x=np.load(a.run/(direction+'_'+response)/'trace.npz');t=x['time'][261:]-5.22;s=x['signals'][261:];w,qx,qy,qz=s[:,3:7].T;roll=np.arctan2(2*(w*qx+qy*qz),1-2*(qx*qx+qy*qy));pitch=np.arcsin(np.clip(2*(w*qy-qz*qx),-1,1));tilt=np.rad2deg(np.maximum(abs(roll),abs(pitch)))
  values=[np.linalg.norm(s[:,13:15],axis=1),s[:,19],tilt]
  for col,y in enumerate(values):
   axes[row,col].plot(t,y,color=color,label=response)
   if response!='zero':axes[row,col].scatter(t[-1],y[-1],color=color,marker='x',s=45)
 for col,label in enumerate(('Body COM horizontal speed (m/s)','Pelvis height (m)','max abs roll/pitch (deg)')):
  axes[row,col].set_ylabel(direction+'\n'+label);axes[row,col].set_xlabel('Seconds after handoff');axes[row,col].set_xlim(0,2);axes[row,col].grid(alpha=.2)
 axes[row,1].axhline(.35,color='black',linestyle='--',label='Height termination')
axes[0,0].legend(fontsize=9);fig.suptitle('Compatible interface does not ensure balance: six posture-hold attempts fail\nFirst 2 s shown; crosses end traces at height failure; actor-zero baseline observed for 10 s',fontsize=12);fig.savefig(a.output/'posture_hold.png',dpi=150);fig.savefig(a.output/'posture_hold.svg');plt.close(fig)
