"""Compare all nine independent command responses from recorded traces."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);colors={'hold':'#f97316','zero':'#2563eb','ramp':'#10b981'}
fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
for row,direction in enumerate(('right','straight','left')):
 for response,color in colors.items():
  x=np.load(a.run/(direction+'_'+response)/'trace.npz');i=261;t=x['time'][i:]-5.22;s=x['signals'][i:];speed=np.linalg.norm(s[:,13:15],axis=1);heading=np.rad2deg(s[:,20]-s[0,20]);axes[row,0].plot(t,speed,color=color,label=response);axes[row,1].plot(t,heading,color=color);axes[row,2].plot(s[:,0]-s[0,0],s[:,1]-s[0,1],color=color,label=response)
 axes[row,0].axhline(.05,color='black',linestyle='--',alpha=.6);axes[row,0].set_ylabel(direction+'\nBody COM horizontal speed (m/s)');axes[row,1].set_ylabel('Relative Euler heading (deg)');axes[row,2].set_ylabel('World delta y (m)');axes[row,2].set_aspect('equal',adjustable='datalim')
 for col in (0,1):axes[row,col].set_xlabel('Seconds after command response')
 axes[row,2].set_xlabel('World delta x (m)')
 for ax in axes[row]:ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Fixed1499 command response: zero command reduces travel but does not stop drift\nIndependent experiment, 10 s after switch; no watchdog or fallback integration',fontsize=13);fig.savefig(a.output/'command_response.png',dpi=150);fig.savefig(a.output/'command_response.svg');plt.close(fig)
rows=json.loads((a.run/'complete.json').read_text())['results'];post=[]
for r in rows:
 x=np.load(a.run/r['label']/'trace.npz');s=x['signals'];i=261
 for seconds in (1.,2.,10.):
  j=i+round(seconds/.02);post.append(dict(label=r['label'],window_s=seconds,path_m=float(np.sum(np.linalg.norm(np.diff(s[i:j+1,:2],axis=0),axis=1))),heading_change_rad=float(s[j,20]-s[i,20]),net_displacement_m=float(np.linalg.norm(s[j,:2]-s[i,:2]))))
(a.output/'window_metrics.json').write_text(json.dumps(post,indent=2))
