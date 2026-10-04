"""Figures from saved metrics only; no simulation."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args();branches=list(csv.DictReader((a.report/'branch_metrics.csv').open()));walk=list(csv.DictReader((a.report/'walking_metrics.csv').open()));variants=list(dict.fromkeys(r['variant'] for r in branches));chosen=list(dict.fromkeys(r['variant'] for r in walk));colors=['#2563eb','#10b981','#f97316'];fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
x=np.arange(len(variants))
for i,split in enumerate(('development','heldout')):
 values=[np.sqrt(np.mean([float(r['leg_rmse_rad'])**2 for r in branches if r['variant']==v and r['split']==split]))*180/np.pi for v in variants];axes[0,0].bar(x+(i-.5)*.36,values,.36,label=split)
axes[0,0].set_xticks(x,['base','mu .6','mu 1.0','tc .005','tc .010','tc .040'],rotation=30,ha='right');axes[0,0].set_title('Held-target leg RMSE (degrees)');axes[0,0].legend(fontsize=8)
for col,speed in enumerate((.5,1.),start=1):
 for variant,color in zip(chosen,colors):
  rr=sorted([r for r in walk if r['variant']==variant and float(r['speed'])==speed],key=lambda r:float(r['yaw_command']));xx=[float(r['yaw_command']) for r in rr]
  axes[0,col].plot(xx,[float(r['forward_rmse']) for r in rr],'-o',color=color,label=variant);axes[1,col].plot(xx,[float(r['mean_world_wz']) for r in rr],'-o',color=color,label=variant)
 axes[0,col].set_title(f'Forward velocity RMSE (m/s), vx={speed:g}');axes[1,col].plot([-.2,.2],[-.2,.2],'k--',label='command');axes[1,col].set_title(f'Mean world yaw rate (rad/s), vx={speed:g}')
 for row in (0,1):axes[row,col].set_xlabel('Yaw-rate command (rad/s)');axes[row,col].set_xticks([-.2,0,.2]);axes[row,col].grid(alpha=.2)
axes[1,0].axis('off');axes[1,0].text(0,1,'Fixed locally trained 1499 actor\nOriginal finger limits and observations\n\n6 contact variants, 36 short branches\n18 closed-loop runs, 20 seconds each\nWalking scores: final 10 seconds\n\nBetter short-response fit did not\nproduce a consistent walking improvement.\nNo candidate promoted.\n\nOne deterministic flat setting;\nnot a robustness acceptance test.',va='top',fontsize=11);axes[1,2].legend(fontsize=8)
fig.suptitle('Single-parameter contact probes: short response versus closed-loop walking',fontsize=14);folder=a.report/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'contact_parameters.png',dpi=150);fig.savefig(folder/'contact_parameters.svg');plt.close(fig)
