"""Plot fixed 2x2 factorial metrics without replaying physics."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader((a.report/'walking_metrics.csv').open()));fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
variants=['baseline','finger_only','contact_only','combined'];colors=['#2563eb','#a855f7','#f97316','#10b981']
for col,speed in enumerate((.5,1.)):
 for v,c in zip(variants,colors):
  rr=sorted([r for r in rows if r['variant']==v and float(r['speed'])==speed],key=lambda r:float(r['yaw_command']));x=[float(r['yaw_command']) for r in rr];axes[0,col].plot(x,[float(r['forward_rmse']) for r in rr],'-o',color=c,label=v);axes[1,col].plot(x,[float(r['mean_world_wz']) for r in rr],'-o',color=c,label=v)
 axes[0,col].set_title(f'Forward tracking RMSE (m/s), vx={speed:g}');axes[1,col].set_title(f'Mean world angular z (rad/s), vx={speed:g}');axes[1,col].plot([-.2,.2],[-.2,.2],'k--',label='command')
 for row in (0,1):axes[row,col].set_xlabel('Yaw-rate command (rad/s)');axes[row,col].set_xticks([-.2,0,.2]);axes[row,col].grid(alpha=.2)
axes[0,0].legend(fontsize=9);axes[1,1].legend(fontsize=9);fig.suptitle('Fixed 1499: finger-limit time .002 x contact time .005\n24 deterministic 20-second runs; scores from final 10 seconds; no candidate promoted',fontsize=12);folder=a.report/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'finger_contact.png',dpi=160);fig.savefig(folder/'finger_contact.svg');plt.close(fig)
