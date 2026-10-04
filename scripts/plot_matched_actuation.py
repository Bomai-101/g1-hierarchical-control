"""Plot actual matched responses and single-joint yaw interventions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.run_root
rt=json.loads((root/'matched_actuation_isaac/runtime.json').read_text());isaac=np.load(root/'matched_actuation_isaac/responses.npz');case=next(i for i,c in enumerate(rt['cases']) if c['name']=='walking_state_100');j=rt['joint_names'].index('right_four_joint');fig,axes=plt.subplots(1,3,figsize=(15,4.5));ax=axes[0]
ax.plot(isaac['time'],isaac['joint_pos'][:,case,j],color='#444',label='Actual Isaac (5 ms)')
for folder,label,color in [('matched_actuation_mujoco','MuJoCo baseline (1 ms)','#bc6845'),('matched_actuation_limits','MuJoCo tighter limits (1 ms)','#36749b')]:
 t=np.load(root/folder/'substep_dt001.npz');ax.plot(t['time'],t['joint_pos'][:,case,j],label=label,color=color)
ax.axhline(0,color='gray',ls='--');ax.set_title('Same initial state and held target\nright_four_joint target = −1.535 rad');ax.set_xlabel('Time (s)');ax.set_ylabel('Joint position (rad)');ax.legend(fontsize=8);ax.grid(alpha=.2)
rows=json.loads((root/'finger_limit_localization/complete.json').read_text())['results'];variants=['baseline','left_four_joint','right_four_joint','left_six_joint','right_six_joint','active4','finger_limit002'];labels=['Baseline','Left four','Right four','Left six','Right six','Active 4','All 14'];x=np.arange(7)
for ax,metric,title,ylabel in [(axes[1],'heading_slope','Zero-turning command: residual yaw bias','Heading rate (rad/s)'),(axes[2],'forward_rmse','Tradeoff: forward speed tracking','Forward RMSE (m/s)')]:
 for speed,offset,color in [(.5,-.18,'#36749b'),(1.,.18,'#bc6845')]:
  y=[next(r[metric] for r in rows if r['variant']==v and r['speed']==speed) for v in variants];ax.bar(x+offset,y,.34,label=f'vx={speed:g}',color=color)
 ax.axhline(0,color='gray',lw=.7);ax.set_xticks(x,labels,rotation=45,ha='right');ax.set_title(title);ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('Fixed actor1499 | physical finger-limit interventions only | diagnostic, not deployment');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(a.output/'matched_actuation.png',dpi=160);fig.savefig(a.output/'matched_actuation.svg')
