"""Plot saved conditional physics/observation interventions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--traces',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.traces/'complete.json').read_text())['results'];variants=['soft_self','soft_paired_tight','tight_self','tight_paired_soft'];labels=['Soft physics\nsees soft','Soft physics\nsees tight','Tight physics\nsees tight','Tight physics\nsees soft'];x=np.arange(4);fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for ax,metric,title,ylabel in [(axes[0],'heading_slope','Zero turning command: residual yaw','Heading rate (rad/s)'),(axes[1],'forward_rmse','Forward speed tracking','Forward RMSE (m/s)')]:
 for speed,offset,color in [(.5,-.18,'#36749b'),(1.,.18,'#bc6845')]:
  y=[next(r[metric] for r in rows if r['variant']==v and r['speed']==speed) for v in variants];ax.bar(x+offset,y,.34,label=f'vx={speed:g} m/s',color=color)
 ax.axhline(0,color='gray',lw=.7);ax.set_xticks(x,labels);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
with (a.analysis/'policy_sensitivity.csv').open() as f:sensitivity=list(csv.DictReader(f))
ax=axes[2]
for speed,offset,color in [(.5,-.18,'#36749b'),(1.,.18,'#bc6845')]:
 vals=[float(next(r['leg_action_change_rms'] for r in sensitivity if r['case']==f'{v}_vx{speed:g}'))*.5 for v in variants];ax.bar(x+offset,vals,.34,color=color,label=f'vx={speed:g} m/s')
ax.set_xticks(x,labels);ax.set_ylabel('Leg target change RMS (rad)');ax.set_title('Same state, replace only 8 input fields');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Fixed actor1499 | four finger position/velocity fields only\nConditional paired model synchronizes all other states each step; diagnostic only');fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(a.output/'feedback_paths.png',dpi=160);fig.savefig(a.output/'feedback_paths.svg')
