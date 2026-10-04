"""Visualize saved physical-isolation measurements without stepping physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.traces/'complete.json').read_text())['results'];variants=['baseline','refresh_constants','central_com_sym','central_inertia_sym','contact_solref01','cone_elliptic','integrator_implicit','mirror_actor'];labels=['Baseline','setConst\ncontrol','Central COM\nsymmetric','Central inertia\nsymmetric','Contact\ntime 0.01 s','Elliptic\ncone','Implicit\nintegrator','Mirrored\ncontroller']
fig,(ax,bx)=plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios':[1,1]})
x=np.arange(len(variants))
for speed,offset,color in [(.5,-.18,'#36749b'),(1.,.18,'#bc6845')]:
 y=[next(r['heading_slope'] for r in rows if r['variant']==v and r['speed']==speed) for v in variants];ax.bar(x+offset,y,.34,color=color,label=f'vx={speed:g} m/s')
ax.axhline(0,color='gray',lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Measured heading rate (rad/s)');ax.set_title('Zero turning command: final 10 s heading slope');ax.legend();ax.grid(axis='y',alpha=.2)
for variant,label,color in [('baseline','Original controller','#bc6845'),('mirror_actor','Mirrored controller','#36749b')]:
 t=np.load(a.traces/f'{variant}_vx1_yaw+0.0.npz');heading=np.rad2deg(np.unwrap(t['signals'][:,24]));bx.plot(t['time'],heading,label=label,color=color)
bx.axhline(0,color='gray',lw=.8);bx.set_xlabel('Time (s)');bx.set_ylabel('Actual heading (deg)');bx.set_title('Same physical model, vx=1 m/s, wz command=0 | positive: left; negative: right');bx.legend();bx.grid(alpha=.2)
fig.suptitle('Fixed actor1499 weights | independent diagnostic interventions\nBias reverses with controller reflection; tested physical changes do not remove it',fontsize=12);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(a.output/'physics_isolation.png',dpi=160);fig.savefig(a.output/'physics_isolation.svg')
