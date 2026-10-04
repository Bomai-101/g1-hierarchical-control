"""Plot recorded heading errors without advancing physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for ax,case,title in zip(axes.flat,['hold0','correct_left','correct_right','left90','right90','wrap_left'],['Hold 0°','Initial +28.65° → 0°','Initial −28.65° → 0°','0° → +90°','0° → −90°','+179° → −179° (shortest +2°)']):
    ax.axhspan(-15,15,color='#e8f3e8');ax.axhspan(-5,5,color='#d6ead6');ax.axhline(0,color='gray',lw=.7)
    for profile,label,color in [('original','Original MuJoCo','#cf6a43'),('usd_position','USD-aligned MuJoCo','#236b9b')]:
        t=np.load(a.traces/f'{profile}_vx1_{case}.npz');ax.plot(t['time'],np.rad2deg(t['signals'][:,25]),label=label,color=color,lw=1.4)
    ax.set_title(title);ax.set_xlabel('Time (s)');ax.set_ylabel('Target − heading (deg)');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9);fig.suptitle('Same locally trained actor1499 | vx=1 m/s | heading command = clip(0.5 × error, −1, +1)\nPositive error: robot heading is to the right of target; shaded bands are exploratory ±5° / ±15°',fontsize=11)
fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(a.output/'heading_error.png',dpi=160);fig.savefig(a.output/'heading_error.svg');plt.close(fig)
rows=json.loads((a.traces/'complete.json').read_text())['results']
fig,ax=plt.subplots(figsize=(10,4));cases=['hold0','correct_left','correct_right','left90','right90','wrap_left','wrap_right'];x=np.arange(7)
for speed,offset,color in [(.5,-.18,'#236b9b'),(1.,.18,'#7cae44')]:
    r=[next(r for r in rows if r['profile']=='usd_position' and r['speed']==speed and r['case']==case) for case in cases]
    ax.bar(x+offset,[r['steady_signed_error_deg'] for r in r],.34,label=f'vx={speed:g} m/s',color=color)
ax.axhline(5,color='#999',ls='--',label='5° exploratory band');ax.set_xticks(x,cases,rotation=15);ax.set_ylabel('Mean target − heading (deg)');ax.set_title('USD-aligned model: residual heading error over final 10 s');ax.legend();fig.tight_layout();fig.savefig(a.output/'steady_error.png',dpi=160);plt.close(fig)
