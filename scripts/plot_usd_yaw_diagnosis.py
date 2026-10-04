"""Plot recorded yaw-response and command-offset diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.run_root/'usd_yaw_diagnosis/complete.json').read_text())['results'];fig,axes=plt.subplots(2,2,figsize=(12,8))
ax=axes[0,0]
for speed,color in [(.5,'#36749b'),(1.,'#bc6845')]:
 r=[r for r in rows if r['variant']=='baseline' and r['speed']==speed];ax.plot([r['yaw_command'] for r in r],[r['heading_slope'] for r in r],'o-',color=color,label=f'vx={speed:g} m/s')
ax.plot([-.2,.2],[-.2,.2],'--',color='gray',label='Ideal tracking');ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set_xlabel('Command wz (rad/s)');ax.set_ylabel('Measured heading rate (rad/s)');ax.set_title('Fixed actor1499: rate response (last 10 s)');ax.legend(fontsize=9)
for ax,case,title in zip([axes[0,1],axes[1,0],axes[1,1]],['hold0','left90','right90'],['Hold 0°','Turn to +90°','Turn to −90°']):
 ax.axhspan(-5,5,color='#e6f2e6');ax.axhline(0,color='gray',lw=.7)
 for folder,label,color in [('usd_heading_replays','Before command offset','#bc6845'),('usd_yaw_compensation','Experimental command offset','#36749b')]:
  t=np.load(a.run_root/folder/f'usd_position_vx1_{case}.npz');ax.plot(t['time'],np.rad2deg(t['signals'][:,25]),label=label,color=color,lw=1.4)
 ax.set_title(title+' | vx=1 m/s');ax.set_xlabel('Time (s)');ax.set_ylabel('Target − heading (deg)');ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Same weights and physical model | command offset calibrated from separate rate-command trials\nDiagnostic only: the offset cancels bias but does not identify its physical cause',fontsize=11);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(a.output/'yaw_diagnosis.png',dpi=160);fig.savefig(a.output/'yaw_diagnosis.svg')
