"""Export diagnostic survival comparisons; no simulator stepping."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();g=json.loads((a.results/'summary.json').read_text())['groups'];fig,axs=plt.subplots(2,2,figsize=(11,7),sharey=True)
protocols=['direct','ramp_0.5','ramp_1.5']
for i,engine in enumerate(('isaac','mujoco')):
 for j,skill in enumerate(('hold','march')):
  ax=axs[i,j]
  for k,vx in enumerate((.5,1.)):
   z=[next(r for r in g if r['engine']==engine and r['skill']==skill and r['protocol']==p and r['vx']==vx) for p in protocols]
   positions=np.arange(3)+(k-.5)*.32;bars=ax.bar(positions,[100*r['full']/r['n'] for r in z],width=.3,label=f'{vx:g} m/s command')
   for bar,r in zip(bars,z):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{r['full']}/{r['n']}",ha='center',fontsize=8)
  ax.set_xticks(range(3),['Original handoff','Brake 0.5s','Brake 1.5s']);ax.set_title(f'{engine.upper()} / {skill}');ax.set_ylim(0,112);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  if j==0:ax.set_ylabel('Full 10s survival (%)')
axs[0,0].legend(loc='lower right',fontsize=8);fig.suptitle('Fixed skill weights: handoff affects survival\nSurvival alone does not establish stop or in-place march performance');fig.tight_layout();fig.savefig(a.results/'handoff_survival.png',dpi=160);plt.close(fig)
