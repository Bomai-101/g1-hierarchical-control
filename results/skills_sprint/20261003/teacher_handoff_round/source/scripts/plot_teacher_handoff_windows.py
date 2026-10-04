"""Saved-state fixed-window comparisons, survivors only; no simulation."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader((a.results/'verification/cases.csv').open()));groups=[(e,s) for e in ('isaac','mujoco') for s in ('hold','march')];fig,axs=plt.subplots(1,2,figsize=(11,4.5));labels=[];initial=[];remaining=[];angles=[]
for e,s in groups:
 rr=[r for r in rows if r['engine']==e and r['skill']==s and r['full']=='True'];labels.append(f'{e.upper()}\n{s} (n={len(rr)})');initial.append(np.median([float(r['initial_net_m'])*100 for r in rr]));remaining.append(np.median([float(r['remaining_net_m'])*100 for r in rr]));angles.append([float(r['remaining_heading_rad'])*180/np.pi for r in rr])
x=np.arange(4)
for offset,vals,label in [(-.18,initial,'First 2s'),(.18,remaining,'Following 8s')]:
 bars=axs[0].bar(x+offset,vals,width=.35,label=label)
 for b,v in zip(bars,vals):axs[0].text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=9)
axs[0].set_ylabel('Median net displacement per window (cm)');axs[0].legend();axs[0].set_ylim(0,max(initial+remaining)*1.15);axs[1].boxplot(angles,tick_labels=labels,showfliers=True);axs[1].set_ylabel('Signed heading change during following 8s (deg)');axs[1].axhline(0,color='gray',linestyle='--',linewidth=1)
for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axs[0].set_xticks(x,labels);fig.suptitle('Teacher-state training: early transition vs remaining motion\nFixed 2s split is diagnostic; it does not certify skill entry. Full 10s survivors only.',fontsize=11);fig.tight_layout();fig.savefig(a.results/'window_comparison.png',dpi=170)
