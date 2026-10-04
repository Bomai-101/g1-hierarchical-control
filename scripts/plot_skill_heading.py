"""Recorded feedback comparison plot; no simulation."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();g=json.loads((a.results/'summary.json').read_text())['groups'];fig,axs=plt.subplots(2,2,figsize=(10,7))
for i,e in enumerate(('isaac','mujoco')):
 for j,s in enumerate(('hold','march')):
  ax=axs[i,j];rr=[r for r in g if r['engine']==e and r['skill']==s];x=np.arange(3);ax.bar(x,[np.rad2deg(r['final_abs_heading_error_median_rad']) for r in rr],label='Final heading error');ax.set_xticks(x,['K=0','K=0.5','K=1']);ax.set_ylabel('Median absolute error (deg)');ax.set_title(f'{e.upper()} / {s}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  for idx,r in enumerate(rr):ax.text(idx,np.rad2deg(r['final_abs_heading_error_median_rad']),f"alive {r['full']}/{r['total']}",ha='center',va='bottom',fontsize=8)
fig.suptitle('Fixed actor weights, bounded upper-layer yaw feedback\nLow-speed prefix; full survivors only; heading improvement is not task acceptance.');fig.tight_layout();fig.savefig(a.results/'heading_feedback.png',dpi=160)
