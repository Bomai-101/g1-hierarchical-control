import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();g=json.loads(a.summary.read_text())['groups']
fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(len(g));labels=[r['protocol'] for r in g]
for i,key in enumerate(['resumed','completed','passed']):ax[0].bar(x+(i-1)*.24,[r[key] for r in g],.24,label=key)
ax[0].set_xticks(x,labels);ax[0].set_ylim(0,52);ax[0].set_ylabel('Cases (48 per protocol)');ax[0].legend();ax[0].set_title('Isaac low-speed walk / hold / walk')
ax[1].bar(x,[r['hold_latency_median_s'] or 0 for r in g]);ax[1].set_xticks(x,labels);ax[1].set_ylabel('Seconds after hold entry');ax[1].set_title('Median readiness latency, resumed cases only')
fig.suptitle('Frozen actors, seed42; exploratory sequence criteria');fig.tight_layout();fig.savefig(a.output,dpi=160)
