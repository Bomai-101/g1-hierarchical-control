import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads(a.summary.read_text());fig,ax=plt.subplots(1,2,figsize=(11,4))
for i,proto in enumerate(['normal','temporary','persistent']):
 rr=[r for r in s['strata'] if r['protocol']==proto];matrix=np.array([r['passed']/r['total'] for r in rr]).reshape(3,3);ax[0].bar(np.arange(3)+(i-1)*.24,matrix.mean(axis=0)*100,.24,label=proto)
ax[0].set_xticks(np.arange(3),['3.22','6.22','9.22']);ax[0].set_ylim(0,105);ax[0].set_xlabel('Planned stop/drop onset (s)');ax[0].set_ylabel('Pass % (3 seeds pooled descriptively)');ax[0].legend();ax[0].set_title('Direct handoff; no retuning')
labels=[];values=[]
for g in s['groups']:
 for key,description in [('request_to_stop_net_m','request-to-stop'),('hold_net_drift_m','hold drift')]:labels.append(g['protocol']+'\n'+description);values.append(g['metrics_all_available'][key]['median'] or 0)
ax[1].bar(np.arange(6),np.array(values)*100);ax[1].set_xticks(np.arange(6),labels,fontsize=8);ax[1].set_ylabel('Median root xy net displacement (cm)');ax[1].set_title('Available confirmed cases; see n in summary')
fig.suptitle('Isaac flat 0.5m/s | seeds101/202/303 | exploratory acceptance');fig.tight_layout();fig.savefig(a.output,dpi=160)
