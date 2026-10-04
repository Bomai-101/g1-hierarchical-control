"""Plot saved branch metrics only; no physics simulation."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader((a.report/'summary.csv').open()));fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for group,label,color in [('large_step','Free: large leg steps','#3b82f6'),('free_walk','Free: walking poses','#14b8a6'),('lifted','Gravity on, lifted walking poses','#a855f7'),('contact','Ground: walking poses','#f97316')]:
 for mode,style in [('dt005','--'),('dt001','-')]:
  rr=[r for r in rows if r['mode']==mode and ((group=='lifted' and r['condition']=='lifted') or (group=='contact' and r['condition']=='contact') or (group=='free_walk' and r['condition']=='free' and r['support']!='large_step') or (group=='large_step' and r['support']=='large_step'))];t=[20,50,200]
  for axis,key,mult,title in [(ax[0,0],'leg_q_max_rad',180/np.pi,'Maximum leg position mismatch (degrees)'),(ax[0,1],'heading_delta_rad',180/np.pi,'Maximum absolute heading mismatch (degrees)'),(ax[1,0],'com_vx_delta_mps',1,'Maximum absolute COM forward speed mismatch (m/s)')]:
   y=[max(abs(float(r[key])) for r in rr if abs(float(r['horizon_s'])*1000-v)<1e-6)*mult for v in t];axis.plot(t,y,style,color=color,marker='o',label=label+' / '+mode);axis.set_title(title);axis.set_xlabel('Held-target duration (ms)');axis.set_xticks(t);axis.grid(alpha=.2)
rr=[r for r in rows if r['condition']=='contact' and r['mode']=='dt001' and float(r['horizon_s'])==.2];labels=[r['case'] for r in rr];x=np.arange(len(rr));width=.18
for i,(engine,foot,color) in enumerate([('isaac','left','#2563eb'),('mu','left','#93c5fd'),('isaac','right','#d97706'),('mu','right','#fcd34d')]):
 ax[1,1].bar(x+(i-1.5)*width,[float(r[f'{foot}_fz_impulse_{engine}_Ns']) for r in rr],width,color=color,label=engine+' '+foot)
ax[1,1].set_xticks(x,labels,rotation=25,ha='right');ax[1,1].set_title('0–200 ms foot vertical force quadrature (N s)');ax[1,1].legend(fontsize=8);ax[0,0].legend(fontsize=7)
fig.suptitle('Fixed 1499 • identical initial states and held targets • reset branches\nPhysX averaged contact impulse vs Mu instantaneous-force quadrature; not full walking trajectories',fontsize=11)
folder=a.report/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'leg_contact.png',dpi=160);fig.savefig(folder/'leg_contact.svg');plt.close(fig)
