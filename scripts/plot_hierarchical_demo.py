"""Recorded task progress, message freshness and watchdog event timeline."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained');colors={'normal':'#2563eb','temporary_dropout':'#10b981','persistent_dropout':'#f97316'}
for scenario,color in colors.items():
 folder=a.run/(scenario+'_events');d=json.loads((folder/'decisions.json').read_text());events=json.loads((folder/'events.json').read_text());t=[r['time_s'] for r in d];axes[0].plot(t,[r['message_age_s'] for r in d],color=color,label=scenario);axes[1].step(t,[r['stage_index'] for r in d],where='post',color=color,label=scenario)
 for event in events:
  if event['event'] in ('watchdog_stale','watchdog_recovered','task_aborted'):
   axes[0].axvline(event['time_s'],color=color,alpha=.35,linestyle=':');axes[0].annotate(event['event'].replace('watchdog_','')+f" {event['time_s']:.2f}s",(event['time_s'],0.4 if event['event']=='watchdog_stale' else .85 if event['event']=='watchdog_recovered' else 1.9),rotation=30,fontsize=8,color=color)
axes[0].axhline(.3,color='black',linestyle='--',label='Message freshness limit 0.30s');axes[0].set_ylabel('Task-message age (simulation seconds)');axes[0].legend(fontsize=9);axes[0].set_ylim(-.03,2.6);axes[1].set_yticks([0,1,2],['walk 1','arc turn','walk 2']);axes[1].set_ylabel('Measured-goal task stage');axes[1].set_xlabel('Simulation time (s)')
for ax in axes:ax.grid(alpha=.2)
handoff=json.loads((a.run/'plan.json').read_text()).get('suite')=='handoff';caption='Dropout spans measured turn completion: hold turning command until fresh stage admission' if handoff else 'Temporary messages recover before next goal: normal and temporary physical trajectories coincide';fig.suptitle('Task-to-skill interface MVP: normal, temporary loss, persistent loss\n'+caption,fontsize=12);fig.savefig(a.output/'task_timeline.png',dpi=160);fig.savefig(a.output/'task_timeline.svg');plt.close(fig)
