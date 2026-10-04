"""Plot actual rollout durations against diagnostic eligibility."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    args=parser.parse_args()
    with (args.results_dir/'comparison.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True,layout='constrained')
    conditions=(('plane',42),('rough',42),('rough',43),('rough',44))
    plans=(('steps',0.),('slow_sine',0.),('steady',0.),('steady',-.1),('steady',.1))
    colors=('#1768ac','#c56516','#238d70','#7851a9','#ba3b46')
    labels=('Steps','Slow sine','Steady 0','Steady -0.1','Steady +0.1')
    for i,policy in enumerate(('baseline','reproduction_20261001')):
        for j,speed in enumerate((.5,1.)):
            ax=axes[i,j]
            for k,(plan,yaw) in enumerate(plans):
                selected=[r for r in rows if r['policy']==policy and float(r['forward_mps'])==speed
                          and r['command_mode']==plan and float(r['steady_yaw_rps'])==yaw]
                x=[conditions.index((r['terrain'],int(r['seed'])))+(k-2)*.065 for r in selected]
                ax.scatter(x,[float(r['elapsed_s']) for r in selected],s=35,color=colors[k],
                           marker='o',label=labels[k])
            ax.axhline(3,color='#444',ls='--',lw=.8,label='1 s window first eligible at 3 s')
            ax.axvspan(-.35,.35,color='#1768ac',alpha=.04)
            ax.set_ylim(0,32);ax.set_xlim(-.4,3.4);ax.grid(axis='y',alpha=.2)
            actor='Supplied actor' if i==0 else 'Locally trained actor'
            ax.set_title(f'{actor} | forward command {speed} m/s')
            ax.set_xticks(range(4),['Plane 42','Rough 42','Rough 43','Rough 44'])
            if j==0:ax.set_ylabel('Continuous rollout elapsed time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('80 physical command cases: termination versus diagnostic availability\n'
                 'Each dot is one off/on pair; reference base-height termination, no resets or recovery')
    target=args.results_dir/'figures';target.mkdir(exist_ok=False)
    fig.savefig(target/'rollout_duration.svg');fig.savefig(target/'rollout_duration.png',dpi=150)
    plt.close(fig)
    (args.results_dir/'plot_runtime.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__),indent=2)+'\n')


if __name__=='__main__':main()
