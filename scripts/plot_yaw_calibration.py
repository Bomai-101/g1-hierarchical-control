"""Show the predeclared baseline calibration tradeoff, without simulation."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    args=parser.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for split,style,color in (('development','o','#1768ac'),('heldout','s','#c56516')):
        with (args.results_dir/f'{split}.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if r['threshold_rps']=='0.3' and r['dwell_s']=='0.2']
        windows=[float(r['window_s']) for r in rows]
        axes[0].plot(windows,[100*float(r['negative_case_alarm_fraction']) for r in rows],
                     marker=style,color=color,label=split.capitalize())
        axes[1].plot(windows,[float(r['positive_delay_p95_s']) for r in rows],
                     marker=style,color=color,label=split.capitalize())
    axes[0].axhline(5,color='#b52232',ls='--',lw=1,label='Exploratory 5% limit')
    axes[1].axhline(3,color='#b52232',ls='--',lw=1,label='Exploratory 3 s limit')
    axes[0].set_ylabel('No-DC cases with a candidate (%)')
    axes[1].set_ylabel('Detection delay P95 (s), detected positives only')
    for ax in axes:
        ax.set_xlabel('Trailing window (s)');ax.set_xticks([1,2,4])
        ax.grid(alpha=.2);ax.legend(fontsize=9);ax.set_ylim(bottom=0)
    fig.suptitle('Signal-level calibration: false candidates versus delay\n'
                 '0.3 rad/s threshold | 0.2 s dwell | 72 no-DC + 24 injected-DC cases per split')
    target=args.results_dir/'figures';target.mkdir(exist_ok=False)
    fig.savefig(target/'calibration_tradeoff.svg');fig.savefig(target/'calibration_tradeoff.png',dpi=150)
    plt.close(fig)
    (args.results_dir/'plot_runtime.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__),indent=2)+'\n')


if __name__=='__main__':
    main()
