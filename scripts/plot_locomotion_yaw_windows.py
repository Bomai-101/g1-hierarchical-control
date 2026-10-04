"""Plot saved offline yaw diagnostics using Matplotlib; no simulation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.results_dir/'summary.json').read_text())
    target = args.results_dir/'figures'
    target.mkdir(exist_ok=False)
    config = report['primary_config']
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True,
                             layout='constrained')
    for run in report['runs']:
        column = 0 if run['policy']=='baseline' else 1
        row = (-.2, 0., .2).index(run['command_yaw_rps'])
        ax = axes[row, column]
        name = f"{run['policy']}_{run['command_yaw_rps']}"
        with (args.trace_dir/f'{name}_windows.csv').open() as stream:
            records = list(csv.DictReader(stream))
        def data(key):
            return np.array([float(r[key]) if r[key] else np.nan for r in records])
        t = data('time_s')
        ax.plot(t, data('instantaneous_body_error_rps'), color='#b8bec8', lw=.6,
                label='Instantaneous body-z error')
        ax.plot(t, data('body_mean_error_rps'), color='#1768ac', lw=1.5,
                label='1 s mean body-z error')
        ax.plot(t, data('heading_mean_error_rps'), color='#c56516', lw=1.3,
                label='1 s mean Euler-heading error')
        ax.axhline(.3, color='#ba3b46', lw=.8, ls='--', label='Exploratory +/-0.3 threshold')
        ax.axhline(-.3, color='#ba3b46', lw=.8, ls='--')
        ax.axhline(0, color='#444', lw=.5)
        ax.axvspan(0, config['warmup_s'], color='#777', alpha=.12,
                   label='Startup excluded')
        ax.axvline(config['first_eligible_full_poststartup_window_s'], color='#555', ls=':', lw=.9)
        first = run['primary_body_window']['first_candidate_time_s']
        if first is not None:
            i = int(np.argmin(np.abs(t-first)))
            ax.scatter([t[i]], [data('body_mean_error_rps')[i]], color='#b52232',
                       s=30, zorder=5, label='First body-window candidate')
        label = 'Supplied actor' if column==0 else 'Locally trained actor'
        ax.set_title(f"{label} | yaw command {run['command_yaw_rps']:+.1f} rad/s")
        ax.set_ylim(-1.3, 1.65)
        ax.set_xlim(0, 20)
        ax.grid(alpha=.18)
        if column==0:
            ax.set_ylabel('Signed yaw tracking error (rad/s)')
        if row==2:
            ax.set_xlabel('Recorded simulation time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Passive yaw-window diagnostics on six saved rollouts\n'
                 '1 s window | 2 s startup exclusion | 0.2 s same-sign dwell | no simulation rerun',
                 fontsize=14)
    fig.savefig(target/'yaw_windows.svg')
    fig.savefig(target/'yaw_windows.png', dpi=150)
    plt.close(fig)
    print(target/'yaw_windows.svg')


if __name__ == '__main__':
    main()
