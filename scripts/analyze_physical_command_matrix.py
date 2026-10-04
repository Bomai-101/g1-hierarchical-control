"""Analyze physical command rollouts; heading proxy is not a fault/safety label."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from g1_control.monitoring.yaw_window import heading_from_wxyz,bias_events
from g1_control.monitoring.command_analysis import scheduled_yaw_stats
from analyze_locomotion_yaw_windows import event_summary


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--trace-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.trace_dir.exists() or (args.results_dir/'analysis.json').exists():
        parser.error('Analysis output already exists')
    args.trace_dir.mkdir(parents=True)
    with (args.results_dir/'comparison.csv').open() as f:cases=list(csv.DictReader(f))
    rows,details=[],[]
    input_manifest={}
    for case in cases:
        p=args.source_dir/case['case']/'on'
        inputs={name:sha(p/name) for name in ('metrics.json','provenance.json','control_trace.npz',
                                                'trajectory.csv','shadow_signals.csv','shadow_events.json')}
        with (p/'trajectory.csv').open() as f:trace=list(csv.DictReader(f))
        t=np.array([float(r['time_s']) for r in trace])
        body=np.array([float(r['body_yaw_rate_rps']) for r in trace])
        command=np.array([float(r['command_yaw_rps']) for r in trace])
        with np.load(p/'control_trace.npz') as x:heading=heading_from_wxyz(x['sampled_states'][:,3:7])
        recorded=np.array([float(r['yaw_rad']) for r in trace])
        mismatch=float(np.max(np.abs(np.arctan2(np.sin(heading-recorded),np.cos(heading-recorded)))))
        if mismatch>1e-9:raise ValueError('Heading cross-check failed')
        legacy_events=json.loads((p/'shadow_events.json').read_text())['events']
        epochs=sorted({int(r['command_epoch']) for r in trace if r['policy_update']=='True'})
        # Independent of angular-error detector thresholds. This is an operational
        # route deviation PROXY, not an independently observed physical-fault label.
        base=scheduled_yaw_stats(t,body,heading,command,1.)
        route_error=heading-base['reference_heading']
        if t[-1]>=2.:
            route_error-=float(np.interp(2.,t,route_error))
        labels=bias_events(t,route_error,.35,1.,2.)
        enters=[e for e in labels if e['event']=='candidate_enter']
        proxy_first=enters[0]['time_s'] if enters else None
        detail=dict(case=case['case'],source_sha256=inputs,
            heading_proxy_definition='Euler route error relative to held-command integral, reanchored at 2 s; '
                                     '|error|>0.35 rad with same-sign 1 s dwell. Operational proxy only.',
            heading_proxy_confirmed_time_s=proxy_first,
            heading_proxy_threshold_onset_s=enters[0]['onset_time_s'] if enters else None,
            quaternion_yaw_crosscheck_max_error_rad=mismatch,
            applied_command_epochs=epochs,window_results={})
        output=dict(case=case['case'],policy=case['policy'],terrain=case['terrain'],seed=int(case['seed']),
            forward_mps=float(case['forward_mps']),command_mode=case['command_mode'],
            steady_yaw_rps=float(case['steady_yaw_rps']),elapsed_s=float(case['elapsed_s']),fell=int(case['fell']),
            applied_command_epochs=json.dumps(epochs),
            step_transitions_applied=max(0,len(epochs)-1) if case['command_mode']=='steps' else 0,
            heading_proxy_confirmed_time_s=proxy_first,
            heading_error_end_rad=float(route_error[-1]) if t[-1]>=2. else None,
            heading_proxy_assessable=bool(t[-1]>=3.))
        for reason in ('tilt','angular_speed','forward_error','yaw_error'):
            entries=[e for e in legacy_events if e['reason']==reason and e['event']=='candidate_enter']
            output[f'legacy_{reason}_entries']=len(entries)
            output[f'legacy_{reason}_first_s']=entries[0]['time_s'] if entries else None
        detail['legacy_events']=legacy_events
        window_trace=[dict(time_s=float(tt),body_angular_z_rps=float(bb),command_yaw_rps=float(cc),
                           heading_rad=float(hh),heading_reference_rad=float(rr))
                      for tt,bb,cc,hh,rr in zip(t,body,command,heading,base['reference_heading'])]
        for window in (1.,2.,4.):
            stats=scheduled_yaw_stats(t,body,heading,command,window)
            eligible=2.+window
            events=bias_events(t,stats['body_mean'],.3,.2,eligible)
            hevents=bias_events(t,stats['heading_mean'],.3,.2,eligible)
            ev=event_summary(t,events,eligible)
            first=ev['first_candidate_time_s']
            output[f'w{int(window)}_eligible']=bool(t[-1]>=eligible-1e-10)
            output[f'w{int(window)}_enter_count']=ev['candidate_enter_count']
            output[f'w{int(window)}_first_s']=first
            output[f'w{int(window)}_candidate_before_fall']=bool(output['fell'] and first is not None and first<t[-1])
            detail['window_results'][str(window)]=dict(body_events=events,heading_events=hevents,
                **ev,eligible_start_s=eligible,
                candidate_minus_heading_proxy_time_s=None if first is None or proxy_first is None else first-proxy_first)
            for i,row in enumerate(window_trace):
                for key in ('body_mean','body_rmse','heading_mean','heading_rmse'):
                    value=stats[key][i]
                    row[f'{key}_w{int(window)}']=None if np.isnan(value) else float(value)
        write_csv(args.trace_dir/f"{case['case']}.csv",window_trace)
        for name,digest in inputs.items():
            if sha(p/name)!=digest:raise ValueError('Input modified during analysis')
        input_manifest[case['case']]=inputs
        rows.append(output);details.append(detail)
    write_csv(args.results_dir/'diagnostics.csv',rows)
    groups=[]
    for terrain in ('plane','rough'):
        for policy in ('baseline','reproduction_20261001'):
            selected=[r for r in rows if r['terrain']==terrain and r['policy']==policy]
            groups.append(dict(terrain=terrain,policy=policy,case_count=len(selected),
                fell_cases=sum(r['fell'] for r in selected),completed_30s_cases=sum(r['elapsed_s']==30 for r in selected),
                elapsed_min_s=min(r['elapsed_s'] for r in selected),elapsed_max_s=max(r['elapsed_s'] for r in selected),
                w1_eligible_cases=sum(r['w1_eligible'] for r in selected),
                w1_candidate_cases=sum(r['w1_enter_count']>0 for r in selected),
                heading_proxy_confirmed_cases=sum(r['heading_proxy_confirmed_time_s'] is not None for r in selected)))
    write_csv(args.results_dir/'group_summary.csv',groups)
    report=dict(mode='offline analysis of real MuJoCo rollouts',case_count=len(rows),
        windows_s=[1.,2.,4.],threshold_rps=.3,dwell_s=.2,warmup_s=2.,
        policy_command_definition='Left-held commands on each policy interval; new steps never affect prior intervals.',
        inputs_unchanged=True,proxy_limits=dict(error_rad=.35,dwell_s=1.,reanchor_s=2.),
        real_fault_ground_truth_available=False,real_false_positive_rate=None,
        switch_authorized=False,groups=groups,cases=details)
    (args.results_dir/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    provenance=dict(inputs_sha256=input_manifest,
        analysis_sources_sha256={str(p.relative_to(REPO)):sha(p) for p in (Path(__file__).resolve(),
            REPO/'src/g1_control/monitoring/command_analysis.py',REPO/'src/g1_control/monitoring/yaw_window.py')})
    (args.results_dir/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
