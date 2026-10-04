"""Development/held-out signal calibration. No G1 dynamics or actions executed."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from g1_control.monitoring.yaw_calibration import FAMILIES, make_case, evaluate_case

DEV_SEEDS=tuple(range(1000,1012))
HELDOUT_SEEDS=tuple(range(10000,10012))
CONFIGS=tuple(dict(window_s=w,threshold_rps=h,dwell_s=d,warmup_s=2.)
              for w in (1.,2.,4.) for h in (.25,.3,.4) for d in (.2,.5,1.))
LIMITS=dict(max_negative_case_alarm_fraction=.05,max_negative_alarm_time_fraction=.01,
            min_positive_detection_fraction=.90,max_positive_delay_p95_s=3.)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate(rows):
    negatives=[r for r in rows if r['negative_case']]
    positives=[r for r in rows if not r['negative_case']]
    delays=[r['detection_delay_s'] for r in positives if r['detected']]
    exposure=sum(r['false_exposure_s'] for r in negatives)
    return dict(case_count=len(rows),negative_count=len(negatives),positive_count=len(positives),
        negative_case_alarm_fraction=sum(r['false_candidate_count']>0 for r in negatives)/len(negatives),
        negative_alarm_time_fraction=sum(r['false_active_duration_s'] for r in negatives)/exposure,
        positive_detection_fraction=sum(r['detected'] for r in positives)/len(positives),
        positive_delay_median_s=float(np.median(delays)) if delays else None,
        positive_delay_p95_s=float(np.quantile(delays,.95)) if delays else None,
        positive_preonset_alarm_fraction=sum(r['false_candidate_count']>0 for r in positives)/len(positives),
        positive_preexisting_alarm_fraction=sum(r['preexisting_alarm_at_onset'] for r in positives)/len(positives),
        positive_wrong_direction_entries=sum(r['wrong_direction_entries'] for r in positives))


def qualifies(row):
    return (row['negative_case_alarm_fraction']<=LIMITS['max_negative_case_alarm_fraction']
        and row['negative_alarm_time_fraction']<=LIMITS['max_negative_alarm_time_fraction']
        and row['positive_detection_fraction']>=LIMITS['min_positive_detection_fraction']
        and row['positive_delay_p95_s'] is not None
        and row['positive_delay_p95_s']<=LIMITS['max_positive_delay_p95_s'])


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--case-output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() or args.case_output_dir.exists():
        parser.error('Use new output directories')
    args.output_dir.mkdir(parents=True);args.case_output_dir.mkdir(parents=True)
    case_metadata=[];partitions={}
    for split,seeds in (('development',DEV_SEEDS),('heldout',HELDOUT_SEEDS)):
        cases=[]
        for seed in seeds:
            for family in FAMILIES:
                case=make_case(seed,family)
                filename=f'{split}_{seed}_{family}.npz'
                p=args.case_output_dir/filename
                np.savez_compressed(p,times=case['times'],command=case['command'],
                                    measured_body_z=case['measured_body_z'])
                case_metadata.append(dict(split=split,seed=seed,family=family,
                    onset_s=case['onset_s'],direction=case['direction'],parameters=case['parameters'],
                    file=filename,sha256=sha(p)))
                cases.append(case)
        partitions[split]=cases
    dev_summaries=[];heldout_summaries=[];detail=[]
    for split in ('development','heldout'):
        print(f'Evaluating {split}: {len(partitions[split])} cases x {len(CONFIGS)} settings',flush=True)
        for idx,config in enumerate(CONFIGS):
            rows=[]
            for case in partitions[split]:
                scored=evaluate_case(case,**config)
                row=dict(split=split,configuration=idx,seed=case['seed'],family=case['family'],
                         **config,**scored)
                rows.append(row);detail.append(row)
            summary=dict(configuration=idx,**config,**aggregate(rows))
            summary['meets_signal_level_limits']=qualifies(summary)
            (dev_summaries if split=='development' else heldout_summaries).append(summary)
        if split=='development':
            acceptable=[r for r in dev_summaries if r['meets_signal_level_limits']]
            selected=min(acceptable,key=lambda r:(r['positive_delay_p95_s'],r['negative_alarm_time_fraction'],
                                                   r['window_s'],r['configuration'])) if acceptable else None
            # Freeze selection before examining held-out performance.
            print('Development selection:',None if selected is None else selected['configuration'],flush=True)
    selection_holdout=next((r for r in heldout_summaries if selected and
                           r['configuration']==selected['configuration']),None)
    # Baselines were specified in advance, independent of held-out scores.
    baseline_indices=[i for i,c in enumerate(CONFIGS) if c['threshold_rps']==.3 and c['dwell_s']==.2]
    by_family=[]
    for split in ('development','heldout'):
        for index in baseline_indices:
            for family in FAMILIES:
                rows=[r for r in detail if r['split']==split and r['configuration']==index and r['family']==family]
                by_family.append(dict(split=split,configuration=index,family=family,
                    case_count=len(rows),false_alarm_cases=sum(r['false_candidate_count']>0 for r in rows),
                    detected_cases=sum(r['detected'] for r in rows),
                    mean_false_active_duration_s=float(np.mean([r['false_active_duration_s'] for r in rows]))))
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),mode='synthetic_signal_calibration_only',
        simulator_steps_executed=0,physical_normal_cases_collected=0,
        duration_s=40.,dt_s=.02,development_seeds=DEV_SEEDS,heldout_seeds=HELDOUT_SEEDS,
        case_count=len(case_metadata),configurations=CONFIGS,selection_limits=LIMITS,
        selection_rule='Development limits first; minimize detection delay p95, then false active time. '
                       'Held-out results do not alter selection or thresholds.',
        selected_development=selected,selected_heldout=selection_holdout,
        selected_passes_heldout=bool(selection_holdout and selection_holdout['meets_signal_level_limits']),
        real_world_switch_authorized=False,
        scope='No injected DC labels apply to generated signal families, not robot safety. '
              'Finite slow oscillations can have legitimate local directional means. '
              'This is not multi-seed G1/terrain validation; no actual supervisor or fallback was executed.')
    write_csv(args.output_dir/'development.csv',dev_summaries)
    write_csv(args.output_dir/'heldout.csv',heldout_summaries)
    write_csv(args.output_dir/'baseline_by_family.csv',by_family)
    write_csv(args.case_output_dir/'all_case_scores.csv',detail)
    (args.case_output_dir/'case_manifest.json').write_text(json.dumps(case_metadata,indent=2)+'\n')
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    provenance=dict(runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        case_manifest_sha256=sha(args.case_output_dir/'case_manifest.json'),
        detailed_scores_sha256=sha(args.case_output_dir/'all_case_scores.csv'),
        source_sha256={str(p.relative_to(REPO)):sha(p) for p in (Path(__file__).resolve(),
            REPO/'src/g1_control/monitoring/yaw_calibration.py',REPO/'src/g1_control/monitoring/yaw_window.py')})
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Calibration complete; held-out qualified selection:',report['selected_passes_heldout'],flush=True)


if __name__=='__main__':
    main()
