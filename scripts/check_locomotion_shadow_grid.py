"""Verify six passive-monitor pairs and archive the preceding turning metrics.

Full rollouts go to an ignored local directory. Only summaries and the six
original evaluator CSVs go to the reviewable archive. Never overwrites runs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify_locomotion_shadow import verify_pair


POLICIES = ('baseline', 'reproduction_20261001')
YAW_COMMANDS = (-0.2, 0.0, 0.2)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--archive-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true',
                        help='Recheck saved pairs without rerunning physics')
    args = parser.parse_args()
    expected = {f'{p}_{yaw}.csv' for p in POLICIES for yaw in YAW_COMMANDS}
    found = {p.name for p in args.archive_source.glob('*.csv')}
    if found != expected:
        parser.error(f'Expected exactly the six turning CSVs; found {sorted(found)}')
    if args.summary_dir.exists():
        parser.error('Summary directory must not already exist')
    if args.verify_existing:
        if not args.output_dir.is_dir():
            parser.error('Existing rollout directory is required')
    elif args.output_dir.exists():
        parser.error('Rollout output directory must not already exist')
    source_rows = {}
    for name in sorted(expected):
        with (args.archive_source / name).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            parser.error(f'{name}: expected exactly one evaluator row')
        row = rows[0]
        if (row['terrain'] != 'plane' or float(row['friction']) != 0.8
                or float(row['command_x']) != 1.0 or float(row['command_y']) != 0.0
                or float(row['duration_s']) != 20.0):
            parser.error(f'{name}: unexpected comparison configuration')
        source_rows[name] = row
    for policy in POLICIES:
        if not (args.reference_root / f'mujoco/policies/{policy}/policy_actor.npz').is_file():
            parser.error(f'Missing exported policy: {policy}')
    if not args.verify_existing:
        args.output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    # Consistent CPU execution; this does not change observations or control.
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    recorder = Path(__file__).with_name('record_locomotion_sim2sim.py')
    manifest, pairs, comparison_rows, original_rows = [], [], [], []
    for policy in POLICIES:
        policy_path = args.reference_root / f'mujoco/policies/{policy}/policy_actor.npz'
        for yaw in YAW_COMMANDS:
            name = f'{policy}_{yaw}'
            source = args.archive_source / f'{name}.csv'
            print(f'Checking {name}: monitoring off/on', flush=True)
            pair_dir = args.output_dir / name
            if not args.verify_existing:
                pair_dir.mkdir()
            for mode in (() if args.verify_existing else ('off', 'on')):
                command = [sys.executable, str(recorder), '--reference-root', str(args.reference_root),
                           '--policy', str(policy_path), '--output-dir', str(pair_dir / mode),
                           '--label', policy, '--command-yaw', str(yaw), '--duration', '20']
                if mode == 'on':
                    command.append('--shadow')
                with (pair_dir / f'{mode}.log').open('w') as stream:
                    subprocess.run(command, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
            result = verify_pair(pair_dir / 'off', pair_dir / 'on')
            metrics = json.loads((pair_dir / 'off/metrics.json').read_text())
            archived = source_rows[source.name]
            # Compare every numeric metric, not just the headline values.
            numeric_keys = [k for k in archived if k not in ('backend', 'model', 'policy', 'metadata', 'terrain')]
            result['original_numeric_metrics_exactly_reproduced'] = all(
                float(archived[k]) == metrics[k] for k in numeric_keys)
            if float(archived['command_yaw']) != yaw:
                raise ValueError(f'{source.name}: command does not match filename')
            result['original_numeric_metric_deltas'] = {
                k: metrics[k] - float(archived[k]) for k in numeric_keys
                if metrics[k] != float(archived[k])}
            result['policy'] = policy
            result['metadata_sha256'] = metrics['metadata_sha256']
            result['source_csv'] = f'raw/{source.name}'
            result['source_csv_sha256'] = sha256(source)
            (pair_dir / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
            pairs.append(result)
            manifest.append(dict(file=result['source_csv'], sha256=result['source_csv_sha256'],
                                 bytes=source.stat().st_size))
            original_rows.append({**archived, 'model': Path(archived['model']).name,
                                  'metadata': Path(archived['metadata']).name, 'policy': policy})
            comparison_rows.append(dict(
                policy=policy, command_yaw_rps=yaw, elapsed_s=metrics['elapsed_s'], fell=metrics['fell'],
                mean_body_speed_x_mps=metrics['mean_body_speed_x_mps'],
                speed_tracking_rmse_mps=metrics['speed_tracking_rmse_mps'],
                yaw_rate_tracking_rmse_rps=metrics['yaw_rate_tracking_rmse_rps'],
                mean_signed_body_yaw_rate_rps=result['mean_signed_body_yaw_rate_rps'],
                distance_x_m=metrics['distance_x_m'], lateral_drift_m=metrics['lateral_drift_m'],
                mean_tilt_deg=metrics['mean_tilt_deg'], max_tilt_deg=metrics['max_tilt_deg'],
                candidate_enter_count=result['candidate_enter_count']))
            print(f"  {result['status']}; archived numeric metrics reproduced: "
                  f"{result['original_numeric_metrics_exactly_reproduced']}", flush=True)
    passed = all(p['status'] == 'PASS' for p in pairs)
    args.summary_dir.mkdir(parents=True)
    raw_dir = args.summary_dir / 'raw'
    raw_dir.mkdir()
    for name in sorted(expected):
        shutil.copyfile(args.archive_source / name, raw_dir / name)
        if sha256(raw_dir / name) != sha256(args.archive_source / name):
            raise RuntimeError(f'Archive checksum mismatch: {name}')
    report = dict(status='PASS' if passed else 'FAIL', verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  pair_count=len(pairs), pairs=pairs,
                  original_numeric_metrics_exactly_reproduced=all(
                      p['original_numeric_metrics_exactly_reproduced'] for p in pairs),
                  historical_replay_note='Historical numeric reproduction is a separate diagnostic; '
                      'it does not change the monitor off/on verdict. Original runs lack a full '
                      'runtime/model snapshot, so any historical differences remain unattributed.',
                  scope='Two exported policies, three yaw commands [-0.2,0,0.2], one seed (42), '
                        'plane/friction 0.8, 20 s per rollout, 1 ms physics, no rendering. '
                        'Non-interference only; no validated safety or recovery claim.')
    (args.summary_dir / 'verification_grid.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.summary_dir / 'manifest.json').write_text(json.dumps(dict(
        original_run_date='2026-10-01', files=manifest,
        provenance='Original CSVs preserved byte for byte. Policy/metadata hashes and configuration '
                   'come from fresh replay, not from an immutable snapshot of the original runs.',
        scripts_sha256={p.name: sha256(p) for p in (recorder, Path(__file__),
                            Path(__file__).with_name('verify_locomotion_shadow.py'),
                            Path(__file__).resolve().parents[1] /
                            'src/g1_control/monitoring/locomotion_shadow.py')}), indent=2) + '\n')
    for filename, rows in (('comparison.csv', original_rows),
                           ('replay_comparison.csv', comparison_rows)):
        with (args.summary_dir / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(f"Grid verification: {report['status']}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
