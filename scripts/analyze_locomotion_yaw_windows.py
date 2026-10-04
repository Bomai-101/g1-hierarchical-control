"""Analyze saved six-pair yaw signals without simulation or recorder changes."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from g1_control.monitoring.locomotion_shadow import LocomotionShadow, ShadowSample, ShadowThresholds
from g1_control.monitoring.yaw_window import (bias_events, exceedance_runs, heading_from_wxyz,
                                            trailing_error_stats, trailing_heading_stats)
from verify_locomotion_shadow import verify_pair

POLICIES = ('baseline', 'reproduction_20261001')
YAWS = (-.2, 0., .2)
WINDOWS = (.5, 1., 2.)
THRESHOLDS = (.2, .3, .4)
WARMUP, DWELL = 2., .2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def event_summary(times, events, eligible):
    entries = [e for e in events if e['event'] == 'candidate_enter']
    first = entries[0]['time_s'] if entries else None
    active, onset, duration = False, None, 0.
    for event in events:
        if event['event'] == 'candidate_enter':
            active, onset = True, event['time_s']
        elif active:
            duration += event['time_s']-onset
            active = False
    if active:
        duration += float(times[-1])-onset
    return dict(candidate_enter_count=len(entries), first_candidate_time_s=first,
                delay_from_eligibility_s=None if first is None else first-eligible,
                active_duration_s=duration, active_at_end=active)


def phase_stats(times, errors, heading, command, start, end):
    inside = (times > start) & (times < end)
    t = np.r_[start, times[inside], end]
    e, h = np.interp(t, times, errors), np.interp(t, times, heading)
    mean, rmse = trailing_error_stats(t, e, end-start)
    hm, hr = trailing_heading_stats(t, h, command, end-start)
    return dict(start_s=start, end_s=end, body_mean_error_rps=float(mean[-1]),
                body_error_rmse_rps=float(rmse[-1]), heading_mean_error_rps=float(hm[-1]),
                heading_error_rmse_rps=float(hr[-1]))


def controls():
    t = np.arange(0., 20.001, .02)
    oscillation = .8*np.sin(8*np.pi*t)
    signals = dict(zero_error=np.zeros_like(t), zero_mean_4hz_gait=oscillation,
                   zero_mean_2p7hz=np.sin(5.4*np.pi*t)*.8,
                   zero_mean_slow_0p5hz=np.sin(np.pi*t)*.8,
                   startup_only=np.where(t < WARMUP, .8, 0.),
                   persistent_negative_bias=-.4+oscillation,
                   bias_after_8s=np.where(t < 8., oscillation, -.4+oscillation),
                   temporary_poststartup_bias=np.where((t >= 6.) & (t < 6.3), .8, 0.))
    rows = []
    for name, values in signals.items():
        for window in WINDOWS:
            mean, rmse = trailing_error_stats(t, values, window)
            for threshold in THRESHOLDS:
                eligible = WARMUP+window
                events = bias_events(t, mean, threshold, DWELL, eligible)
                result = event_summary(t, events, eligible)
                rows.append(dict(signal=name, window_s=window, threshold_rps=threshold,
                                 **result,
                                 rmse_threshold_sample_fraction=float(np.mean(rmse[t >= eligible] > threshold)),
                                 delay_from_known_bias_onset_s=(None if result['first_candidate_time_s'] is None
                                     else result['first_candidate_time_s']-(8. if name=='bias_after_8s' else 0.))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--trace-output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.trace_output_dir.exists():
        parser.error('Outputs must be new directories')
    model = args.reference_root / 'mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml'
    xml = ET.parse(model).getroot()
    pelvis = xml.find('worldbody/body')
    joint = pelvis.find('joint')
    if (pelvis.get('name') != 'pelvis' or joint is None or joint.get('type') != 'free'
            or pelvis.get('quat', '1 0 0 0') != '1 0 0 0'):
        parser.error('Expected first body pelvis with a free joint and identity fixed orientation')
    # These are CURRENT hashes, not retroactively captured historical model hashes.
    meshdir = xml.find('compiler').get('meshdir', '')
    model_files = [model] + [model.parent / meshdir / m.get('file')
                            for m in xml.findall('asset/mesh') if m.get('file')]
    metadata = args.reference_root / 'mujoco/policies/isaac_metadata.json'
    manifest = dict(analysis_runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                         platform=platform.platform()),
                    reference_head_current=subprocess.check_output(
                        ['git', '-C', str(args.reference_root), 'rev-parse', 'HEAD'], text=True).strip(),
                    model_snapshot_timing='Current model/XML/mesh hashes; original rollouts did not record model hashes.',
                    current_model_files={str(p.relative_to(args.reference_root)): sha(p) for p in model_files},
                    metadata_sha256_current=sha(metadata), inputs={}, policies={},
                    analysis_files_sha256={str(p.relative_to(REPO)): sha(p) for p in (
                        Path(__file__).resolve(), REPO/'src/g1_control/monitoring/yaw_window.py',
                        REPO/'src/g1_control/monitoring/locomotion_shadow.py',
                        REPO/'scripts/verify_locomotion_shadow.py')})
    args.output_dir.mkdir(parents=True)
    args.trace_output_dir.mkdir(parents=True)
    runs, sweep, metrics_rows = [], [], []
    for policy in POLICIES:
        actor = args.reference_root/f'mujoco/policies/{policy}/policy_actor.npz'
        manifest['policies'][policy] = sha(actor)
        for yaw in YAWS:
            name = f'{policy}_{yaw}'
            pair = args.source_dir/name
            verification = verify_pair(pair/'off', pair/'on')
            if verification['status'] != 'PASS':
                raise ValueError(f'{name}: saved monitor pair no longer matches')
            paths = [pair/mode/file for mode, files in (
                ('off', ('metrics.json', 'control_trace.npz', 'trajectory.csv')),
                ('on', ('metrics.json', 'control_trace.npz', 'trajectory.csv',
                        'shadow_signals.csv', 'shadow_events.json'))) for file in files]
            manifest['inputs'][name] = {str(p.relative_to(args.source_dir)): sha(p) for p in paths}
            source_metrics = json.loads((pair/'on/metrics.json').read_text())
            if source_metrics['policy_sha256'] != sha(actor) or source_metrics['metadata_sha256'] != sha(metadata):
                raise ValueError(f'{name}: current actor or metadata differs from recorded inputs')
            signals = read_csv(pair/'on/shadow_signals.csv')
            trajectory = read_csv(pair/'on/trajectory.csv')
            t = np.array([float(r['time_s']) for r in signals])
            errors = np.array([float(r['yaw_error_rps']) for r in signals])
            if not all(float(r['command_yaw_rps']) == yaw for r in signals):
                raise ValueError('Unexpected variable command')
            with np.load(pair/'on/control_trace.npz', allow_pickle=False) as trace:
                states = trace['sampled_states']
            if states.shape != (len(t), 87):
                raise ValueError('Expected 44 qpos + 43 qvel in saved 37D model')
            heading = heading_from_wxyz(states[:, 3:7])
            recorded = np.array([float(r['yaw_rad']) for r in trajectory])
            heading_mismatch = float(np.max(np.abs(np.arctan2(np.sin(heading-recorded), np.cos(heading-recorded)))))
            if heading_mismatch > 1e-9:
                raise ValueError('Root quaternion yaw disagrees with recorded pelvis yaw')
            archived = json.loads((pair/'on/shadow_events.json').read_text())
            legacy = LocomotionShadow(ShadowThresholds(**archived['thresholds']))
            for row in signals:
                legacy.observe(ShadowSample(**{key: float(row[key]) for key in (
                    'time_s', 'tilt_deg', 'angular_speed_rps', 'body_forward_speed_mps',
                    'body_yaw_rate_rps', 'command_forward_mps', 'command_yaw_rps')}))
            if legacy.events != archived['events']:
                raise ValueError('Offline legacy replay does not reproduce saved events')
            legacy_yaw = [e for e in legacy.events if e['reason']=='yaw_error']
            post_runs = exceedance_runs(t, errors, start_s=WARMUP)
            startup = phase_stats(t, errors, heading, yaw, 0., WARMUP)
            post = phase_stats(t, errors, heading, yaw, WARMUP, float(t[-1]))
            mean, rmse = trailing_error_stats(t, errors, 1.)
            heading_mean, heading_rmse = trailing_heading_stats(t, heading, yaw, 1.)
            eligible = WARMUP+1.
            body_events = bias_events(t, mean, .3, DWELL, eligible)
            heading_events = bias_events(t, heading_mean, .3, DWELL, eligible)
            primary = event_summary(t, body_events, eligible)
            heading_primary = event_summary(t, heading_events, eligible)
            q = states[:, 3:7]/np.linalg.norm(states[:, 3:7], axis=1)[:, None]
            w, x, y, z = q.T
            max_pitch = float(np.max(np.abs(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))))
            if np.cos(max_pitch) < .1:
                raise ValueError('Euler heading approaches pitch singularity')
            run = dict(policy=policy, command_yaw_rps=yaw, source_samples=len(t),
                startup=startup, poststartup=post, legacy_yaw_events=legacy_yaw,
                longest_poststartup_continuous_exceedance_s=post_runs['longest_sampled_exceedance_s'],
                poststartup_exceedance_run_count=post_runs['run_count'],
                above_threshold_sample_fraction=post_runs['above_threshold_sample_fraction'],
                primary_body_window=primary, primary_heading_window=heading_primary,
                body_window_events=body_events, heading_window_events=heading_events,
                net_unwrapped_heading_rad=float(heading[-1]-heading[0]),
                quaternion_vs_recorded_yaw_max_circular_error_rad=heading_mismatch,
                max_abs_pitch_deg=float(np.degrees(max_pitch)),
                max_adjacent_heading_change_rad=float(np.max(np.abs(np.diff(heading)))),
                source_rollout_config=source_metrics['rollout_config'],
                policy_sha256=source_metrics['policy_sha256'], metadata_sha256=source_metrics['metadata_sha256'],
                reference_head_recorded=source_metrics['reference_head'], source_pair_status=verification['status'])
            runs.append(run)
            metrics_rows.append(dict(policy=policy, command_yaw_rps=yaw,
                startup_body_mean_error_rps=startup['body_mean_error_rps'],
                poststartup_body_mean_error_rps=post['body_mean_error_rps'],
                poststartup_body_error_rmse_rps=post['body_error_rmse_rps'],
                poststartup_heading_mean_error_rps=post['heading_mean_error_rps'],
                legacy_yaw_enter_count=sum(e['event']=='candidate_enter' for e in legacy_yaw),
                legacy_yaw_poststartup_enter_count=sum(e['event']=='candidate_enter' and e['time_s']>=WARMUP for e in legacy_yaw),
                longest_legacy_poststartup_run_s=post_runs['longest_sampled_exceedance_s'],
                window_enter_count=primary['candidate_enter_count'], window_first_s=primary['first_candidate_time_s'],
                window_active_duration_s=primary['active_duration_s'],
                heading_window_first_s=heading_primary['first_candidate_time_s'],
                heading_net_rad=run['net_unwrapped_heading_rad']))
            for window in WINDOWS:
                bm, br = trailing_error_stats(t, errors, window)
                hm, hr = trailing_heading_stats(t, heading, yaw, window)
                for threshold in THRESHOLDS:
                    for signal, values, rms in (('body_angular_z', bm, br), ('euler_heading_rate', hm, hr)):
                        eligible = WARMUP+window
                        events = bias_events(t, values, threshold, DWELL, eligible)
                        sweep.append(dict(policy=policy, command_yaw_rps=yaw, signal=signal,
                            window_s=window, threshold_rps=threshold, dwell_s=DWELL, warmup_s=WARMUP,
                            **event_summary(t, events, eligible),
                            mean_poststartup_window_rmse_rps=float(np.mean(rms[t>=eligible-1e-10]))))
            write_csv(args.trace_output_dir/f'{name}_windows.csv', [dict(
                time_s=float(t[i]), instantaneous_body_error_rps=float(errors[i]),
                body_mean_error_rps=None if np.isnan(mean[i]) else float(mean[i]),
                body_rmse_rps=None if np.isnan(rmse[i]) else float(rmse[i]),
                heading_mean_error_rps=None if np.isnan(heading_mean[i]) else float(heading_mean[i]),
                heading_rmse_rps=None if np.isnan(heading_rmse[i]) else float(heading_rmse[i]),
                unwrapped_heading_rad=float(heading[i])) for i in range(len(t))])
            (args.trace_output_dir/f'{name}_legacy_runs.json').write_text(json.dumps(post_runs, indent=2)+'\n')
            print(f"{name}: legacy yaw enters={metrics_rows[-1]['legacy_yaw_enter_count']}, "
                  f"max run={post_runs['longest_sampled_exceedance_s']:.3f}s, "
                  f"window first={primary['first_candidate_time_s']}s", flush=True)
    # Prove analysis only read the saved rollouts.
    for items in manifest['inputs'].values():
        for relative, before in items.items():
            if sha(args.source_dir/relative) != before:
                raise ValueError(f'Input changed during analysis: {relative}')
    report = dict(analysis_time_utc=datetime.now(timezone.utc).isoformat(), mode='offline_only',
        inputs_unchanged=True, simulator_steps_executed=0, recorder_modified=False,
        primary_config=dict(window_s=1., threshold_rps=.3, dwell_s=DWELL, warmup_s=WARMUP,
                            first_eligible_full_poststartup_window_s=3.),
        interpolation=dict(body_error='Piecewise linear; exact first and second moment integrals.',
                           heading='Quaternion wxyz -> world ZYX Euler yaw -> unwrap; '
                                   'piecewise constant backward interval rate.'),
        scope='Exploratory diagnostics on six existing seed-42 runs. No real-run anomaly labels; '
              'no measured real-world false-positive rate, safety boundary or recovery authority.', runs=runs)
    write_csv(args.output_dir/'metrics.csv', metrics_rows)
    write_csv(args.output_dir/'parameter_sweep.csv', sweep)
    write_csv(args.output_dir/'synthetic_controls.csv', controls())
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
