"""Verify paired recorder outputs without modifying either rollout."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def verify_pair(off_dir, on_dir):
    off_dir, on_dir = Path(off_dir), Path(on_dir)
    off = json.loads((off_dir / 'metrics.json').read_text())
    on = json.loads((on_dir / 'metrics.json').read_text())
    if off['shadow_enabled'] or not on['shadow_enabled']:
        raise AssertionError('Expected monitoring off/on, respectively')
    identical_metrics = {k: v for k, v in off.items() if k != 'shadow_enabled'} == {
        k: v for k, v in on.items() if k != 'shadow_enabled'}
    with np.load(off_dir / 'control_trace.npz', allow_pickle=False) as a, np.load(
        on_dir / 'control_trace.npz', allow_pickle=False
    ) as b:
        states_equal = np.array_equal(a['sampled_states'], b['sampled_states'])
        actions_equal = np.array_equal(a['actions'], b['actions'])
        state_samples, policy_actions = len(a['sampled_states']), len(a['actions'])
    digest_equal = off['physics_state_control_sha256'] == on['physics_state_control_sha256']
    trajectory_equal = (off_dir / 'trajectory.csv').read_bytes() == (on_dir / 'trajectory.csv').read_bytes()
    with (on_dir / 'shadow_signals.csv').open(newline='') as stream:
        signals = list(csv.DictReader(stream))
    events = json.loads((on_dir / 'shadow_events.json').read_text())
    if state_samples == 0 or policy_actions == 0:
        raise ValueError('Empty control trace')
    if len(signals) != state_samples:
        raise ValueError('Monitor must record every sampled state')
    if events['mode'] != 'shadow_only':
        raise ValueError('Expected passive monitoring mode')
    if (off_dir / 'shadow_signals.csv').exists():
        raise ValueError('Off run contains monitor signals')
    if (off_dir / 'shadow_events.json').exists():
        raise ValueError('Off run contains monitor events')
    with (on_dir / 'trajectory.csv').open(newline='') as stream:
        trajectory_rows = list(csv.DictReader(stream))
    if [r['time_s'] for r in signals] != [r['time_s'] for r in trajectory_rows]:
        raise ValueError('Signal timestamps differ')
    summary = dict(
        status='PASS' if all((identical_metrics, states_equal, actions_equal, digest_equal, trajectory_equal)) else 'FAIL',
        duration_s=on['elapsed_s'], physics_steps=on['physics_steps'],
        sampled_state_count=state_samples, policy_action_count=policy_actions,
        metrics_equal_except_monitor_flag=identical_metrics,
        sampled_states_exactly_equal=states_equal, policy_actions_exactly_equal=actions_equal,
        per_physics_step_state_control_digest_equal=digest_equal,
        trajectory_csv_exactly_equal=trajectory_equal,
        policy_sha256=on['policy_sha256'], reference_head=on['reference_head'],
        mean_signed_body_yaw_rate_rps=float(np.mean([float(r['body_yaw_rate_rps']) for r in signals])),
        candidate_enter_count=sum(e['event'] == 'candidate_enter' for e in events['events']),
        first_candidate_entries=[e for e in events['events'] if e['event'] == 'candidate_enter'][:5],
        command=[on['command_x'], on['command_y'], on['command_yaw']],
        terrain=on['terrain'], friction=on['friction'],
        rollout_config=on.get('rollout_config'),
        scope='This paired rollout only; no safety/recovery or heading-tracking claim.',
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; refusing to overwrite')
    summary = verify_pair(args.off, args.on)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
