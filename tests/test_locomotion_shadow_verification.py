"""Ensure passive-monitor verification rejects altered or incomplete evidence."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_locomotion_shadow import verify_pair


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.off, self.on = (Path(self.tmp.name) / mode for mode in ('off', 'on'))
        self.states = np.array([[0., 1.], [0.1, 1.1]])
        self.actions = np.array([[0.25, -0.25]])
        for directory in (self.off, self.on):
            directory.mkdir()
            metrics = dict(shadow_enabled=directory == self.on, elapsed_s=.02,
                           physics_steps=20, physics_state_control_sha256='a' * 64,
                           policy_sha256='b' * 64, reference_head='c' * 40,
                           command_x=1., command_y=0., command_yaw=-.2,
                           terrain='plane', friction=.8)
            (directory / 'metrics.json').write_text(json.dumps(metrics))
            np.savez(directory / 'control_trace.npz', sampled_states=self.states, actions=self.actions)
            (directory / 'trajectory.csv').write_text('time_s,world_x_m\n0.0,0.0\n0.02,0.1\n')
        self.write_signals([('0.0', '-0.3'), ('0.02', '-0.4')])
        (self.on / 'shadow_events.json').write_text(json.dumps(dict(mode='shadow_only', events=[])))

    def write_signals(self, rows):
        with (self.on / 'shadow_signals.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'body_yaw_rate_rps'])
            writer.writerows(rows)

    def test_exact_pair_and_actual_command(self):
        report = verify_pair(self.off, self.on)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['command'], [1., 0., -.2])
        self.assertEqual(report['policy_action_count'], 1)

    def test_action_difference_fails_even_when_metrics_match(self):
        changed = self.actions.copy()
        changed[0, 0] += 1e-12
        np.savez(self.on / 'control_trace.npz', sampled_states=self.states, actions=changed)
        report = verify_pair(self.off, self.on)
        self.assertEqual(report['status'], 'FAIL')
        self.assertFalse(report['policy_actions_exactly_equal'])

    def test_physics_digest_difference_fails(self):
        path = self.on / 'metrics.json'
        metrics = json.loads(path.read_text())
        metrics['physics_state_control_sha256'] = 'd' * 64
        path.write_text(json.dumps(metrics))
        self.assertEqual(verify_pair(self.off, self.on)['status'], 'FAIL')

    def test_missing_sample_rejected(self):
        self.write_signals([('0.0', '-0.3')])
        with self.assertRaisesRegex(ValueError, 'every sampled state'):
            verify_pair(self.off, self.on)

    def test_wrong_sample_time_rejected(self):
        self.write_signals([('0.0', '-0.3'), ('0.03', '-0.4')])
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            verify_pair(self.off, self.on)

    def test_empty_evidence_rejected(self):
        for directory in (self.off, self.on):
            np.savez(directory / 'control_trace.npz', sampled_states=np.empty((0, 2)),
                     actions=np.empty((0, 2)))
        with self.assertRaisesRegex(ValueError, 'Empty control trace'):
            verify_pair(self.off, self.on)


if __name__ == '__main__':
    unittest.main()
