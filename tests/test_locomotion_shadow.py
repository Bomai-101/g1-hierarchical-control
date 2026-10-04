"""Standalone tests for passive temporal diagnostic screens."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from g1_control.monitoring.locomotion_shadow import LocomotionShadow, ShadowSample, ShadowThresholds


def sample(t, yaw=0.0):
    return ShadowSample(t, 0.0, abs(yaw), 1.0, yaw, 1.0, 0.0)


class ShadowTests(unittest.TestCase):
    def test_dwell_clear_and_sign(self):
        monitor = LocomotionShadow()
        self.assertEqual(monitor.observe(sample(0, -.4))['yaw_error_rps'], -.4)
        self.assertFalse(monitor.observe(sample(.18, -.4))['candidate_reasons'])
        self.assertEqual(monitor.observe(sample(.2, -.4))['candidate_reasons'], ['yaw_error'])
        self.assertFalse(monitor.observe(sample(.22))['candidate_reasons'])
        self.assertEqual([e['event'] for e in monitor.events], ['candidate_enter', 'candidate_clear'])

    def test_transient_does_not_accumulate(self):
        monitor = LocomotionShadow()
        for t, yaw in [(0, .4), (.1, 0), (.2, .4), (.3, 0)]:
            monitor.observe(sample(t, yaw))
        self.assertFalse(monitor.events)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ShadowThresholds(dwell_s=0)
        monitor = LocomotionShadow()
        monitor.observe(sample(0))
        with self.assertRaises(ValueError):
            monitor.observe(sample(0))
        with self.assertRaises(ValueError):
            LocomotionShadow().observe(sample(0, float('nan')))


if __name__ == '__main__':
    unittest.main()
