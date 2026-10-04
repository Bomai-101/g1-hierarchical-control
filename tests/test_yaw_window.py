"""Analytic and causal checks for passive window diagnostics."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from g1_control.monitoring.yaw_window import (
    bias_events, exceedance_runs, heading_from_wxyz,
    trailing_error_stats, trailing_heading_stats,
)


class WindowTests(unittest.TestCase):
    def test_irregular_linear_signal_exact_integrals(self):
        t = np.array([0., .1, .7, 1.1, 2.])
        mean, rmse = trailing_error_stats(t, t, .8)
        self.assertTrue(np.isnan(mean[0]))
        self.assertAlmostEqual(mean[-1], 1.6)
        self.assertAlmostEqual(rmse[-1]**2, (2**3-1.2**3)/(3*.8))

    def test_signed_cancellation_retains_rmse(self):
        mean, rmse = trailing_error_stats([0., 1., 2.], [-1., 0., 1.], 2.)
        self.assertAlmostEqual(mean[-1], 0.)
        self.assertAlmostEqual(rmse[-1], 1/np.sqrt(3))

    def test_future_data_cannot_change_prefix(self):
        t = np.arange(0, 3.01, .02)
        errors = np.sin(t)
        before = trailing_error_stats(t, errors, .5)
        errors[t > 2.] += 100
        after = trailing_error_stats(t, errors, .5)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a[t <= 2.], b[t <= 2.])

    def test_startup_exclusion_and_same_sign_dwell(self):
        t = np.arange(0, 1.01, .1)
        events = bias_events(t, np.full_like(t, .4), eligible_after_s=.5)
        self.assertAlmostEqual(events[0]['time_s'], .7)
        self.assertFalse(bias_events([0., .1, .2, .3], [.4, .4, -.4, -.4]))

    def test_oscillatory_bias_evades_legacy_but_window_detects(self):
        t = np.arange(0, 6.01, .02)
        errors = -.4 + .8*np.sin(8*np.pi*t)
        self.assertLess(exceedance_runs(t, errors)['longest_sampled_exceedance_s'], .2)
        mean, _ = trailing_error_stats(t, errors, 1.)
        events = bias_events(t, mean, eligible_after_s=3.)
        self.assertEqual(events[0]['direction'], -1)
        self.assertAlmostEqual(events[0]['time_s'], 3.2)

    def test_zero_mean_gait_does_not_trigger_mean_bias(self):
        t = np.arange(0, 6.01, .02)
        mean, rmse = trailing_error_stats(t, .8*np.sin(8*np.pi*t), 1.)
        self.assertFalse(bias_events(t, mean, eligible_after_s=3.))
        self.assertGreater(np.nanmean(rmse), .3)

    def test_slow_zero_mean_oscillation_is_a_documented_false_alarm(self):
        t = np.arange(0, 20.01, .02)
        error = .8*np.sin(np.pi*t)
        self.assertAlmostEqual(np.trapezoid(error, t), 0.)
        mean, _ = trailing_error_stats(t, error, 1.)
        self.assertTrue(bias_events(t, mean, eligible_after_s=3.))

    def test_quaternion_unwrap_crosses_pi_without_rate_spike(self):
        yaw = np.array([3., 3.2, 3.4])
        q = np.c_[np.cos(yaw/2), np.zeros((3, 2)), np.sin(yaw/2)]
        heading = heading_from_wxyz(q)
        np.testing.assert_allclose(heading, yaw)
        mean, rmse = trailing_heading_stats([0., .1, .2], heading, 1.5, .2)
        self.assertAlmostEqual(mean[-1], .5)
        self.assertAlmostEqual(rmse[-1], .5)

    def test_heading_rmse_on_irregular_intervals(self):
        mean, rmse = trailing_heading_stats([0., .3, 1.], [0., .6, .6], 0., .8)
        self.assertAlmostEqual(mean[-1], .25)
        self.assertAlmostEqual(rmse[-1]**2, .5)

    def test_invalid_data_rejected(self):
        for times, errors, window in (([0., 0.], [0., 1.], 1.),
                                       ([0., 1.], [0., np.nan], 1.),
                                       ([0., 1.], [0., 1.], 0.)):
            with self.assertRaises(ValueError):
                trailing_error_stats(times, errors, window)
        with self.assertRaises(ValueError):
            heading_from_wxyz(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            bias_events([0., 1.], [0., np.inf])


if __name__ == '__main__':
    unittest.main()
