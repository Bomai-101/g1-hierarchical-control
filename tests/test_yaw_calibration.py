"""Check labeled scoring, command timing and reproducible signal controls."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.monitoring.yaw_calibration import (
    evaluate_case, make_case, score_events, tracking_error,
)
from g1_control.monitoring.yaw_window import trailing_error_stats, bias_events


class CalibrationTests(unittest.TestCase):
    def test_perfect_command_step_does_not_create_bias(self):
        t=np.arange(0, 12.01, .02)
        command=np.where(t>=8, .8, 0.)
        error=tracking_error(command, command)
        mean,_=trailing_error_stats(t, error, 1.)
        self.assertFalse(bias_events(t, mean, eligible_after_s=3.))

    def test_preexisting_alarm_is_not_new_detection(self):
        events=[dict(event='candidate_enter',time_s=4.,direction=-1)]
        r=score_events([0,20],events,8.,-1,3.)
        self.assertFalse(r['detected'])
        self.assertTrue(r['preexisting_alarm_at_onset'])
        self.assertEqual(r['false_active_duration_s'],4.)

    def test_wrong_direction_is_not_detection(self):
        r=score_events([0,20],[dict(event='candidate_enter',time_s=9.,direction=1)],8.,-1,3.)
        self.assertFalse(r['detected'])
        self.assertEqual(r['wrong_direction_entries'],1)

    def test_labeled_detection_delay(self):
        events=[dict(event='candidate_enter',time_s=9.,direction=-1),
                dict(event='candidate_clear',time_s=10.,direction=-1)]
        r=score_events([0,20],events,8.,-1,3.)
        self.assertTrue(r['detected'])
        self.assertEqual(r['detection_delay_s'],1.)
        self.assertFalse(r['preexisting_alarm_at_onset'])
        self.assertEqual(r['false_active_duration_s'],0.)

    def test_negative_alarm_duration_clips_to_exposure(self):
        r=score_events([0,20],[dict(event='candidate_enter',time_s=4.,direction=-1)],None,0,3.)
        self.assertEqual(r['false_candidate_count'],1)
        self.assertEqual(r['false_active_duration_s'],16.)
        self.assertEqual(r['false_exposure_s'],17.)

    def test_generation_reproducible_and_seed_varies(self):
        a,b,c=make_case(1000,'negative_bias'),make_case(1000,'negative_bias'),make_case(1001,'negative_bias')
        np.testing.assert_array_equal(a['measured_body_z'],b['measured_body_z'])
        self.assertFalse(np.array_equal(a['measured_body_z'],c['measured_body_z']))
        self.assertEqual(a['direction'],-1)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            score_events([0,20],[],2.,-1,3.)
        with self.assertRaises(ValueError):
            tracking_error([0,1],[0])


if __name__=='__main__':
    unittest.main()
