"""Meaningful timing/settling boundary cases for command-response analysis."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_command_stop import command_at,settling_evidence

class ResponseTests(unittest.TestCase):
 def test_continuous_ramp_and_all_channels(self):
  np.testing.assert_allclose(command_at(5.22,-.2,'ramp'),[.5,0,-.2]);np.testing.assert_allclose(command_at(5.47,-.2,'ramp'),[.25,0,-.1]);np.testing.assert_allclose(command_at(5.72,-.2,'ramp'),[0,0,0],atol=1e-15)
  np.testing.assert_array_equal(command_at(5.20,.2,'zero'),[.5,0,.2]);np.testing.assert_array_equal(command_at(5.22,.2,'zero'),[0,0,0])
 def test_confirmation_cannot_precede_full_window_and_dwell(self):
  t=np.arange(101)*.02;r=settling_evidence(t,np.zeros(101),np.zeros(101));self.assertAlmostEqual(r['settling_start_s'],.5);self.assertAlmostEqual(r['settling_confirmed_s'],1.5);self.assertTrue(r['eligible_through_end_after_confirmation'])
 def test_yaw_motion_prevents_false_stop(self):
  t=np.arange(101)*.02;r=settling_evidence(t,np.zeros(101),np.full(101,.1));self.assertIsNone(r['settling_confirmed_s'])
 def test_escape_reported_after_initial_settling(self):
  t=np.arange(201)*.02;speed=np.where(t>2.,.3,0.);r=settling_evidence(t,speed,np.zeros(201));self.assertAlmostEqual(r['settling_confirmed_s'],1.5);self.assertGreater(r['first_escape_after_confirmation_s'],2.);self.assertFalse(r['eligible_through_end_after_confirmation'])
 def test_brief_slow_interval_does_not_confirm(self):
  t=np.arange(101)*.02;speed=np.where((t>=.2)&(t<1.3),0.,.3);self.assertIsNone(settling_evidence(t,speed,np.zeros(101))['settling_confirmed_s'])

if __name__=='__main__':unittest.main()
