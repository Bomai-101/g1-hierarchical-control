"""Current-interface handoff continuity and posture-probe lifecycle."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.hierarchy.experimental_hold import ExperimentalPostureHold

class HoldTests(unittest.TestCase):
 def new(self,mode='capture_pose'):return ExperimentalPostureHold(np.zeros(37),.5,np.full(37,-1.),np.ones(37),mode=mode)
 def test_capture_entry_and_continuous_handoff(self):
  s=self.new();obs=np.zeros(123);obs[12:49]=.2;previous=np.full(37,-.4);s.enter(5.,obs,previous)
  np.testing.assert_array_equal(s.act(5.,obs),previous);np.testing.assert_allclose(s.act(5.25,obs),np.zeros(37),atol=1e-15);np.testing.assert_allclose(s.act(5.5,obs),np.full(37,.4))
 def test_capture_is_frozen_and_clipped(self):
  s=self.new();obs=np.zeros(123);obs[12:49]=3.;s.enter(0,obs,np.zeros(37));obs[12:49]=-.9;np.testing.assert_array_equal(s.act(.5,obs),np.full(37,2.))
 def test_nominal_action_and_nonzero_defaults(self):
  s=ExperimentalPostureHold(np.full(37,.2),.5,np.full(37,-1),np.ones(37),mode='nominal_pose');s.enter(0,np.zeros(123),np.ones(37));np.testing.assert_array_equal(s.act(.5,np.zeros(123)),np.zeros(37))
 def test_wrong_interface_or_nonfinite_rejected(self):
  for obs in (np.zeros(64),np.full(123,np.nan)):
   with self.assertRaises(ValueError):self.new().enter(0,obs,np.zeros(37))
  with self.assertRaises(ValueError):self.new().enter(0,np.zeros(123),np.zeros(6))
 def test_lifecycle_and_time_validation(self):
  s=self.new()
  with self.assertRaises(RuntimeError):s.act(0,np.zeros(123))
  s.enter(1,np.zeros(123),np.zeros(37));s.act(1,np.zeros(123))
  with self.assertRaises(RuntimeError):s.enter(2,np.zeros(123),np.zeros(37))
  for t in (1.,.9,float('nan')):
   with self.assertRaises(ValueError):s.act(t,np.zeros(123))

if __name__=='__main__':unittest.main()
