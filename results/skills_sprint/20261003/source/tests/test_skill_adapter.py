import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.hierarchy.skill_adapter import skill_observation,handoff_action

class AdapterTests(unittest.TestCase):
 def test_hold_keeps_schema(self):
  x=np.arange(123,dtype=np.float32);y=skill_observation(x,'hold',0);np.testing.assert_array_equal(x,y);self.assertFalse(np.shares_memory(x,y))
 def test_phase_is_explicit_and_periodic(self):
  x=np.arange(123,dtype=np.float32);a=skill_observation(x,'march',0);b=skill_observation(x,'march',.9);np.testing.assert_array_equal(a[:123],x);self.assertEqual(a.shape,(125,));np.testing.assert_allclose(a[-2:],[0,1],atol=1e-6);np.testing.assert_allclose(a,b,atol=1e-6)
 def test_handoff_first_and_final_actions(self):
  prev=np.ones(37);proposal=np.full(37,-1.);np.testing.assert_array_equal(handoff_action(prev,proposal,0),prev);np.testing.assert_array_equal(handoff_action(prev,proposal,.25),np.zeros(37));np.testing.assert_array_equal(handoff_action(prev,proposal,.5),proposal)
 def test_invalid_inputs_rejected(self):
  with self.assertRaises(ValueError):skill_observation(np.zeros(64),'hold',0)
  with self.assertRaises(ValueError):skill_observation(np.zeros(123),'march',float('nan'))
  with self.assertRaises(ValueError):handoff_action(np.zeros(37),np.zeros(6),0)
  with self.assertRaises(ValueError):handoff_action(np.zeros(37),np.zeros(37),-.1)

if __name__=='__main__':unittest.main()
