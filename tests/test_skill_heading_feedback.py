import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skill_heading_feedback import heading_command
class FeedbackTests(unittest.TestCase):
 def test_wrap_and_correct_direction(self):
  self.assertGreater(heading_command(np.pi-.01,-np.pi+.01,3.,1.),0)
  self.assertLess(heading_command(.1,0.,3.,1.),0)
 def test_delayed_continuous_start(self):
  self.assertEqual(heading_command(1.,0.,2.,1.),0)
  self.assertAlmostEqual(heading_command(1.,0.,2.25,1.),-.1)
  self.assertAlmostEqual(heading_command(1.,0.,2.5,1.),-.2)
 def test_zero_gain_and_saturation(self):
  np.testing.assert_array_equal(heading_command([1.,-1.],0.,3.,0.),[0,0])
  np.testing.assert_allclose(heading_command([1.,-1.],0.,3.,1.),[-.2,.2])
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):heading_command(np.nan,0.,3.,1.)
if __name__=='__main__':unittest.main()
