from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.monitoring.command_plan import CommandPlan


class PlanTests(unittest.TestCase):
    def test_step_boundaries(self):
        p=CommandPlan('steps',.5)
        for t,yaw,epoch in ((0,0,0),(7.999,0,0),(8,.2,1),(16,-.2,2),(24,0,3)):
            command,e=p.at(t)
            self.assertEqual(command,(.5,0.,yaw));self.assertEqual(e,epoch)

    def test_sine_and_steady(self):
        self.assertAlmostEqual(CommandPlan('slow_sine').at(5)[0][2],.2)
        self.assertEqual(CommandPlan('steady',1.,-.1).at(20)[0],(1.,0.,-.1))

    def test_invalid_parameters(self):
        for kwargs in (dict(mode='unknown'),dict(forward_mps=float('nan')),dict(frequency_hz=0)):
            with self.assertRaises(ValueError):CommandPlan(**kwargs)
        with self.assertRaises(ValueError):CommandPlan().at(-1)


if __name__=='__main__':unittest.main()
