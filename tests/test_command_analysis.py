from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.monitoring.command_analysis import scheduled_yaw_stats
from g1_control.monitoring.yaw_window import trailing_error_stats,trailing_heading_stats


class CommandAnalysisTests(unittest.TestCase):
    def test_step_cannot_change_preceding_interval(self):
        r=scheduled_yaw_stats([0,.5,1,1.5],[0,0,0,0],[0,0,0,0],[0,0,.4,.4],1.)
        self.assertAlmostEqual(r['body_mean'][2],0.)
        self.assertAlmostEqual(r['heading_mean'][2],0.)
        self.assertAlmostEqual(r['body_mean'][3],-.2)
        self.assertAlmostEqual(r['heading_mean'][3],-.2)
        self.assertAlmostEqual(r['body_rmse'][3]**2,.08)
        self.assertAlmostEqual(r['reference_heading'][3],.2)

    def test_constant_command_matches_previous_definitions(self):
        t=np.array([0.,.1,.7,1.1,2.]);body=np.sin(t);heading=t*.5
        r=scheduled_yaw_stats(t,body,heading,np.full_like(t,.2),.8)
        bm,br=trailing_error_stats(t,body-.2,.8)
        hm,hr=trailing_heading_stats(t,heading,.2,.8)
        for actual,expected in ((r['body_mean'],bm),(r['body_rmse'],br),
                                (r['heading_mean'],hm),(r['heading_rmse'],hr)):
            np.testing.assert_allclose(actual,expected,equal_nan=True)

    def test_future_command_cannot_change_prefix(self):
        t=np.arange(0,4.01,.02);cmd=np.zeros_like(t)
        before=scheduled_yaw_stats(t,np.sin(t),t*.1,cmd,1.)
        cmd[t>2]=.8
        after=scheduled_yaw_stats(t,np.sin(t),t*.1,cmd,1.)
        for key in before:np.testing.assert_array_equal(before[key][t<=2],after[key][t<=2])


if __name__=='__main__':unittest.main()
