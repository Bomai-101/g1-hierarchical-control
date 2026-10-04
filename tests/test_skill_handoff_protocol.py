import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skill_handoff_protocol import cases,command_at

class HandoffProtocolTests(unittest.TestCase):
    def test_branches_share_predecision_commands(self):
        branches=[c for c in cases() if c['vx']==.5 and c['yaw']==.2 and c['decision_s']==5.22]
        for t in (0.,2.4,5.20):
            for c in branches:np.testing.assert_array_equal(command_at(t,c),command_at(t,branches[0]))
        for c in branches[1:]:
            np.testing.assert_allclose(command_at(c['decision_s'],c),[.5,0,.2])
            np.testing.assert_allclose(command_at(c['decision_s']+c['ramp_s']/2,c),[.25,0,.1])
            np.testing.assert_array_equal(command_at(c['switch_s'],c),np.zeros(3))
    def test_static_has_no_teacher_prelude(self):
        for c in cases():
            if c['protocol']=='static':
                self.assertEqual(c['switch_s'],0.)
                np.testing.assert_array_equal(command_at(0.,c),np.zeros(3))
    def test_full_matrix_keeps_both_speeds_yaws_and_phases(self):
        self.assertEqual(len(cases()),38);self.assertEqual(len(cases(native=True)),19)
        for vx in (.5,1.):
            for yaw in (-.2,0.,.2):
                self.assertEqual(len([c for c in cases() if c['vx']==vx and c['yaw']==yaw]),6)

if __name__=='__main__':unittest.main()
