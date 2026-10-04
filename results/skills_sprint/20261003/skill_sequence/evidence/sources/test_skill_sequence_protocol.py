import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skill_sequence_protocol import Sequence,HOLD,RESUME,ABORT

def signal(speed=0.):
    s=np.zeros(21);s[3]=1.;s[13]=speed;return s

def test_gate_requires_retained_readiness():
    q=Sequence(0.,'blend')
    for i in range(177):q.update(i*.02,signal(),[True,True])
    assert q.stage==RESUME and abs(q.resume_start-3.52)<1e-7

def test_motion_and_contact_block_resume():
    for speed,feet in [(1.,[True,True]),(0.,[True,False])]:
        q=Sequence(0.,'direct')
        for i in range(601):q.update(i*.02,signal(speed),feet)
        assert q.stage==ABORT and q.resume_start is None

def test_braking_and_blend():
    q=Sequence(1.,'brake_blend');q.update(1.,signal(),[True,True])
    np.testing.assert_allclose(q.command(1.75,.2),[.25,0.,.1])
    q.update(2.5,signal(),[True,True]);assert q.stage==HOLD and q.weight(2.5)==0
    assert q.weight(3.)==1

def test_failure_aborts():
    q=Sequence(1.,'blend');assert q.update(.5,signal(),[True,True],False)==ABORT
