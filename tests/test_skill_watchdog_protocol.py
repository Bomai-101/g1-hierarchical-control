import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skill_watchdog_protocol import WatchdogSequence
from skill_sequence_protocol import DONE,ABORT

def run(fault):
    q=WatchdogSequence(1.02,fault);s=np.zeros(21);s[3]=1
    for i in range(751):q.update(i*.02,s,[True,True])
    return q

def test_temporary_returns_only_after_fresh():
    q=run('temporary');assert abs(q.decision-1.32)<1e-8
    fresh=next(e['time_s'] for e in q.watchdog_events if e['event']=='fresh_two_messages')
    assert abs(fresh-2.2)<1e-8 and q.resume_start>=fresh and q.stage==DONE

def test_persistent_never_resumes():
    q=run('persistent');assert q.resume_start is None and q.stage==ABORT
    assert any(e['event']=='resume_blocked_stale' for e in q.watchdog_events)
