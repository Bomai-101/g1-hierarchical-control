import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_skill_sequence_retest import measure_case

def fixture():
    t=np.arange(401)*.02;s=np.zeros((len(t),1,21));s[:,0,0]=np.where(t<=2,t,2+.01*(t-2));s[:,0,3]=1
    z=dict(time=t,signals=s,valid=np.ones((len(t),1),bool))
    r=dict(planned_trigger_s=.8,hold_start_s=1.,resume_start_s=4.,drop_start_s=None,decision_s=1.)
    return z,r

def test_stop_and_hold_are_separate():
    z,r=fixture();m=measure_case(z,0,r,len(z['time']),4.)
    np.testing.assert_allclose([m['stop_net_m'],m['hold_net_drift_m'],m['request_to_stop_net_m']],[1.,.02,1.2],atol=1e-12)
    np.testing.assert_allclose([m['stopping_latency_s'],m['hold_duration_s']],[1.,2.])

def test_failed_reset_is_excluded():
    z,r=fixture();r['resume_start_s']=None;z['valid'][250:,0]=False;z['signals'][250:,0,0]=1000
    m=measure_case(z,0,r,len(z['time']),4.)
    np.testing.assert_allclose(m['valid_sample_end_s'],4.98)
    if m['hold_net_drift_m']>1:raise RuntimeError('Reset jump included')

def test_unconfirmed_stop_has_no_hold_metric():
    z,r=fixture();m=measure_case(z,0,r,len(z['time']),None)
    if m['hold_net_drift_m'] is not None or m['stop_net_m'] is not None:raise RuntimeError('Missing stop converted to zero')

def test_earlier_confirmation_is_not_inferred_from_later_gate():
    z,r=fixture();m=measure_case(z,0,r,len(z['time']),4.,first_confirmation=1.5)
    np.testing.assert_allclose([m['stopping_latency_s'],m['stop_net_m'],m['hold_net_drift_m']],[.5,.5,.52],atol=1e-12)

def test_resume_confirmation_uses_full_one_second_dwell():
    z,r=fixture();z['signals'][:,0,13]=.5
    m=measure_case(z,0,r,len(z['time']),4.)
    np.testing.assert_allclose(m['resume_speed_confirm_latency_s'],1.5,atol=1e-12)
