"""Frozen exploratory handoff branches; unchanged first-round skill criteria."""
import numpy as np


def cases(native=False):
    result=[]
    entries=[(0.,0.)] if native else [(0.,0.),(0.,np.pi/2)]
    for switch,phase in entries:
        result.append(dict(protocol='static',vx=0.,yaw=0.,decision_s=0.,switch_s=switch,phase_offset=phase,ramp_s=0.))
    for vx in (.5,1.):
        for yaw in (-.2,0.,.2):
            for decision,phase in ([(5.22,0.)] if native else [(5.22,0.),(5.44,np.pi/2)]):
                for protocol,ramp in (('direct',0.),('ramp_0.5',.5),('ramp_1.5',1.5)):
                    result.append(dict(protocol=protocol,vx=vx,yaw=yaw,decision_s=decision,switch_s=decision+ramp,phase_offset=phase,ramp_s=ramp))
    return result


def command_at(now,case):
    if case['protocol']=='static' or now>=case['switch_s']-1e-9:return np.zeros(3)
    command=np.array([case['vx'],0.,0. if now<2.4-1e-9 else case['yaw']])
    if case['ramp_s']>0:
        command*=np.clip(1.-(now-case['decision_s'])/case['ramp_s'],0.,1.)
    return command
