"""Experimental causal walk/hold/walk sequencing; no watchdog authority."""
from collections import deque
import numpy as np

WALK, BRAKE, HOLD, RESUME, DONE, ABORT = range(6)
NAMES = ('walk', 'brake', 'hold', 'resume_walk', 'done', 'abort')

class Sequence:
    def __init__(self, decision, protocol):
        self.decision = decision
        self.protocol = protocol
        self.brake_s = 1.5 if protocol == 'brake_blend' else 0.
        self.blend_s = 0. if protocol == 'direct' else .5
        self.hold_start = decision + self.brake_s
        self.resume_start = None
        self.stage = WALK
        self.history = deque(maxlen=26)
        self.stable_since = None
        self.events = []
        self.last_heading = None

    def update(self, now, signal, contacts, alive=True):
        old = self.stage
        heading = float(signal[20])
        rate = np.nan if self.last_heading is None else np.arctan2(np.sin(heading-self.last_heading), np.cos(heading-self.last_heading))/.02
        self.last_heading = heading
        w,x,y,z = signal[3:7]
        tilt = max(abs(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))), abs(np.arcsin(np.clip(2*(w*y-z*x),-1,1))))
        if not alive and self.stage not in (DONE, ABORT):
            self.stage = ABORT
        elif self.stage == WALK and now >= self.decision-1e-8:
            self.stage = BRAKE if self.brake_s else HOLD
        if self.stage == BRAKE and now >= self.hold_start-1e-8:
            self.stage = HOLD
        if self.stage == HOLD:
            self.history.append((np.linalg.norm(signal[13:15]),rate))
            values = np.array(self.history)
            ready = len(values)==26 and np.isfinite(values).all() and values[:,0].mean()<=.05 and np.sqrt(np.mean(values[:,1]**2))<=.05 and tilt<=np.deg2rad(20) and bool(np.all(contacts))
            if ready:
                if self.stable_since is None:self.stable_since=now
            else:self.stable_since=None
            # One-second confirmation followed by two seconds retained readiness.
            if now-self.hold_start>=2.-1e-8 and self.stable_since is not None and now-self.stable_since>=3.-1e-8:
                self.resume_start=now;self.stage=RESUME
            elif now-self.hold_start>=12.-1e-8:self.stage=ABORT
        if self.stage==RESUME and now-self.resume_start>=10.-1e-8:self.stage=DONE
        if self.stage!=old:self.events.append(dict(time_s=now,previous=NAMES[old],stage=NAMES[self.stage]))
        return self.stage

    def command(self, now, yaw):
        if self.stage==WALK:return np.array([.5,0.,0. if now<2.4-1e-8 else yaw])
        if self.stage==BRAKE:return np.array([.5,0.,yaw])*max(0.,1-(now-self.decision)/self.brake_s)
        if self.stage in (RESUME,DONE):return np.array([.5,0.,0.])
        return np.zeros(3)

    def weight(self, now):
        start=self.resume_start if self.stage==RESUME else self.hold_start
        if not self.blend_s:return 1.
        u=np.clip((now-start)/self.blend_s,0,1)
        return float(u*u*(3-2*u))
