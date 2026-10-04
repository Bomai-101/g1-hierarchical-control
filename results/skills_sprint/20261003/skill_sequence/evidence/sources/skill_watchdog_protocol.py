"""Experimental heartbeat watchdog around the verified low-speed sequence."""
import numpy as np
from skill_sequence_protocol import Sequence,WALK,HOLD,RESUME,ABORT,NAMES

class WatchdogSequence(Sequence):
    def __init__(self, decision, fault):
        super().__init__(float('inf'),'direct')
        self.drop_start=decision;self.fault=fault;self.last_message=None;self.stale=False;self.fresh_count=0;self.watchdog_events=[];self.permission_block_recorded=False

    def update(self, now, signal, contacts, alive=True):
        tick=round(now/.02)
        missing=now>=self.drop_start-1e-8 and (self.fault=='persistent' or now<self.drop_start+1.-1e-8)
        received=tick%5==0 and not missing
        if received:
            self.last_message=now
            if self.stale:
                self.fresh_count+=1
                if self.fresh_count>=2:self.stale=False;self.watchdog_events.append(dict(time_s=now,event='fresh_two_messages'))
        elif self.stale and self.last_message is not None and now-self.last_message>.3+1e-8:self.fresh_count=0
        age=float('inf') if self.last_message is None else now-self.last_message
        if age>.3+1e-8 and not self.stale:
            self.stale=True;self.fresh_count=0;self.watchdog_events.append(dict(time_s=now,event='stale'))
            if self.stage==WALK:self.decision=now;self.hold_start=now
        stage=super().update(now,signal,contacts,alive)
        if stage==RESUME and self.stale:
            self.stage=HOLD;self.resume_start=None
            if self.events and self.events[-1]['stage']=='resume_walk':self.events.pop()
            if not self.permission_block_recorded:self.watchdog_events.append(dict(time_s=now,event='resume_blocked_stale'));self.permission_block_recorded=True
        if self.stage==HOLD and self.stale and now-self.hold_start>=12.-1e-8:
            self.stage=ABORT;self.events.append(dict(time_s=now,previous=NAMES[HOLD],stage=NAMES[ABORT],reason='bounded_hold_observation_ended'))
        return self.stage
