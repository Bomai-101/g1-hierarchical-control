"""Independent 123D/37D posture probes, with no watchdog/fallback registration.

These position holds are diagnostic candidates, not balance controllers.
Their affine action/target mapping matches the current locomotion interface.
"""
import numpy as np


class ExperimentalPostureHold:
    def __init__(self, default_positions, scale, lower, upper, *, mode, blend_s=.5):
        self.default=self._array(default_positions,37)
        self.lower=self._array(lower,37);self.upper=self._array(upper,37)
        if not np.isfinite(scale) or scale<=0 or not np.isfinite(blend_s) or blend_s<=0:
            raise ValueError('Positive finite scale and blend time required')
        if np.any(self.lower>self.upper) or mode not in ('capture_pose','nominal_pose'):
            raise ValueError('Invalid bounds or mode')
        self.scale=float(scale);self.blend_s=float(blend_s);self.mode=mode
        self.start=None;self.previous_time=None;self.initial=None;self.goal=None

    @staticmethod
    def _array(values,n):
        array=np.asarray(values,dtype=float)
        if array.shape!=(n,) or not np.all(np.isfinite(array)):
            raise ValueError(f'Finite {n}D array required')
        return array.copy()

    def enter(self, time_s, observation, previous_action):
        if self.start is not None:raise RuntimeError('Explicit new probe required for re-entry')
        if not np.isfinite(time_s) or time_s<0:raise ValueError('Invalid entry time')
        obs=self._array(observation,123);self.initial=self._array(previous_action,37)
        positions=self.default+obs[12:49] if self.mode=='capture_pose' else self.default
        self.goal=(np.clip(positions,self.lower,self.upper)-self.default)/self.scale
        self.start=float(time_s)

    def act(self,time_s,observation):
        self._array(observation,123)
        if self.start is None:raise RuntimeError('Probe must enter explicitly')
        if not np.isfinite(time_s) or time_s<self.start or (self.previous_time is not None and time_s<=self.previous_time):
            raise ValueError('Increasing finite sample time required')
        self.previous_time=float(time_s);u=float(np.clip((time_s-self.start)/self.blend_s,0,1));weight=u*u*(3-2*u)
        return self.initial*(1-weight)+self.goal*weight
