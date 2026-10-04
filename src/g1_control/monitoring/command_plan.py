"""Explicit experiment commands; independent of diagnostic candidates."""
from dataclasses import dataclass, asdict
from math import isfinite, sin, pi


@dataclass(frozen=True)
class CommandPlan:
    mode: str = 'steady'
    forward_mps: float = 1.
    steady_yaw_rps: float = 0.
    amplitude_rps: float = .2
    frequency_hz: float = .05

    def __post_init__(self):
        if self.mode not in ('steady', 'slow_sine', 'steps'):
            raise ValueError('Unknown command plan')
        if not all(isfinite(v) for k,v in asdict(self).items() if k != 'mode'):
            raise ValueError('Command parameters must be finite')
        if self.frequency_hz <= 0 or self.amplitude_rps < 0:
            raise ValueError('Invalid sine frequency or amplitude')

    def at(self, time_s):
        if not isfinite(time_s) or time_s < 0:
            raise ValueError('Command time must be finite and nonnegative')
        if self.mode == 'steady':
            yaw, epoch = self.steady_yaw_rps, 0
        elif self.mode == 'slow_sine':
            yaw, epoch = self.amplitude_rps*sin(2*pi*self.frequency_hz*time_s), 0
        else:
            epoch = sum(time_s >= edge for edge in (8.,16.,24.))
            yaw = (0.,.2,-.2,0.)[epoch]
        return (self.forward_mps,0.,yaw), epoch
