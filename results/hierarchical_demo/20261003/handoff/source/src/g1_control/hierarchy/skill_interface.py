"""Velocity-command boundary for the existing 123D/37D locomotion actor."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class VelocityCommand:
    forward_mps: float
    lateral_mps: float
    yaw_rate_rps: float

    def __post_init__(self):
        if not all(isfinite(v) for v in self.values):
            raise ValueError('Finite velocity command required')
        # Reference task command ranges, not hardware safety ratings.
        if not (0 <= self.forward_mps <= 1 and abs(self.lateral_mps) <= .5 and abs(self.yaw_rate_rps) <= 1):
            raise ValueError('Outside this demonstration command contract')

    @property
    def values(self):
        return (self.forward_mps, self.lateral_mps, self.yaw_rate_rps)
