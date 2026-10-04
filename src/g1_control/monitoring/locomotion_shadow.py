"""Passive locomotion screens, not validated safety or recovery criteria."""

from dataclasses import dataclass, asdict
from math import isfinite


@dataclass(frozen=True)
class ShadowThresholds:
    tilt_deg: float = 15.0
    angular_speed_rps: float = 1.0
    forward_error_mps: float = 0.3
    yaw_error_rps: float = 0.3
    dwell_s: float = 0.2

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Screen thresholds must be finite and positive')


@dataclass(frozen=True)
class ShadowSample:
    time_s: float
    tilt_deg: float
    angular_speed_rps: float
    body_forward_speed_mps: float
    body_yaw_rate_rps: float
    command_forward_mps: float
    command_yaw_rps: float


class LocomotionShadow:
    """Observe scalar copies only; no action, model or simulator access."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or ShadowThresholds()
        self.events = []
        self._last_time = None
        self._pending = {}
        self._active = set()

    def observe(self, sample):
        if not all(isfinite(v) for v in asdict(sample).values()):
            raise ValueError('Signals must be finite')
        if min(sample.time_s, sample.tilt_deg, sample.angular_speed_rps) < 0:
            raise ValueError('Time and magnitudes cannot be negative')
        if self._last_time is not None and sample.time_s <= self._last_time:
            raise ValueError('Time must increase; create a new monitor after reset')
        self._last_time = sample.time_s
        forward_error = sample.body_forward_speed_mps - sample.command_forward_mps
        yaw_error = sample.body_yaw_rate_rps - sample.command_yaw_rps
        checks = {
            'tilt': (sample.tilt_deg, self.thresholds.tilt_deg),
            'angular_speed': (sample.angular_speed_rps, self.thresholds.angular_speed_rps),
            'forward_error': (abs(forward_error), self.thresholds.forward_error_mps),
            'yaw_error': (abs(yaw_error), self.thresholds.yaw_error_rps),
        }
        for reason, (value, limit) in checks.items():
            if value > limit:
                onset = self._pending.setdefault(reason, sample.time_s)
                if reason not in self._active and sample.time_s - onset >= self.thresholds.dwell_s - 1e-12:
                    self._active.add(reason)
                    self.events.append(dict(event='candidate_enter', reason=reason,
                                            onset_time_s=onset, time_s=sample.time_s,
                                            value=value, threshold=limit))
            else:
                self._pending.pop(reason, None)
                if reason in self._active:
                    self._active.remove(reason)
                    self.events.append(dict(event='candidate_clear', reason=reason,
                                            time_s=sample.time_s))
        return dict(**asdict(sample), forward_error_mps=forward_error,
                    yaw_error_rps=yaw_error, candidate_reasons=sorted(self._active))
