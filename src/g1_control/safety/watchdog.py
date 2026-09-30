"""Day 10 safety monitor for controller selection.

The watchdog observes state but never sends torque.  It requests a switch
before the frozen Day 9 environment reaches its fall-termination thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SafetyLevel(str, Enum):
    """Ordered watchdog outcomes."""

    NOMINAL = "nominal"
    WARNING = "warning"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class WatchdogThresholds:
    """Conservative, pre-termination thresholds for the existing G1Env."""

    warning_min_height: float = 0.65
    fallback_min_height: float = 0.55
    warning_abs_tilt: float = 0.35
    fallback_abs_tilt: float = 0.55
    warning_gyro_norm: float = 2.5
    fallback_gyro_norm: float = 4.0


@dataclass(frozen=True)
class SafetySnapshot:
    """Signals used by the watchdog at one policy step."""

    height: float
    roll: float
    pitch: float
    gyro_norm: float
    observation_finite: bool

    @classmethod
    def from_observation(
        cls,
        observation: np.ndarray,
        height: float,
    ) -> "SafetySnapshot":
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (64,):
            raise ValueError(f"Expected a 64D observation, got {obs.shape}")
        return cls(
            height=float(height),
            roll=float(obs[58]),
            pitch=float(obs[59]),
            gyro_norm=float(np.linalg.norm(obs[61:64])),
            observation_finite=bool(np.isfinite(obs).all() and np.isfinite(height)),
        )


@dataclass(frozen=True)
class WatchdogDecision:
    """A safety classification and its machine-readable reasons."""

    level: SafetyLevel
    reasons: tuple[str, ...]
    snapshot: SafetySnapshot


class Watchdog:
    """Classify state without modifying the environment or controller action."""

    def __init__(self, thresholds: WatchdogThresholds | None = None):
        self.thresholds = thresholds or WatchdogThresholds()

    def evaluate(self, snapshot: SafetySnapshot) -> WatchdogDecision:
        limits = self.thresholds
        if not snapshot.observation_finite:
            return WatchdogDecision(SafetyLevel.FALLBACK, ("non_finite_signal",), snapshot)

        tilt = max(abs(snapshot.roll), abs(snapshot.pitch))
        fallback_reasons: list[str] = []
        warning_reasons: list[str] = []

        self._add_reason(
            fallback_reasons, warning_reasons, "height",
            snapshot.height < limits.fallback_min_height,
            snapshot.height < limits.warning_min_height,
        )
        self._add_reason(
            fallback_reasons, warning_reasons, "tilt",
            tilt > limits.fallback_abs_tilt,
            tilt > limits.warning_abs_tilt,
        )
        self._add_reason(
            fallback_reasons, warning_reasons, "gyro",
            snapshot.gyro_norm > limits.fallback_gyro_norm,
            snapshot.gyro_norm > limits.warning_gyro_norm,
        )

        if fallback_reasons:
            return WatchdogDecision(SafetyLevel.FALLBACK, tuple(fallback_reasons), snapshot)
        if warning_reasons:
            return WatchdogDecision(SafetyLevel.WARNING, tuple(warning_reasons), snapshot)
        return WatchdogDecision(SafetyLevel.NOMINAL, (), snapshot)

    @staticmethod
    def _add_reason(
        fallback_reasons: list[str],
        warning_reasons: list[str],
        name: str,
        fallback: bool,
        warning: bool,
    ) -> None:
        if fallback:
            fallback_reasons.append(name)
        elif warning:
            warning_reasons.append(name)
