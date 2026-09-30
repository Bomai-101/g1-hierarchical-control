"""Interpretable recovery candidate around the frozen Day 9 PD baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_control.controllers.residual import (
    ControlMode,
    ControllerOutput,
    ResidualController,
    _make_output,
    _validate_observation,
)


@dataclass(frozen=True)
class RecoveryGains:
    """Pitch and pitch-rate feedback in normalized residual-action units."""

    hip_kp: float = 4.0
    hip_kd: float = 0.4
    knee_kp: float = 2.0
    knee_kd: float = 0.2
    ankle_kp: float = 0.0
    ankle_kd: float = 0.5


class RecoveryStandController(ResidualController):
    """Symmetric sagittal feedback measured from Day 10 authority tests.

    The output remains a 6D residual around the unchanged Day 9 fixed pose.
    Positive hip/knee feedback and pitch-rate ankle feedback are the restoring
    directions observed in the paired pulse experiment.
    """

    mode = ControlMode.RECOVERY_STAND

    def __init__(self, gains: RecoveryGains | None = None):
        self.gains = gains or RecoveryGains()

    def act(self, observation: np.ndarray) -> ControllerOutput:
        obs = _validate_observation(observation)
        pitch = float(obs[59])
        pitch_rate = float(obs[62])
        gains = self.gains

        hip = gains.hip_kp * pitch + gains.hip_kd * pitch_rate
        knee = gains.knee_kp * pitch + gains.knee_kd * pitch_rate
        ankle = gains.ankle_kp * pitch + gains.ankle_kd * pitch_rate

        action = np.array(
            [hip, knee, ankle, hip, knee, ankle],
            dtype=np.float32,
        )
        return _make_output(self.mode, action)
