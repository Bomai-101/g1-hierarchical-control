"""Strict recovery metrics shared by Day 10 evaluation and later training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_control.config import balance as balance_config
from g1_control.config.recovery import RecoveryCriteria


@dataclass(frozen=True)
class RecoveryState:
    roll: float
    pitch: float
    gyro_norm: float
    height: float
    pose_error_rms: float
    selected_joint_velocity_rms: float


def measure_recovery_state(
    observation: np.ndarray,
    height: float,
    default_q: np.ndarray,
) -> RecoveryState:
    obs = np.asarray(observation, dtype=np.float32)
    reference = np.asarray(default_q, dtype=np.float32)
    if obs.shape != (balance_config.OBS_DIM,):
        raise ValueError(f"Expected observation shape {(balance_config.OBS_DIM,)}, got {obs.shape}")
    if reference.shape != (balance_config.NUM_JOINTS,):
        raise ValueError(
            f"Expected default_q shape {(balance_config.NUM_JOINTS,)}, got {reference.shape}"
        )

    q = obs[: balance_config.NUM_JOINTS]
    dq = obs[balance_config.NUM_JOINTS : 2 * balance_config.NUM_JOINTS]
    selected_dq = dq[np.asarray(balance_config.POLICY_JOINT_INDICES)]

    return RecoveryState(
        roll=float(obs[58]),
        pitch=float(obs[59]),
        gyro_norm=float(np.linalg.norm(obs[61:64])),
        height=float(height),
        pose_error_rms=float(np.sqrt(np.mean((q - reference) ** 2))),
        selected_joint_velocity_rms=float(np.sqrt(np.mean(selected_dq**2))),
    )


def inside_entry_envelope(state: RecoveryState, criteria: RecoveryCriteria) -> bool:
    return (
        abs(state.roll) <= criteria.entry_abs_roll
        and abs(state.pitch) <= criteria.entry_abs_pitch
        and state.gyro_norm <= criteria.entry_gyro_norm
        and state.height >= criteria.entry_min_height
        and state.pose_error_rms <= criteria.entry_pose_error_rms
        and state.selected_joint_velocity_rms
        <= criteria.entry_selected_joint_velocity_rms
    )


def inside_hold_envelope(state: RecoveryState, criteria: RecoveryCriteria) -> bool:
    return (
        abs(state.roll) <= criteria.hold_abs_roll
        and abs(state.pitch) <= criteria.hold_abs_pitch
        and state.gyro_norm <= criteria.hold_gyro_norm
        and state.height >= criteria.hold_min_height
        and state.pose_error_rms <= criteria.hold_pose_error_rms
        and state.selected_joint_velocity_rms
        <= criteria.hold_selected_joint_velocity_rms
    )


class RecoveryTracker:
    """Track entry, relapse, and sustained final recovery."""

    def __init__(self, criteria: RecoveryCriteria):
        self.criteria = criteria
        self.entry_count = 0
        self.final_hold_count = 0
        self.total_steps = 0
        self.entry_envelope_steps = 0
        self.hold_envelope_steps = 0
        self.entered_at: int | None = None
        self.relapsed = False
        self.last_state: RecoveryState | None = None

    def update(self, step: int, state: RecoveryState) -> None:
        self.last_state = state
        self.total_steps += 1

        in_entry = inside_entry_envelope(state, self.criteria)
        in_hold = inside_hold_envelope(state, self.criteria)
        self.entry_envelope_steps += int(in_entry)
        self.hold_envelope_steps += int(in_hold)

        if in_entry:
            self.entry_count += 1
        else:
            self.entry_count = 0

        if (
            self.entered_at is None
            and self.entry_count >= self.criteria.entry_steps
        ):
            self.entered_at = step - self.criteria.entry_steps + 1

        if in_hold:
            self.final_hold_count += 1
        else:
            self.final_hold_count = 0
            if self.entered_at is not None:
                self.relapsed = True

    def succeeded(self, survived: bool) -> bool:
        return (
            survived
            and self.entered_at is not None
            and self.final_hold_count >= self.criteria.final_hold_steps
        )

    @property
    def hold_fraction(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.hold_envelope_steps / self.total_steps
