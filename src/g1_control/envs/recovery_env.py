"""Day 10 environment with recovery-specific reward shaping."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from g1_control.config.recovery import (
    DEFAULT_RECOVERY_CRITERIA,
    DEFAULT_RECOVERY_REWARD_WEIGHTS,
    RECOVERY_EPISODE_STEPS,
    RecoveryCriteria,
    RecoveryRewardWeights,
)
from g1_control.envs.balance_env import G1Env
from g1_control.evaluation.recovery_protocol import (
    RecoveryState,
    inside_entry_envelope,
    measure_recovery_state,
)


def compute_recovery_reward_terms(
    state: RecoveryState,
    action: np.ndarray,
    reference_height: float,
    criteria: RecoveryCriteria = DEFAULT_RECOVERY_CRITERIA,
    weights: RecoveryRewardWeights = DEFAULT_RECOVERY_REWARD_WEIGHTS,
) -> dict[str, float]:
    """Compute interpretable reward terms for a measured full-body state."""

    bounded_action = np.asarray(action, dtype=np.float32)
    tilt_error = (
        (state.roll / criteria.entry_abs_roll) ** 2
        + (state.pitch / criteria.entry_abs_pitch) ** 2
    )
    tilt_quality = float(np.exp(-0.5 * tilt_error))
    gyro_quality = float(
        np.exp(-0.5 * (state.gyro_norm / criteria.entry_gyro_norm) ** 2)
    )
    pose_quality = float(
        np.exp(-0.5 * (state.pose_error_rms / criteria.entry_pose_error_rms) ** 2)
    )
    joint_velocity_quality = float(
        np.exp(
            -0.5
            * (
                state.selected_joint_velocity_rms
                / criteria.entry_selected_joint_velocity_rms
            )
            ** 2
        )
    )
    height_scale = max(reference_height - criteria.entry_min_height, 0.02)
    height_quality = float(
        np.exp(-0.5 * ((state.height - reference_height) / height_scale) ** 2)
    )
    equilibrium_bonus = float(inside_entry_envelope(state, criteria))
    action_cost = float(np.mean(bounded_action**2))

    total = (
        weights.alive
        + weights.tilt_quality * tilt_quality
        + weights.gyro_quality * gyro_quality
        + weights.pose_quality * pose_quality
        + weights.joint_velocity_quality * joint_velocity_quality
        + weights.height_quality * height_quality
        + weights.equilibrium_bonus * equilibrium_bonus
        - weights.action_penalty * action_cost
    )
    return {
        "alive": weights.alive,
        "tilt_quality": tilt_quality,
        "gyro_quality": gyro_quality,
        "pose_quality": pose_quality,
        "joint_velocity_quality": joint_velocity_quality,
        "height_quality": height_quality,
        "equilibrium_bonus": equilibrium_bonus,
        "action_cost": action_cost,
        "total": float(total),
    }


class RecoveryEnv(G1Env):
    """G1Env with the frozen plant/control path and a Day 10 reward only."""

    def __init__(
        self,
        criteria: RecoveryCriteria = DEFAULT_RECOVERY_CRITERIA,
        reward_weights: RecoveryRewardWeights = DEFAULT_RECOVERY_REWARD_WEIGHTS,
    ):
        self.recovery_criteria = criteria
        self.recovery_reward_weights = reward_weights
        self.last_reward_terms: dict[str, float] = {}
        super().__init__()
        self.max_episode_steps = RECOVERY_EPISODE_STEPS

    def compute_reward(self, action) -> float:
        observation = self.get_observation()
        state = measure_recovery_state(
            observation=observation,
            height=float(self.data.qpos[2]),
            default_q=self.default_q,
        )
        self.last_reward_terms = compute_recovery_reward_terms(
            state=state,
            action=np.asarray(action, dtype=np.float32),
            reference_height=self.reference_height,
            criteria=self.recovery_criteria,
            weights=self.recovery_reward_weights,
        )
        return self.last_reward_terms["total"]

    def recovery_config(self) -> dict[str, object]:
        return {
            "episode_steps": self.max_episode_steps,
            "criteria": asdict(self.recovery_criteria),
            "reward_weights": asdict(self.recovery_reward_weights),
        }
