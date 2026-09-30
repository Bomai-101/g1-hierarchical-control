import numpy as np

from g1_control.config import balance as balance_config
from g1_control.envs.recovery_env import compute_recovery_reward_terms
from g1_control.evaluation.recovery_protocol import RecoveryState


def state(pitch=0.0, gyro=0.0, pose=0.0, joint_velocity=0.0, height=0.80):
    return RecoveryState(
        roll=0.0,
        pitch=pitch,
        gyro_norm=gyro,
        height=height,
        pose_error_rms=pose,
        selected_joint_velocity_rms=joint_velocity,
    )


def test_equilibrium_scores_above_unstable_state():
    action = np.zeros(balance_config.ACTION_DIM, dtype=np.float32)
    equilibrium = compute_recovery_reward_terms(state(), action, reference_height=0.80)
    unstable = compute_recovery_reward_terms(
        state(pitch=0.20, gyro=1.0, pose=0.15, joint_velocity=1.0, height=0.65),
        action,
        reference_height=0.80,
    )
    assert equilibrium["total"] > unstable["total"]
    assert equilibrium["equilibrium_bonus"] == 1.0
    assert unstable["equilibrium_bonus"] == 0.0


def test_action_cost_reduces_reward():
    zero_action = np.zeros(balance_config.ACTION_DIM, dtype=np.float32)
    full_action = np.ones(balance_config.ACTION_DIM, dtype=np.float32)
    zero = compute_recovery_reward_terms(state(), zero_action, reference_height=0.80)
    full = compute_recovery_reward_terms(state(), full_action, reference_height=0.80)
    assert zero["total"] > full["total"]


if __name__ == "__main__":
    test_equilibrium_scores_above_unstable_state()
    test_action_cost_reduces_reward()
    print("recovery reward tests: PASS")
