"""Day 10 recovery curriculum and evaluation configuration.

The Day 9 balance configuration remains frozen.  This module only describes
how Day 10 samples states around that fixed standing equilibrium and how a
recovery attempt is judged.
"""

from __future__ import annotations

from dataclasses import dataclass

from g1_control.config import balance as balance_config


# Day 10 artifacts live under the already local-only checkpoint tree, but in
# their own directory so they cannot be confused with Day 9 checkpoints.
RECOVERY_RUNS_DIR = balance_config.CHECKPOINT_DIR / "day10_recovery_runs"
RECOVERY_EPISODE_STEPS = 150
RECOVERY_ROLLOUT_STEPS = balance_config.ROLLOUT_STEPS
RECOVERY_NUM_UPDATES = balance_config.NUM_UPDATES
# Recovery requires wider exploration than the frozen Day 9 nominal policy.
# This is local to train_recovery.py; Actor's default remains Day 9 -3.0.
RECOVERY_ACTOR_LOG_STD = -2.0


@dataclass(frozen=True)
class RecoveryCurriculumStage:
    """Bounds for one initial-state recovery curriculum stage."""

    name: str
    nominal_probability: float
    roll_range: tuple[float, float]
    pitch_range: tuple[float, float]
    roll_rate_range: tuple[float, float]
    pitch_rate_range: tuple[float, float]
    joint_position_delta_range: tuple[float, float]
    joint_velocity_delta_range: tuple[float, float]
    max_perturbed_joints: int


# Stage 1 is deliberately smaller than the disturbances that defeated the
# Day 9 policy.  Later stages are declared now for provenance, but training
# must not advance to them until the preceding fixed evaluation set passes.
RECOVERY_CURRICULUM_STAGES = (
    RecoveryCurriculumStage(
        name="stage_1_small",
        nominal_probability=0.20,
        roll_range=(-0.01, 0.01),
        pitch_range=(-0.02, 0.02),
        roll_rate_range=(-0.05, 0.05),
        pitch_rate_range=(-0.10, 0.10),
        joint_position_delta_range=(-0.02, 0.02),
        joint_velocity_delta_range=(-0.10, 0.10),
        max_perturbed_joints=1,
    ),
    RecoveryCurriculumStage(
        name="stage_2_multi_joint",
        nominal_probability=0.20,
        roll_range=(-0.02, 0.02),
        pitch_range=(-0.05, 0.05),
        roll_rate_range=(-0.10, 0.10),
        pitch_rate_range=(-0.25, 0.25),
        joint_position_delta_range=(-0.04, 0.04),
        joint_velocity_delta_range=(-0.20, 0.20),
        max_perturbed_joints=2,
    ),
    RecoveryCurriculumStage(
        name="stage_3_local_recovery",
        nominal_probability=0.15,
        roll_range=(-0.04, 0.04),
        pitch_range=(-0.10, 0.10),
        roll_rate_range=(-0.20, 0.20),
        pitch_rate_range=(-0.50, 0.50),
        joint_position_delta_range=(-0.08, 0.08),
        joint_velocity_delta_range=(-0.40, 0.40),
        max_perturbed_joints=4,
    ),
)

DEFAULT_RECOVERY_STAGE = "stage_1_small"


def get_recovery_stage(name: str) -> RecoveryCurriculumStage:
    for stage in RECOVERY_CURRICULUM_STAGES:
        if stage.name == name:
            return stage
    valid = ", ".join(stage.name for stage in RECOVERY_CURRICULUM_STAGES)
    raise ValueError(f"Unknown recovery stage {name!r}; expected one of: {valid}")


@dataclass(frozen=True)
class RecoveryCriteria:
    """Two-envelope recovery definition.

    The entry envelope detects that the trajectory genuinely returned near
    the fixed standing state.  The wider hold envelope then detects relapse.
    A run is successful only when it is alive and finishes with a continuous
    hold window, so briefly crossing upright is not labelled as recovery.
    """

    entry_abs_roll: float = 0.05
    entry_abs_pitch: float = 0.05
    entry_gyro_norm: float = 0.25
    entry_min_height: float = 0.72
    entry_pose_error_rms: float = 0.08
    entry_selected_joint_velocity_rms: float = 0.40
    entry_steps: int = 10

    hold_abs_roll: float = 0.08
    hold_abs_pitch: float = 0.08
    hold_gyro_norm: float = 0.50
    hold_min_height: float = 0.70
    hold_pose_error_rms: float = 0.12
    hold_selected_joint_velocity_rms: float = 0.75
    final_hold_steps: int = 25


DEFAULT_RECOVERY_CRITERIA = RecoveryCriteria()


@dataclass(frozen=True)
class RecoveryRewardWeights:
    """Weights for equilibrium recovery rather than simple fall delay."""

    alive: float = 0.20
    tilt_quality: float = 1.25
    gyro_quality: float = 0.75
    pose_quality: float = 0.50
    joint_velocity_quality: float = 0.35
    height_quality: float = 0.35
    equilibrium_bonus: float = 0.75
    action_penalty: float = 0.02


DEFAULT_RECOVERY_REWARD_WEIGHTS = RecoveryRewardWeights()

# Deterministic cases are evaluation-only and must not be sampled from the
# training RNG.  This prevents the benchmark from silently changing as the
# curriculum advances.
FIXED_EVALUATION_PITCHES = (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)
