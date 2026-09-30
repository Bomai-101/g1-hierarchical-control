"""Fixed, transparent cases for the Day 10 Stage 1 checkpoint gate."""

from __future__ import annotations

from dataclasses import dataclass

from g1_control.config import balance as balance_config
from g1_control.training.recovery_curriculum import RecoveryPerturbation


@dataclass(frozen=True)
class FixedRecoveryCase:
    name: str
    category: str
    perturbation: RecoveryPerturbation


def _perturbation(
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    roll_rate: float = 0.0,
    pitch_rate: float = 0.0,
    joint_index: int | None = None,
    joint_position: float = 0.0,
    joint_velocity: float = 0.0,
) -> RecoveryPerturbation:
    has_joint = joint_index is not None
    return RecoveryPerturbation(
        mode="randomized",
        stage="fixed_stage_1_evaluation",
        roll_delta=roll,
        pitch_delta=pitch,
        roll_rate_delta=roll_rate,
        pitch_rate_delta=pitch_rate,
        joint_indices=(joint_index,) if has_joint else (),
        joint_position_deltas=(joint_position,) if has_joint else (),
        joint_velocity_deltas=(joint_velocity,) if has_joint else (),
    )


def fixed_stage_1_cases() -> tuple[FixedRecoveryCase, ...]:
    """Return in-domain gate cases plus clearly marked stress cases."""

    cases: list[FixedRecoveryCase] = [
        FixedRecoveryCase(
            "nominal",
            "in_domain",
            RecoveryPerturbation(
                mode="nominal",
                stage="fixed_stage_1_evaluation",
                roll_delta=0.0,
                pitch_delta=0.0,
                roll_rate_delta=0.0,
                pitch_rate_delta=0.0,
                joint_indices=(),
                joint_position_deltas=(),
                joint_velocity_deltas=(),
            ),
        )
    ]

    for value in (-0.01, 0.01):
        cases.append(FixedRecoveryCase(f"roll_{value:+.2f}", "in_domain", _perturbation(roll=value)))
    for value in (-0.01, 0.01, -0.02, 0.02):
        cases.append(FixedRecoveryCase(f"pitch_{value:+.2f}", "in_domain", _perturbation(pitch=value)))
    for value in (-0.05, 0.05):
        cases.append(FixedRecoveryCase(f"roll_rate_{value:+.2f}", "in_domain", _perturbation(roll_rate=value)))
    for value in (-0.05, 0.05, -0.10, 0.10):
        cases.append(FixedRecoveryCase(f"pitch_rate_{value:+.2f}", "in_domain", _perturbation(pitch_rate=value)))

    joint_labels = ("l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle")
    for label, joint_index in zip(
        joint_labels,
        balance_config.POLICY_JOINT_INDICES,
        strict=True,
    ):
        for value in (-0.02, 0.02):
            cases.append(
                FixedRecoveryCase(
                    f"{label}_q_{value:+.2f}",
                    "in_domain",
                    _perturbation(joint_index=joint_index, joint_position=value),
                )
            )
        for value in (-0.10, 0.10):
            cases.append(
                FixedRecoveryCase(
                    f"{label}_dq_{value:+.2f}",
                    "in_domain",
                    _perturbation(joint_index=joint_index, joint_velocity=value),
                )
            )

    for value in (-0.05, 0.05, -0.10, 0.10):
        cases.append(
            FixedRecoveryCase(
                f"stress_pitch_{value:+.2f}",
                "stress",
                _perturbation(pitch=value),
            )
        )
    return tuple(cases)
