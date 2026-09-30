"""Reproducible reset sampling around the frozen Day 9 standing baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from g1_control.config.recovery import RecoveryCurriculumStage


@dataclass(frozen=True)
class RecoveryPerturbation:
    mode: str
    stage: str
    roll_delta: float
    pitch_delta: float
    roll_rate_delta: float
    pitch_rate_delta: float
    joint_indices: tuple[int, ...]
    joint_position_deltas: tuple[float, ...]
    joint_velocity_deltas: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def sample_recovery_perturbation(
    stage: RecoveryCurriculumStage,
    rng: np.random.Generator,
    policy_joint_indices: np.ndarray,
) -> RecoveryPerturbation:
    """Sample one perturbation without mutating an environment."""

    joint_pool = np.asarray(policy_joint_indices, dtype=np.int64)
    if joint_pool.ndim != 1 or joint_pool.size == 0:
        raise ValueError("policy_joint_indices must be a non-empty 1D array")
    if not 0.0 <= stage.nominal_probability <= 1.0:
        raise ValueError("nominal_probability must be inside [0, 1]")
    if not 1 <= stage.max_perturbed_joints <= joint_pool.size:
        raise ValueError("max_perturbed_joints must fit the policy joint set")

    if rng.random() < stage.nominal_probability:
        return RecoveryPerturbation(
            mode="nominal",
            stage=stage.name,
            roll_delta=0.0,
            pitch_delta=0.0,
            roll_rate_delta=0.0,
            pitch_rate_delta=0.0,
            joint_indices=(),
            joint_position_deltas=(),
            joint_velocity_deltas=(),
        )

    joint_count = int(rng.integers(1, stage.max_perturbed_joints + 1))
    chosen = np.sort(rng.choice(joint_pool, size=joint_count, replace=False))

    return RecoveryPerturbation(
        mode="randomized",
        stage=stage.name,
        roll_delta=_uniform(rng, stage.roll_range),
        pitch_delta=_uniform(rng, stage.pitch_range),
        roll_rate_delta=_uniform(rng, stage.roll_rate_range),
        pitch_rate_delta=_uniform(rng, stage.pitch_rate_range),
        joint_indices=tuple(int(index) for index in chosen),
        joint_position_deltas=tuple(
            _uniform(rng, stage.joint_position_delta_range)
            for _ in range(joint_count)
        ),
        joint_velocity_deltas=tuple(
            _uniform(rng, stage.joint_velocity_delta_range)
            for _ in range(joint_count)
        ),
    )


def apply_recovery_perturbation(env, perturbation: RecoveryPerturbation) -> np.ndarray:
    """Apply a sampled perturbation after the environment's PD settling."""

    if perturbation.mode == "nominal":
        return env.get_observation()

    half_roll = 0.5 * perturbation.roll_delta
    half_pitch = 0.5 * perturbation.pitch_delta
    roll_quat = np.array(
        [np.cos(half_roll), np.sin(half_roll), 0.0, 0.0],
        dtype=np.float64,
    )
    pitch_quat = np.array(
        [np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0],
        dtype=np.float64,
    )
    relative_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(relative_quat, roll_quat, pitch_quat)
    perturbed_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(perturbed_quat, env.data.qpos[3:7].copy(), relative_quat)
    env.data.qpos[3:7] = perturbed_quat

    env.data.qvel[3] += perturbation.roll_rate_delta
    env.data.qvel[4] += perturbation.pitch_rate_delta

    for joint_index, q_delta, dq_delta in zip(
        perturbation.joint_indices,
        perturbation.joint_position_deltas,
        perturbation.joint_velocity_deltas,
        strict=True,
    ):
        env.data.qpos[7 + joint_index] += q_delta
        env.data.qvel[6 + joint_index] += dq_delta

    mujoco.mj_forward(env.model, env.data)
    observation = env.get_observation()
    if not np.isfinite(observation).all():
        raise RuntimeError("Recovery reset produced a non-finite observation")
    return observation


def reset_recovery_env(
    env,
    rng: np.random.Generator,
    stage: RecoveryCurriculumStage,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reset, settle on frozen PD, then apply one curriculum perturbation."""

    env.reset()
    perturbation = sample_recovery_perturbation(
        stage=stage,
        rng=rng,
        policy_joint_indices=env.POLICY_JOINT_INDICES,
    )
    observation = apply_recovery_perturbation(env, perturbation)
    return observation, perturbation.to_dict()
