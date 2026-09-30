"""Training-only initial-state randomization for the Day 9 Stage B run."""

import mujoco
import numpy as np

from g1_control.config import balance as config


def reset_training_env(env, rng):
    """Reset the environment and optionally apply a small relative perturbation."""
    obs = env.reset()

    if rng.random() < config.RESET_NOMINAL_PROBABILITY:
        return obs, {
            "mode": "nominal",
            "pitch_delta": 0.0,
            "pitch_rate_delta": 0.0,
        }

    pitch_delta = float(rng.uniform(*config.RESET_PITCH_RANGE))
    pitch_rate_delta = float(rng.uniform(*config.RESET_PITCH_RATE_RANGE))

    relative_pitch = np.array(
        [
            np.cos(0.5 * pitch_delta),
            0.0,
            np.sin(0.5 * pitch_delta),
            0.0,
        ],
        dtype=np.float64,
    )
    perturbed_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(
        perturbed_quaternion,
        env.data.qpos[3:7].copy(),
        relative_pitch,
    )
    env.data.qpos[3:7] = perturbed_quaternion
    env.data.qvel[4] += pitch_rate_delta

    mujoco.mj_forward(env.model, env.data)

    return env.get_observation(), {
        "mode": "randomized",
        "pitch_delta": pitch_delta,
        "pitch_rate_delta": pitch_rate_delta,
    }
