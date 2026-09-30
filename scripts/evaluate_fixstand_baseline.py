"""Benchmark official Unitree FixStand targets in the existing G1Env.

This evaluator adapts only the documented 29-DoF q target and PD gains.  It
does not modify the frozen Day 9 environment or claim equivalence with the
full Unitree DDS/C++ deployment stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.envs.balance_env import G1Env


OFFICIAL_KP = np.array(
    [
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        200, 200, 200,
        40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40,
    ],
    dtype=np.float32,
)

OFFICIAL_KD = np.array(
    [
        2, 2, 2, 4, 2, 2,
        2, 2, 2, 4, 2, 2,
        5, 5, 5,
        10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10,
    ],
    dtype=np.float32,
)

OFFICIAL_LEGS_AND_WAIST = [
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    0.0, 0.0, 0.0,
]

RL_LAB_ARMS = [
    0.0, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,
    0.0, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,
]

MJLAB_ARMS = [
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
]


@dataclass(frozen=True)
class StandSpec:
    name: str
    target_q: np.ndarray | None
    kp: np.ndarray | None
    kd: np.ndarray | None


def termination_reason(env: G1Env, observation: np.ndarray, terminated: bool, truncated: bool) -> str:
    if truncated:
        return "timeout"
    if not terminated:
        return "max_test_steps"
    if float(env.data.qpos[2]) < 0.45:
        return "height"
    if abs(float(observation[58])) > 0.8:
        return "roll"
    if abs(float(observation[59])) > 0.8:
        return "pitch"
    return "unknown"


def run_spec(spec: StandSpec) -> dict[str, float | int | str | bool]:
    env = G1Env()
    if spec.target_q is not None:
        env.default_q[:] = spec.target_q
        env.kp[:] = spec.kp
        env.kd[:] = spec.kd

    observation = env.reset()
    zero_action = np.zeros(env.ACTION_DIM, dtype=np.float32)
    total_reward = 0.0
    min_height = float(env.data.qpos[2])
    max_abs_pitch = abs(float(observation[59]))
    max_abs_roll = abs(float(observation[58]))
    terminated = False
    truncated = False

    for step in range(env.max_episode_steps):
        observation, reward, terminated, truncated = env.step(zero_action)
        total_reward += reward
        min_height = min(min_height, float(env.data.qpos[2]))
        max_abs_roll = max(max_abs_roll, abs(float(observation[58])))
        max_abs_pitch = max(max_abs_pitch, abs(float(observation[59])))
        if terminated or truncated:
            break

    return {
        "name": spec.name,
        "length": step + 1,
        "return": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "reason": termination_reason(env, observation, terminated, truncated),
        "min_height": min_height,
        "max_abs_roll": max_abs_roll,
        "max_abs_pitch": max_abs_pitch,
        "final_height": float(env.data.qpos[2]),
    }


def main() -> None:
    rl_lab_q = np.array(OFFICIAL_LEGS_AND_WAIST + RL_LAB_ARMS, dtype=np.float32)
    mjlab_q = np.array(OFFICIAL_LEGS_AND_WAIST + MJLAB_ARMS, dtype=np.float32)
    specs = (
        StandSpec("day9_pd", None, None, None),
        StandSpec("unitree_rl_lab_fixstand", rl_lab_q, OFFICIAL_KP, OFFICIAL_KD),
        StandSpec("unitree_mjlab_fixstand", mjlab_q, OFFICIAL_KP, OFFICIAL_KD),
    )

    print("=== FIXSTAND BASELINE BENCHMARK ===")
    print("Same local G1 XML, zero residual action, and 1000-step termination limit.")
    print("Official 2/3-second entry interpolation is excluded; this measures steady-state hold.")
    print()

    for spec in specs:
        result = run_spec(spec)
        print(
            f"{result['name']:27} length={result['length']:4d} "
            f"return={result['return']:8.2f} reason={result['reason']:14} "
            f"min_h={result['min_height']:.3f} max_roll={result['max_abs_roll']:.3f} "
            f"max_pitch={result['max_abs_pitch']:.3f} final_h={result['final_height']:.3f}"
        )


if __name__ == "__main__":
    main()
