"""Measure short-horizon pitch authority of each symmetric 6D action group.

This is system identification, not PPO training.  Every case uses the frozen
Day 9 action scale and is paired with a zero-action run from the same injected
base pitch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.config import balance as config
from g1_control.envs.balance_env import G1Env


SAGITTAL_GROUPS = {
    "hip_pair": (0, 3),
    "knee_pair": (1, 4),
    "ankle_pair": (2, 5),
}


@dataclass(frozen=True)
class Response:
    pitch: float
    pitch_rate: float
    height: float
    terminated: bool


def inject_pitch(env: G1Env, pitch: float) -> np.ndarray:
    """Set one controlled test-only base pitch before a short pulse."""
    env.data.qpos[3:7] = np.array(
        [np.cos(0.5 * pitch), 0.0, np.sin(0.5 * pitch), 0.0],
        dtype=np.float64,
    )
    mujoco.mj_forward(env.model, env.data)
    return env.get_observation()


def run_response(
    initial_pitch: float,
    action: np.ndarray,
    pulse_steps: int,
) -> Response:
    env = G1Env()
    observation = env.reset()
    observation = inject_pitch(env, initial_pitch)
    terminated = False

    for _ in range(pulse_steps):
        observation, _, terminated, truncated = env.step(action)
        if terminated or truncated:
            break

    return Response(
        pitch=float(observation[59]),
        pitch_rate=float(observation[62]),
        height=float(env.data.qpos[2]),
        terminated=terminated,
    )


def make_action(indices: tuple[int, int], amplitude: float) -> np.ndarray:
    action = np.zeros(config.ACTION_DIM, dtype=np.float32)
    action[list(indices)] = amplitude
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=0.50)
    parser.add_argument("--pulse-steps", type=int, default=10)
    parser.add_argument("--pitch-values", type=float, nargs="+", default=(-0.10, 0.10))
    args = parser.parse_args()

    if not 0.0 < args.amplitude <= 1.0:
        raise ValueError("--amplitude must be in (0, 1]")
    if args.pulse_steps <= 0:
        raise ValueError("--pulse-steps must be positive")

    print("=== DAY 10 ACTION-AUTHORITY CALIBRATION ===")
    print(f"Frozen action scale: {config.ACTION_SCALE:.2f}")
    print(f"Pulse: {args.pulse_steps} policy steps; normalized amplitude: {args.amplitude:.2f}")
    print("Positive restoring score means the action moves pitch toward zero versus zero action.")
    print()

    for initial_pitch in args.pitch_values:
        zero = run_response(
            initial_pitch,
            np.zeros(config.ACTION_DIM, dtype=np.float32),
            args.pulse_steps,
        )
        direction = 1.0 if initial_pitch > 0.0 else -1.0
        print(
            f"Initial pitch={initial_pitch:+.3f} | zero pitch={zero.pitch:+.4f} "
            f"rate={zero.pitch_rate:+.4f} height={zero.height:.3f}"
        )

        for name, indices in SAGITTAL_GROUPS.items():
            for sign in (-1.0, 1.0):
                response = run_response(
                    initial_pitch,
                    make_action(indices, sign * args.amplitude),
                    args.pulse_steps,
                )
                pitch_delta = response.pitch - zero.pitch
                rate_delta = response.pitch_rate - zero.pitch_rate
                restoring_score = -direction * pitch_delta
                restoring_rate_score = -direction * rate_delta
                outcome = "terminated" if response.terminated else "active"
                print(
                    f"  {name:11} action={sign * args.amplitude:+.2f} "
                    f"d_pitch={pitch_delta:+.5f} d_rate={rate_delta:+.5f} "
                    f"restore_pitch={restoring_score:+.5f} "
                    f"restore_rate={restoring_rate_score:+.5f} {outcome}"
                )
        print()


if __name__ == "__main__":
    main()
