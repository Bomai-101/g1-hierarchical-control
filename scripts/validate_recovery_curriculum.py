"""Validate Day 10 recovery resets before connecting them to PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.config import balance as balance_config
from g1_control.config.recovery import DEFAULT_RECOVERY_STAGE, get_recovery_stage
from g1_control.envs.balance_env import G1Env
from g1_control.training.recovery_curriculum import reset_recovery_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_RECOVERY_STAGE)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=balance_config.SEED)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    stage = get_recovery_stage(args.stage)
    env = G1Env()
    rng = np.random.default_rng(args.seed)
    modes = {"nominal": 0, "randomized": 0}
    invalid = 0
    immediately_terminated = 0
    observed: dict[str, list[float]] = {
        "roll_delta": [],
        "pitch_delta": [],
        "roll_rate_delta": [],
        "pitch_rate_delta": [],
        "joint_position_delta": [],
        "joint_velocity_delta": [],
        "joint_count": [],
    }

    for _ in range(args.samples):
        observation, info = reset_recovery_env(env, rng, stage)
        modes[str(info["mode"])] += 1
        invalid += int(not np.isfinite(observation).all())
        terminated, _ = env.check_termination()
        immediately_terminated += int(terminated)

        if info["mode"] == "randomized":
            for key in ("roll_delta", "pitch_delta", "roll_rate_delta", "pitch_rate_delta"):
                observed[key].append(float(info[key]))
            observed["joint_position_delta"].extend(info["joint_position_deltas"])
            observed["joint_velocity_delta"].extend(info["joint_velocity_deltas"])
            observed["joint_count"].append(float(len(info["joint_indices"])))

    print("=== DAY 10 RECOVERY RESET VALIDATION ===")
    print(f"stage: {stage.name}")
    print(f"seed: {args.seed}")
    print(f"samples: {args.samples}")
    print(f"nominal/randomized: {modes['nominal']}/{modes['randomized']}")
    print(f"nonfinite observations: {invalid}")
    print(f"immediate terminations: {immediately_terminated}")
    for key, values in observed.items():
        if values:
            print(f"{key}: min={min(values):+.6f} max={max(values):+.6f}")

    if invalid or immediately_terminated:
        raise SystemExit("Reset validation failed")
    print("validation: PASS")


if __name__ == "__main__":
    main()
