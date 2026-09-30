"""Run one Day 10 controller-interface smoke episode in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.envs.balance_env import G1Env
from g1_control.controllers.residual import (
    ControlMode,
    ControllerRegistry,
    PDStandController,
    PPOStandController,
)
from g1_control.controllers.recovery import RecoveryStandController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[mode.value for mode in ControlMode], default=ControlMode.PD_STAND.value)
    parser.add_argument("--checkpoint", type=Path, help="Required only for --mode ppo_stand.")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    mode = ControlMode(args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controllers = [PDStandController(), RecoveryStandController()]
    if args.checkpoint is not None:
        controllers.append(PPOStandController.from_checkpoint(args.checkpoint, device))
    if mode is ControlMode.PPO_STAND and args.checkpoint is None:
        parser.error("--checkpoint is required for --mode ppo_stand")

    registry = ControllerRegistry(controllers)
    env = G1Env()
    observation = env.reset()
    total_reward = 0.0
    max_action_norm = 0.0

    for step in range(args.steps):
        output = registry.act(mode, observation)
        observation, reward, terminated, truncated = env.step(output.action)
        total_reward += reward
        max_action_norm = max(max_action_norm, output.action_l2_norm)
        if terminated or truncated:
            break

    print(f"mode: {mode.value}")
    print(f"available modes: {[entry.value for entry in registry.available_modes]}")
    print(f"steps executed: {step + 1}")
    print(f"total reward: {total_reward:.2f}")
    print(f"max action L2 norm: {max_action_norm:.6f}")
    print(f"termination: {terminated}")
    print(f"truncation: {truncated}")
    print(f"final height: {float(env.data.qpos[2]):.3f}")
    print(f"final pitch: {float(observation[59]):.3f}")


if __name__ == "__main__":
    main()
