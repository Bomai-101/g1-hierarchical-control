"""Run Day 10 watchdog selection over a MuJoCo standing episode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.controllers.residual import (
    ControlMode,
    ControllerRegistry,
    PDStandController,
    PPOStandController,
)
from g1_control.controllers.recovery import RecoveryStandController
from g1_control.envs.balance_env import G1Env
from g1_control.safety import SafetySupervisor


def inject_pitch(env: G1Env, pitch: float) -> None:
    """Apply an explicit test-only base pitch disturbance before one decision."""
    env.data.qpos[3:7] = np.array(
        [np.cos(0.5 * pitch), 0.0, np.sin(0.5 * pitch), 0.0],
        dtype=np.float64,
    )
    mujoco.mj_forward(env.model, env.data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[mode.value for mode in ControlMode], default=ControlMode.PD_STAND.value)
    parser.add_argument("--checkpoint", type=Path, help="Required only for --mode ppo_stand.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inject-step", type=int, help="Optional test-only pitch injection step.")
    parser.add_argument("--inject-pitch", type=float, default=0.60)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.inject_step is not None and not 0 <= args.inject_step < args.steps:
        raise ValueError("--inject-step must be inside [0, --steps)")

    requested_mode = ControlMode(args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controllers = [PDStandController(), RecoveryStandController()]
    if args.checkpoint is not None:
        controllers.append(PPOStandController.from_checkpoint(args.checkpoint, device))
    if requested_mode is ControlMode.PPO_STAND and args.checkpoint is None:
        parser.error("--checkpoint is required for --mode ppo_stand")

    env = G1Env()
    supervisor = SafetySupervisor(ControllerRegistry(controllers), requested_mode)
    observation = env.reset()
    total_reward = 0.0
    warning_steps = 0
    final_decision = None

    for step in range(args.steps):
        if step == args.inject_step:
            inject_pitch(env, args.inject_pitch)
            observation = env.get_observation()
            print(f"injected pitch at step {step}: {args.inject_pitch:+.3f} rad")

        output, decision = supervisor.act(observation, float(env.data.qpos[2]), step)
        final_decision = decision
        warning_steps += decision.level.value == "warning"
        observation, reward, terminated, truncated = env.step(output.action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"requested mode: {requested_mode.value}")
    print(f"final selected mode: {supervisor.selected_mode.value}")
    print(f"fallback latched: {supervisor.fallback_latched}")
    print(f"steps executed: {step + 1}")
    print(f"total reward: {total_reward:.2f}")
    print(f"warning steps: {warning_steps}")
    print(f"last watchdog level: {final_decision.level.value}")
    print(f"last watchdog reasons: {list(final_decision.reasons)}")
    print(f"safety events: {len(supervisor.events)}")
    for event in supervisor.events:
        print(f"event step={event.step} {event.previous_mode.value}->{event.selected_mode.value} reasons={list(event.reasons)}")
    print(f"termination: {terminated}")
    print(f"truncation: {truncated}")


if __name__ == "__main__":
    main()
