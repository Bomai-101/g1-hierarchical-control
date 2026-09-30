"""Measure Day 10 watchdog and fallback behavior across pitch disturbances."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from g1_control.envs.balance_env import G1Env
from g1_control.safety import SafetySnapshot, SafetySupervisor, Watchdog


DEFAULT_PITCH_VALUES = (-0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class CaseResult:
    label: str
    injected_pitch: float
    episode_steps: int
    terminated: bool
    warning_steps: int
    unsafe_step: int | None
    switch_step: int | None
    unsafe_reasons: tuple[str, ...]
    mean_action_norm: float
    max_action_norm: float


def inject_pitch(env: G1Env, pitch: float) -> None:
    env.data.qpos[3:7] = np.array(
        [np.cos(0.5 * pitch), 0.0, np.sin(0.5 * pitch), 0.0],
        dtype=np.float64,
    )
    mujoco.mj_forward(env.model, env.data)


def run_case(
    registry: ControllerRegistry,
    mode: ControlMode,
    label: str,
    watchdog_enabled: bool,
    pitch: float,
    inject_step: int,
    max_steps: int,
) -> CaseResult:
    env = G1Env()
    supervisor = SafetySupervisor(registry, mode)
    watchdog = Watchdog()
    observation = env.reset()
    warning_steps = 0
    terminated = False
    unsafe_step: int | None = None
    unsafe_reasons: tuple[str, ...] = ()
    action_norms: list[float] = []

    for step in range(max_steps):
        if step == inject_step:
            inject_pitch(env, pitch)
            observation = env.get_observation()

        if watchdog_enabled or mode is ControlMode.PD_STAND:
            output, decision = supervisor.act(observation, float(env.data.qpos[2]), step)
        else:
            snapshot = SafetySnapshot.from_observation(observation, float(env.data.qpos[2]))
            decision = watchdog.evaluate(snapshot)
            output = registry.act(mode, observation)
        warning_steps += decision.level.value == "warning"
        if decision.level.value == "fallback" and unsafe_step is None:
            unsafe_step = step
            unsafe_reasons = decision.reasons
        action_norms.append(output.action_l2_norm)
        observation, _, terminated, truncated = env.step(output.action)
        if terminated or truncated:
            break

    event = supervisor.events[0] if supervisor.events else None
    return CaseResult(
        label=label,
        injected_pitch=pitch,
        episode_steps=step + 1,
        terminated=terminated,
        warning_steps=warning_steps,
        unsafe_step=unsafe_step,
        switch_step=event.step if event else None,
        unsafe_reasons=unsafe_reasons,
        mean_action_norm=float(np.mean(action_norms)),
        max_action_norm=float(np.max(action_norms)),
    )


def print_result(result: CaseResult) -> None:
    unsafe = "--" if result.unsafe_step is None else str(result.unsafe_step)
    switch = "--" if result.switch_step is None else str(result.switch_step)
    reasons = ",".join(result.unsafe_reasons) or "--"
    outcome = "fall" if result.terminated else "survived"
    print(
        f"{result.label:15} pitch={result.injected_pitch:+.2f} "
        f"steps={result.episode_steps:3d} outcome={outcome:8} "
        f"warnings={result.warning_steps:2d} unsafe_at={unsafe:>2} switch_at={switch:>2} "
        f"action_mean={result.mean_action_norm:.4f} action_max={result.max_action_norm:.4f} reasons={reasons}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inject-step", type=int, default=20)
    parser.add_argument("--pitch-values", type=float, nargs="+", default=DEFAULT_PITCH_VALUES)
    args = parser.parse_args()

    if args.steps <= 0 or not 0 <= args.inject_step < args.steps:
        raise ValueError("--inject-step must be inside [0, --steps)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = ControllerRegistry(
        [
            PDStandController(),
            PPOStandController.from_checkpoint(args.checkpoint, device),
        ]
    )

    print("=== DAY 10 WATCHDOG BOUNDARY SWEEP ===")
    print(f"Injection step: {args.inject_step}; maximum steps: {args.steps}")
    print("A fallback is a watchdog intervention, not proof of recovery.")
    print("PPO is evaluated both unprotected and with watchdog fallback.")
    print()

    results = []
    variants = (
        ("pd_stand", ControlMode.PD_STAND, False),
        ("ppo_unprotected", ControlMode.PPO_STAND, False),
        ("ppo_watchdog", ControlMode.PPO_STAND, True),
    )
    for pitch in args.pitch_values:
        pitch_results: list[CaseResult] = []
        for label, mode, watchdog_enabled in variants:
            result = run_case(
                registry, mode, label, watchdog_enabled,
                pitch, args.inject_step, args.steps,
            )
            results.append(result)
            pitch_results.append(result)
            print_result(result)
        baseline = pitch_results[0].episode_steps
        print(
            f"  delta vs PD: unprotected={pitch_results[1].episode_steps - baseline:+d}, "
            f"watchdog={pitch_results[2].episode_steps - baseline:+d}"
        )

    ppo_interventions = sum(
        result.label == "ppo_watchdog" and result.switch_step is not None
        for result in results
    )
    ppo_survivals = sum(
        result.label == "ppo_watchdog" and not result.terminated
        for result in results
    )
    ppo_unprotected_survivals = sum(
        result.label == "ppo_unprotected" and not result.terminated
        for result in results
    )
    print()
    print("=== SUMMARY ===")
    print(f"PPO watchdog interventions: {ppo_interventions}/{len(args.pitch_values)}")
    print(f"PPO survivors at {args.steps} steps: {ppo_survivals}/{len(args.pitch_values)}")
    print(f"Unprotected PPO survivors at {args.steps} steps: {ppo_unprotected_survivals}/{len(args.pitch_values)}")


if __name__ == "__main__":
    main()
