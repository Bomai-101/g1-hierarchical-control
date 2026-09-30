"""Compare recovery candidates under the strict Day 10 success protocol."""

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

from g1_control.config.recovery import (
    DEFAULT_RECOVERY_CRITERIA,
    FIXED_EVALUATION_PITCHES,
)
from g1_control.controllers.recovery import RecoveryGains, RecoveryStandController
from g1_control.controllers.residual import PDStandController, ResidualController
from g1_control.envs.balance_env import G1Env
from g1_control.evaluation.recovery_protocol import (
    RecoveryTracker,
    measure_recovery_state,
)


@dataclass(frozen=True)
class Candidate:
    label: str
    controller: ResidualController
    handoff_to_pd: bool = False


@dataclass(frozen=True)
class Result:
    label: str
    pitch: float
    steps: int
    terminated: bool
    entered_at: int | None
    handoff_at: int | None
    relapsed: bool
    success: bool
    final_pitch: float
    final_rate: float
    final_gyro_norm: float
    final_pose_error_rms: float
    final_selected_dq_rms: float
    min_height: float
    mean_action_norm: float
    max_action_norm: float
    saturation_fraction: float


def inject_pitch(env: G1Env, pitch: float) -> np.ndarray:
    """Set one deterministic evaluation disturbance after nominal settling."""

    env.data.qpos[3:7] = np.array(
        [np.cos(0.5 * pitch), 0.0, np.sin(0.5 * pitch), 0.0],
        dtype=np.float64,
    )
    mujoco.mj_forward(env.model, env.data)
    return env.get_observation()


def run_case(
    candidate: Candidate,
    pitch: float,
    inject_step: int,
    max_steps: int,
) -> Result:
    env = G1Env()
    observation = env.reset()
    min_height = float(env.data.qpos[2])
    action_norms: list[float] = []
    saturated_steps = 0
    handoff_at: int | None = None
    terminated = False

    # Keep the fixed Day 9 PD baseline active before the disturbance.  This
    # prevents a recovery candidate from changing the pre-injection state.
    pd_controller = PDStandController()
    active_controller: ResidualController = pd_controller
    tracker = RecoveryTracker(DEFAULT_RECOVERY_CRITERIA)

    for step in range(max_steps):
        if step == inject_step:
            observation = inject_pitch(env, pitch)
            active_controller = candidate.controller

        output = active_controller.act(observation)
        if step >= inject_step:
            action_norms.append(output.action_l2_norm)
            saturated_steps += int(np.any(np.abs(output.action) >= 0.999))

        observation, _, terminated, truncated = env.step(output.action)
        height = float(env.data.qpos[2])
        min_height = min(min_height, height)

        if step >= inject_step:
            state = measure_recovery_state(
                observation=observation,
                height=height,
                default_q=env.default_q,
            )
            previous_entered_at = tracker.entered_at
            tracker.update(step=step, state=state)
            if (
                candidate.handoff_to_pd
                and previous_entered_at is None
                and tracker.entered_at is not None
            ):
                active_controller = pd_controller
                handoff_at = step

        if terminated or truncated:
            break

    survived = not terminated and not truncated
    success = tracker.succeeded(survived=survived)
    final_state = tracker.last_state
    if final_state is None:
        raise RuntimeError("Evaluation ended before the disturbance was observed")

    return Result(
        label=candidate.label,
        pitch=pitch,
        steps=step + 1,
        terminated=terminated,
        entered_at=tracker.entered_at,
        handoff_at=handoff_at,
        relapsed=tracker.relapsed,
        success=success,
        final_pitch=final_state.pitch,
        final_rate=float(observation[62]),
        final_gyro_norm=final_state.gyro_norm,
        final_pose_error_rms=final_state.pose_error_rms,
        final_selected_dq_rms=final_state.selected_joint_velocity_rms,
        min_height=min_height,
        mean_action_norm=float(np.mean(action_norms)),
        max_action_norm=float(np.max(action_norms)),
        saturation_fraction=float(saturated_steps / len(action_norms)),
    )


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("day9_pd", PDStandController()),
        Candidate(
            "hip_only",
            RecoveryStandController(
                RecoveryGains(
                    hip_kp=5.0,
                    hip_kd=0.5,
                    knee_kp=0.0,
                    knee_kd=0.0,
                    ankle_kp=0.0,
                    ankle_kd=0.0,
                )
            ),
        ),
        Candidate(
            "hip_knee",
            RecoveryStandController(
                RecoveryGains(
                    hip_kp=4.0,
                    hip_kd=0.4,
                    knee_kp=2.0,
                    knee_kd=0.2,
                    ankle_kp=0.0,
                    ankle_kd=0.0,
                )
            ),
        ),
        Candidate("hip_knee_ankle", RecoveryStandController(RecoveryGains())),
        Candidate(
            "recovery_then_pd",
            RecoveryStandController(RecoveryGains()),
            handoff_to_pd=True,
        ),
    )


def _outcome(result: Result) -> str:
    if result.success:
        return "success"
    if result.terminated:
        return "fall"
    return "unstable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inject-step", type=int, default=20)
    parser.add_argument(
        "--pitch-values",
        type=float,
        nargs="+",
        default=FIXED_EVALUATION_PITCHES,
    )
    args = parser.parse_args()

    if args.steps <= 0 or not 0 <= args.inject_step < args.steps:
        raise ValueError("--inject-step must be inside [0, --steps)")
    if args.steps - args.inject_step < DEFAULT_RECOVERY_CRITERIA.final_hold_steps:
        raise ValueError("Evaluation horizon is shorter than the required final hold window")

    criteria = DEFAULT_RECOVERY_CRITERIA
    print("=== DAY 10 STRICT RECOVERY PROTOCOL ===")
    print("Anchor: frozen Day 9 pose + PD; no PPO training.")
    print(
        "Success requires survival, entry into the full-body equilibrium "
        f"envelope, and {criteria.final_hold_steps} stable steps at the end."
    )
    print("Before injection every candidate uses the same Day 9 PD controller.")
    print()

    all_candidates = candidates()
    success_counts = {candidate.label: 0 for candidate in all_candidates}

    for pitch in args.pitch_values:
        results = [
            run_case(candidate, pitch, args.inject_step, args.steps)
            for candidate in all_candidates
        ]
        baseline_steps = results[0].steps
        print(f"Injected pitch={pitch:+.3f}")
        for result in results:
            success_counts[result.label] += int(result.success)
            entered = "--" if result.entered_at is None else str(result.entered_at)
            handoff = "--" if result.handoff_at is None else str(result.handoff_at)
            print(
                f"  {result.label:17} steps={result.steps:3d} "
                f"delta={result.steps - baseline_steps:+3d} "
                f"outcome={_outcome(result):8} entered={entered:>2} "
                f"relapsed={str(result.relapsed):5} handoff={handoff:>2} "
                f"pitch={result.final_pitch:+.3f} gyro={result.final_gyro_norm:.3f} "
                f"pose_rms={result.final_pose_error_rms:.3f} "
                f"dq_rms={result.final_selected_dq_rms:.3f} "
                f"min_h={result.min_height:.3f} action_max={result.max_action_norm:.3f} "
                f"sat={result.saturation_fraction:.2f}"
            )
        print()

    print("=== STRICT SUCCESS SUMMARY ===")
    total = len(args.pitch_values)
    for candidate in all_candidates:
        print(f"{candidate.label:17} {success_counts[candidate.label]}/{total}")


if __name__ == "__main__":
    main()
