"""Evaluate every Day 10 recovery checkpoint on one fixed deterministic suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.config.recovery import DEFAULT_RECOVERY_CRITERIA
from g1_control.controllers.residual import PDStandController, PPOStandController, ResidualController
from g1_control.envs.balance_env import G1Env
from g1_control.evaluation.recovery_cases import FixedRecoveryCase, fixed_stage_1_cases
from g1_control.evaluation.recovery_protocol import RecoveryTracker, measure_recovery_state
from g1_control.training.recovery_curriculum import apply_recovery_perturbation


@dataclass(frozen=True)
class CaseResult:
    case: str
    category: str
    steps: int
    terminated: bool
    entered: bool
    relapsed: bool
    success: bool
    hold_fraction: float
    final_pitch: float
    final_gyro_norm: float
    min_height: float
    mean_action_norm: float
    max_action_norm: float
    saturation_fraction: float


def run_case(
    controller: ResidualController,
    case: FixedRecoveryCase,
    max_steps: int,
) -> CaseResult:
    env = G1Env()
    env.reset()
    observation = apply_recovery_perturbation(env, case.perturbation)
    tracker = RecoveryTracker(DEFAULT_RECOVERY_CRITERIA)
    min_height = float(env.data.qpos[2])
    action_norms: list[float] = []
    saturated_steps = 0
    terminated = False
    truncated = False

    for step in range(max_steps):
        output = controller.act(observation)
        action_norms.append(output.action_l2_norm)
        saturated_steps += int(np.any(np.abs(output.action) >= 0.999))
        observation, _, terminated, truncated = env.step(output.action)
        height = float(env.data.qpos[2])
        min_height = min(min_height, height)
        state = measure_recovery_state(observation, height, env.default_q)
        tracker.update(step, state)
        if terminated or truncated:
            break

    survived = not terminated
    final_state = tracker.last_state
    if final_state is None:
        raise RuntimeError("Recovery case produced no states")
    return CaseResult(
        case=case.name,
        category=case.category,
        steps=step + 1,
        terminated=terminated,
        entered=tracker.entered_at is not None,
        relapsed=tracker.relapsed,
        success=tracker.succeeded(survived=survived),
        hold_fraction=tracker.hold_fraction,
        final_pitch=final_state.pitch,
        final_gyro_norm=final_state.gyro_norm,
        min_height=min_height,
        mean_action_norm=float(np.mean(action_norms)),
        max_action_norm=float(np.max(action_norms)),
        saturation_fraction=saturated_steps / len(action_norms),
    )


def summarize(results: list[CaseResult]) -> dict[str, float | int]:
    in_domain = [result for result in results if result.category == "in_domain"]
    stress = [result for result in results if result.category == "stress"]
    return {
        "in_domain_cases": len(in_domain),
        "in_domain_successes": sum(result.success for result in in_domain),
        "in_domain_falls": sum(result.terminated for result in in_domain),
        "in_domain_entries": sum(result.entered for result in in_domain),
        "in_domain_relapses": sum(result.relapsed for result in in_domain),
        "in_domain_mean_hold_fraction": float(np.mean([result.hold_fraction for result in in_domain])),
        "in_domain_mean_steps": float(np.mean([result.steps for result in in_domain])),
        "stress_cases": len(stress),
        "stress_successes": sum(result.success for result in stress),
        "stress_falls": sum(result.terminated for result in stress),
        "mean_action_norm": float(np.mean([result.mean_action_norm for result in results])),
        "mean_saturation_fraction": float(np.mean([result.saturation_fraction for result in results])),
    }


def ranking_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        float(row["in_domain_successes"]),
        -float(row["in_domain_falls"]),
        float(row["in_domain_mean_hold_fraction"]),
        float(row["in_domain_mean_steps"]),
        -float(row["mean_saturation_fraction"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=True)
    checkpoints = sorted(checkpoint_dir.glob("update_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No update_*.pt checkpoints in {checkpoint_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    cases = fixed_stage_1_cases()

    print("=== DAY 10 RECOVERY CHECKPOINT SWEEP ===")
    print(f"checkpoint directory: {checkpoint_dir}")
    print(f"checkpoints: {len(checkpoints)}")
    print(f"cases: {len(cases)} ({sum(c.category == 'in_domain' for c in cases)} in-domain + {sum(c.category == 'stress' for c in cases)} stress)")
    print(f"horizon: {args.steps} steps; device: {device}")
    print()

    baseline_results = [run_case(PDStandController(), case, args.steps) for case in cases]
    baseline = summarize(baseline_results)
    print(
        "PD baseline | "
        f"success={baseline['in_domain_successes']}/{baseline['in_domain_cases']} | "
        f"falls={baseline['in_domain_falls']} | "
        f"hold={baseline['in_domain_mean_hold_fraction']:.3f} | "
        f"stress={baseline['stress_successes']}/{baseline['stress_cases']}"
    )
    print()

    rows: list[dict[str, object]] = []
    details: dict[str, list[dict[str, object]]] = {}
    for index, checkpoint in enumerate(checkpoints, start=1):
        controller = PPOStandController.from_checkpoint(checkpoint, device)
        results = [run_case(controller, case, args.steps) for case in cases]
        summary = summarize(results)
        update = int(checkpoint.stem.split("_")[-1])
        row: dict[str, object] = {
            "checkpoint": checkpoint.name,
            "update": update,
            **summary,
        }
        rows.append(row)
        details[checkpoint.name] = [asdict(result) for result in results]
        print(
            f"[{index:02d}/{len(checkpoints):02d}] update={update:03d} | "
            f"success={summary['in_domain_successes']:2d}/{summary['in_domain_cases']} | "
            f"falls={summary['in_domain_falls']:2d} | "
            f"entries={summary['in_domain_entries']:2d} | "
            f"relapses={summary['in_domain_relapses']:2d} | "
            f"hold={summary['in_domain_mean_hold_fraction']:.3f} | "
            f"stress={summary['stress_successes']}/{summary['stress_cases']} | "
            f"sat={summary['mean_saturation_fraction']:.3f}"
        )

    best = max(rows, key=ranking_key)
    qualified = [
        row
        for row in rows
        if row["in_domain_successes"] > baseline["in_domain_successes"]
        and row["in_domain_falls"] <= baseline["in_domain_falls"]
    ]

    evaluation_id = datetime.now(timezone.utc).strftime("strict_sweep_%Y%m%dT%H%M%SZ")
    output_dir = checkpoint_dir / "evaluations" / evaluation_id
    output_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "checkpoint_dir": str(checkpoint_dir),
        "steps": args.steps,
        "criteria": asdict(DEFAULT_RECOVERY_CRITERIA),
        "cases": [asdict(case) for case in cases],
        "baseline": baseline,
        "baseline_details": [asdict(result) for result in baseline_results],
        "checkpoint_summaries": rows,
        "checkpoint_details": details,
        "diagnostic_best": best["checkpoint"],
        "qualified": [row["checkpoint"] for row in sorted(qualified, key=ranking_key, reverse=True)],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "checkpoint_summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== SELECTION SUMMARY ===")
    print(f"Diagnostic best: {best['checkpoint']}")
    if qualified:
        print(f"Stage 1 qualified: {', '.join(row['checkpoint'] for row in sorted(qualified, key=ranking_key, reverse=True))}")
    else:
        print("Stage 1 qualified: none")
    print(f"Saved JSON: {output_dir / 'summary.json'}")
    print(f"Saved CSV:  {output_dir / 'checkpoint_summary.csv'}")


if __name__ == "__main__":
    main()
