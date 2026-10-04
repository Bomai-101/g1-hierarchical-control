"""Test a pre-declared early rate guard on five identical branch states.

All cases retain the verified 10-step knee pulse, then compare zero-residual
PD against one small, state-triggered pulse and a reversed-direction control.
This is diagnostic control, not a trained recovery policy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from evaluate_early_correction import load_sources
from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, NEIGHBORS, PULSE_ACTION, PULSE_STEPS,
    ZERO_ACTION, hold_metrics,
)
from evaluate_trajectory_branches import DEFAULT_DATA, run_case, unique_output
from record_instability_trajectories import sha256


HORIZON = 500
MONITOR_FROM = 30
RATE_THRESHOLD = 0.003
PITCH_GATE = 0.020
CONSECUTIVE_STEPS = 3
ACTION_AMPLITUDE = 0.25
ACTION_STEPS = 5
EXPECTED_BASELINE_STEPS = (347, 153, 153, 192, 192)


class EarlyRateGuard:
    """Apply at most one bounded knee pulse after an early rate departure."""

    def __init__(self, direction: int):
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or +1")
        self.direction = direction
        self.streak = 0
        self.fired_at: int | None = None
        self.trigger_pitch: float | None = None
        self.trigger_rate: float | None = None

    def __call__(self, elapsed: int, obs: np.ndarray) -> np.ndarray:
        if elapsed < PULSE_STEPS:
            return PULSE_ACTION.copy()
        if self.direction == 0:
            return ZERO_ACTION.copy()

        if self.fired_at is None and elapsed >= MONITOR_FROM:
            pitch = float(obs[59])
            rate = float(obs[62])  # body-local angular-y, not Euler pitch derivative
            self.streak = self.streak + 1 if (
                abs(rate) >= RATE_THRESHOLD and abs(pitch) <= PITCH_GATE
            ) else 0
            if self.streak >= CONSECUTIVE_STEPS:
                self.fired_at = elapsed
                self.trigger_pitch = pitch
                self.trigger_rate = rate

        if self.fired_at is not None and elapsed < self.fired_at + ACTION_STEPS:
            action = ZERO_ACTION.copy()
            restoring_sign = 1 if self.trigger_rate > 0 else -1
            action[[1, 4]] = self.direction * restoring_sign * ACTION_AMPLITUDE
            return action
        return ZERO_ACTION.copy()


def post_trigger_metrics(result: dict, controller: EarlyRateGuard) -> dict:
    trigger = controller.fired_at
    if trigger is None:
        return {"trigger_step": None, "trigger_pitch": None,
                "trigger_rate": None, "rate_after_pulse": None,
                "rate_after_10_steps": None, "rate_magnitude_reduced_at_pulse_end": None,
                "opposite_rate_at_pulse_end": None,
                "pulse_end_rate_gain": None,
                "large_opposite_rate_after_10_steps": None}
    trace = result["trace"]
    pulse_end = trigger + ACTION_STEPS
    after_10 = trigger + 10
    rate_end = float(trace[pulse_end - 1]["pitch_rate"]) if pulse_end <= len(trace) else None
    rate_10 = float(trace[after_10 - 1]["pitch_rate"]) if after_10 <= len(trace) else None
    return {
        "trigger_step": trigger,
        "trigger_pitch": controller.trigger_pitch,
        "trigger_rate": controller.trigger_rate,
        "rate_after_pulse": rate_end,
        "rate_after_10_steps": rate_10,
        "rate_magnitude_reduced_at_pulse_end": (
            abs(rate_end) < abs(controller.trigger_rate) if rate_end is not None else None
        ),
        "opposite_rate_at_pulse_end": (
            bool(np.sign(controller.trigger_rate) * rate_end < 0)
            if rate_end is not None else None
        ),
        "pulse_end_rate_gain": (
            abs(rate_end) / abs(controller.trigger_rate) if rate_end is not None else None
        ),
        "large_opposite_rate_after_10_steps": (
            bool(np.sign(controller.trigger_rate) * rate_10 < -abs(controller.trigger_rate))
            if rate_10 is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "early_rate_guard")
    parser.add_argument("--center-only", action="store_true",
                        help="Check the exact center before the full five-state run.")
    args = parser.parse_args()

    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source, states, details = load_sources(directory, fine_dir, checkpoint)
    quiet_pitch = float(source["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    selected = NEIGHBORS[:1] if args.center_only else NEIGHBORS
    cases = (("pulse_then_pd", 0), ("early_rate_guard", 1),
             ("reversed_rate_guard", -1))
    output = unique_output(args.output_root, "early_guard")
    records = []

    print("=== EARLY RATE GUARD: FIVE-STATE PAIRED TEST ===", flush=True)
    print(f"states={len(selected)} cases={len(cases)} horizon={HORIZON}", flush=True)
    for index, (dp, dq) in enumerate(selected):
        group = []
        for name, direction in cases:
            controller = EarlyRateGuard(direction)
            result = run_case(states[(dp, dq)], BRANCH_STEP, quiet_pitch,
                              {"name": name}, None, HORIZON,
                              action_override=controller)
            result.update(hold_metrics(result, HORIZON))
            result.update(post_trigger_metrics(result, controller))
            result.update({"pitch_offset_rad": dp, "rate_offset_rad_s": dq,
                           "initial_pitch_rad": details[(dp, dq)]["pitch_rad"],
                           "initial_body_local_angular_y_rad_s":
                           details[(dp, dq)]["body_local_angular_y_rad_s"]})
            group.append(result)
            records.append(result)
        baseline = group[0]
        expected = EXPECTED_BASELINE_STEPS[index]
        if baseline["steps"] != expected or not baseline["terminated"]:
            raise RuntimeError(f"Baseline did not reproduce at {(dp, dq)}: "
                               f"{baseline['steps']} vs {expected}")
        print(f"pitch={dp:+.4f} angular-y={dq:+.4f} "
              + " ".join(f"{r['candidate']}={r['steps']} "
                         f"(trigger={r['trigger_step']}, "
                         f"4s={r['passed_4s']}, 10s={r['held_continuously_after_2s']})"
                         for r in group), flush=True)

    fields = [key for key in records[0] if key != "trace"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    (output / "results.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "frozen Day 9 environment and first 10-step knee pulse",
        "source_summary_sha256": sha256(directory / "summary.json"),
        "phase3_results_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "branch_step": BRANCH_STEP,
        "horizon_steps": HORIZON,
        "first_pulse": {"action": PULSE_ACTION.tolist(), "steps": PULSE_STEPS},
        "guard": {"monitor_from_step": MONITOR_FROM,
                  "abs_body_local_angular_y_threshold_rad_s": RATE_THRESHOLD,
                  "abs_pitch_gate_rad": PITCH_GATE,
                  "consecutive_steps": CONSECUTIVE_STEPS,
                  "action_amplitude": ACTION_AMPLITUDE,
                  "action_steps": ACTION_STEPS,
                  "one_shot": True},
        "states": [details[key] for key in selected],
        "case_directions": dict(cases),
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} paired cases: {output}", flush=True)


if __name__ == "__main__":
    main()
