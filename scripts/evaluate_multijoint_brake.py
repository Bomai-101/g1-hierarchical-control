"""Compare predeclared multi-joint correction and counter-braking from exact states.

All cases share the prior 10-step knee pulse and the same early-rate trigger.
This is a scripted diagnostic, not a learned recovery policy or PPO training.
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
from evaluate_damped_rate_guard import EXTRA_STATES, load_states
from evaluate_early_rate_guard import (
    ACTION_STEPS, CONSECUTIVE_STEPS, HORIZON, MONITOR_FROM,
    PITCH_GATE, RATE_THRESHOLD,
)
from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, NEIGHBORS, PULSE_ACTION, PULSE_STEPS,
    ZERO_ACTION, hold_metrics,
)
from evaluate_trajectory_branches import DEFAULT_DATA, run_case, unique_output
from record_instability_trajectories import sha256


BLEND_ACTION = 0.05
ANKLE_COUNTER_ACTION = 0.05
HIP_KNEE_COUNTER_ACTION = 0.03
STRONG_KNEE_ACTION = 0.25
COUNTER_STEPS = 5
CASES = (
    "pulse_then_pd",
    "strong_knee_guard",
    "hip_knee_blend",
    "hip_knee_then_ankle",
    "hip_knee_then_counter",
)


class MultiJointBrake:
    """One rate-triggered intervention after the shared first knee pulse."""

    def __init__(self, mode: str) -> None:
        if mode not in CASES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.streak = 0
        self.fired_at: int | None = None
        self.trigger_pitch: float | None = None
        self.trigger_rate: float | None = None
        self.counter_steps_applied = 0

    def __call__(self, elapsed: int, obs: np.ndarray) -> np.ndarray:
        if elapsed < PULSE_STEPS:
            return PULSE_ACTION.copy()
        action = ZERO_ACTION.copy()
        if self.mode == "pulse_then_pd":
            return action
        pitch, rate = float(obs[59]), float(obs[62])
        if self.fired_at is None and elapsed >= MONITOR_FROM:
            self.streak = self.streak + 1 if (
                abs(rate) >= RATE_THRESHOLD and abs(pitch) <= PITCH_GATE
            ) else 0
            if self.streak >= CONSECUTIVE_STEPS:
                self.fired_at = elapsed
                self.trigger_pitch = pitch
                self.trigger_rate = rate
        if self.fired_at is None:
            return action
        direction = 1.0 if self.trigger_rate > 0 else -1.0
        relative_step = elapsed - self.fired_at
        if 0 <= relative_step < ACTION_STEPS:
            if self.mode == "strong_knee_guard":
                action[[1, 4]] = direction * STRONG_KNEE_ACTION
            else:
                action[[0, 3]] = direction * BLEND_ACTION
                action[[1, 4]] = direction * BLEND_ACTION
        elif ACTION_STEPS <= relative_step < ACTION_STEPS + COUNTER_STEPS:
            # Counter-brake only when the first phase reversed the observed rate.
            if direction * rate < 0:
                if self.mode == "hip_knee_then_ankle":
                    action[[2, 5]] = -direction * ANKLE_COUNTER_ACTION
                elif self.mode == "hip_knee_then_counter":
                    action[[0, 3]] = -direction * HIP_KNEE_COUNTER_ACTION
                    action[[1, 4]] = -direction * HIP_KNEE_COUNTER_ACTION
                if np.any(action):
                    self.counter_steps_applied += 1
        return action


def intervention_metrics(result: dict, controller: MultiJointBrake) -> dict:
    trigger = controller.fired_at
    if trigger is None:
        return {
            "trigger_step": None, "trigger_pitch": None, "trigger_rate": None,
            "rate_after_phase1": None, "rate_after_phase2": None,
            "reversed_after_phase1": None, "reversed_after_phase2": None,
            "rate_magnitude_reduced_after_phase1": None,
            "rate_magnitude_reduced_after_phase2": None,
            "counter_steps_applied": 0,
        }
    trace = result["trace"]
    def rate_at(step: int):
        return float(trace[step - 1]["pitch_rate"]) if step <= len(trace) else None
    phase1 = rate_at(trigger + ACTION_STEPS)
    phase2 = rate_at(trigger + ACTION_STEPS + COUNTER_STEPS)
    return {
        "trigger_step": trigger,
        "trigger_pitch": controller.trigger_pitch,
        "trigger_rate": controller.trigger_rate,
        "rate_after_phase1": phase1,
        "rate_after_phase2": phase2,
        "reversed_after_phase1": (controller.trigger_rate * phase1 < 0
                                  if phase1 is not None else None),
        "reversed_after_phase2": (controller.trigger_rate * phase2 < 0
                                  if phase2 is not None else None),
        "rate_magnitude_reduced_after_phase1": (
            abs(phase1) < abs(controller.trigger_rate) if phase1 is not None else None
        ),
        "rate_magnitude_reduced_after_phase2": (
            abs(phase2) < abs(controller.trigger_rate) if phase2 is not None else None
        ),
        "counter_steps_applied": controller.counter_steps_applied,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "multijoint_brake")
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source, states, details = load_states(directory, fine_dir, checkpoint)
    quiet_pitch = float(source["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    selected = NEIGHBORS[:1] if args.center_only else NEIGHBORS + EXTRA_STATES
    output = unique_output(args.output_root, "multijoint")
    records = []
    print("=== EXACT-STATE MULTI-JOINT CORRECTION / COUNTER-BRAKE ===", flush=True)
    print(f"states={len(selected)} cases/state={len(CASES)} horizon={HORIZON}", flush=True)
    for offset in selected:
        group = []
        for mode in CASES:
            controller = MultiJointBrake(mode)
            result = run_case(states[offset], BRANCH_STEP, quiet_pitch,
                              {"name": mode}, None, HORIZON,
                              action_override=controller)
            result.update(hold_metrics(result, HORIZON))
            result.update(intervention_metrics(result, controller))
            result.update({
                "state_set": "design" if offset in NEIGHBORS else "extra",
                "pitch_offset_rad": offset[0],
                "angular_y_offset_rad_s": offset[1],
                "initial_state_key": details[offset]["key"],
            })
            group.append(result)
            records.append(result)
        print(f"pitch={offset[0]:+.4f} angular-y={offset[1]:+.4f}: "
              + " ".join(f"{r['candidate']}={r['steps']} "
                         f"(4s={r['passed_4s']}, hold={r['held_continuously_after_2s']})"
                         for r in group), flush=True)
    fields = [key for key in records[0] if key != "trace"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    (output / "results.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_sha256": sha256(directory / "summary.json"),
        "phase3_results_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "branch_step": BRANCH_STEP,
        "horizon_steps": HORIZON,
        "shared_first_pulse": {"action": PULSE_ACTION.tolist(), "steps": PULSE_STEPS},
        "trigger": {"monitor_from_step": MONITOR_FROM,
                    "abs_body_local_angular_y_threshold_rad_s": RATE_THRESHOLD,
                    "abs_pitch_gate_rad": PITCH_GATE,
                    "consecutive_steps": CONSECUTIVE_STEPS},
        "blend_action": BLEND_ACTION,
        "strong_knee_action": STRONG_KNEE_ACTION,
        "ankle_counter_action": ANKLE_COUNTER_ACTION,
        "hip_knee_counter_action": HIP_KNEE_COUNTER_ACTION,
        "phase1_steps": ACTION_STEPS,
        "conditional_counter_steps": COUNTER_STEPS,
        "design_states": [details[key] for key in NEIGHBORS],
        "extra_states": [details[key] for key in EXTRA_STATES],
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} cases: {output}", flush=True)


if __name__ == "__main__":
    main()
