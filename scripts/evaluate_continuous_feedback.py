"""Predeclared continuous bounded feedback after the common exact-state pulse.

The frozen Day 9 policy, physics, reward, and 64D observation are unchanged.
This is scripted diagnostic control, not trained PPO recovery.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from evaluate_damped_rate_guard import EXTRA_STATES, load_states
from evaluate_early_rate_guard import (
    CONSECUTIVE_STEPS, HORIZON, MONITOR_FROM, PITCH_GATE, RATE_THRESHOLD,
)
from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, NEIGHBORS, PULSE_ACTION, PULSE_STEPS,
    ZERO_ACTION, hold_metrics,
)
from evaluate_trajectory_branches import DEFAULT_DATA, run_case, unique_output
from record_instability_trajectories import sha256


GAIN = 2.0
PITCH_RATE_TARGET_GAIN_S_INV = 0.5
ACTION_LIMIT = 0.04
RATE_DEADBAND = 0.001
PITCH_DEADBAND = 0.002
STRONG_KNEE_ACTION = 0.25
ADDITIONAL_GRID_STATES = ((-0.002, 0.0), (0.002, 0.0), (0.0, -0.01), (0.0, 0.01))
CASES = ("pulse_then_pd", "strong_knee_guard", "rate_feedback", "phase_feedback")


class ContinuousFeedback:
    """A bounded 50 Hz hip/knee response with no one-shot time cutoff."""

    def __init__(self, mode: str) -> None:
        if mode not in CASES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.streak = 0
        self.fired_at: int | None = None
        self.trigger_pitch: float | None = None
        self.trigger_rate: float | None = None
        self.active_steps = 0
        self.saturated_steps = 0
        self.deadband_steps = 0
        self.peak_feedback_abs = 0.0

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
        if self.mode == "strong_knee_guard":
            if elapsed < self.fired_at + 5:
                action[[1, 4]] = (1.0 if self.trigger_rate > 0 else -1.0) * STRONG_KNEE_ACTION
            return action
        if self.mode == "rate_feedback":
            inside_deadband = abs(rate) <= RATE_DEADBAND
            rate_error = rate
        else:
            inside_deadband = abs(rate) <= RATE_DEADBAND and abs(pitch) <= PITCH_DEADBAND
            # Target angular-y rate = -0.5 * pitch (rad/s); pitch is in radians.
            rate_error = rate + PITCH_RATE_TARGET_GAIN_S_INV * pitch
        if inside_deadband:
            self.deadband_steps += 1
            return action
        raw = GAIN * rate_error
        value = float(np.clip(raw, -ACTION_LIMIT, ACTION_LIMIT))
        if abs(raw) >= ACTION_LIMIT:
            self.saturated_steps += 1
        self.peak_feedback_abs = max(self.peak_feedback_abs, abs(value))
        if value != 0.0:
            self.active_steps += 1
        action[[0, 1, 3, 4]] = value
        return action


def load_all_states(directory: Path, fine_dir: Path, checkpoint: Path):
    source, states, details = load_states(directory, fine_dir, checkpoint)
    metadata = json.loads((fine_dir / "initial_states.json").read_text(encoding="utf-8"))
    with np.load(fine_dir / "initial_states.npz") as arrays:
        for offset in ADDITIONAL_GRID_STATES:
            matches = [item for item in metadata
                       if (item["pitch_offset_rad"], item["rate_offset_rad_s"]) == offset]
            if len(matches) != 1:
                raise RuntimeError(f"Missing unique additional state: {offset}")
            detail = matches[0]
            snapshot = arrays[detail["key"]].copy()
            if hashlib.sha256(snapshot.tobytes()).hexdigest() != detail["state_sha256"]:
                raise RuntimeError(f"Saved integration-state hash mismatch: {offset}")
            states[offset], details[offset] = snapshot, detail
    return source, states, details


def short_response(result: dict, controller: ContinuousFeedback) -> dict:
    trigger = controller.fired_at
    if trigger is None:
        return {"trigger_step": None, "trigger_pitch": None, "trigger_rate": None,
                "rate_after_5_steps": None, "rate_after_10_steps": None,
                "reversed_at_5_steps": None, "rate_reduced_at_5_steps": None}
    trace = result["trace"]
    def rate_at(offset: int):
        index = trigger + offset - 1
        return float(trace[index]["pitch_rate"]) if index < len(trace) else None
    rate5, rate10 = rate_at(5), rate_at(10)
    return {
        "trigger_step": trigger,
        "trigger_pitch": controller.trigger_pitch,
        "trigger_rate": controller.trigger_rate,
        "rate_after_5_steps": rate5,
        "rate_after_10_steps": rate10,
        "reversed_at_5_steps": (controller.trigger_rate * rate5 < 0
                                if rate5 is not None else None),
        "rate_reduced_at_5_steps": (abs(rate5) < abs(controller.trigger_rate)
                                    if rate5 is not None else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "continuous_feedback")
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source, states, details = load_all_states(directory, fine_dir, checkpoint)
    quiet_pitch = float(source["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    selected = (NEIGHBORS[:1] if args.center_only else
                NEIGHBORS + EXTRA_STATES + ADDITIONAL_GRID_STATES)
    output = unique_output(args.output_root, "continuous")
    records = []
    print("=== EXACT-STATE CONTINUOUS HIP/KNEE FEEDBACK ===", flush=True)
    print(f"states={len(selected)} cases/state={len(CASES)} horizon={HORIZON}", flush=True)
    for offset in selected:
        group = []
        for mode in CASES:
            controller = ContinuousFeedback(mode)
            result = run_case(states[offset], BRANCH_STEP, quiet_pitch,
                              {"name": mode}, None, HORIZON,
                              action_override=controller)
            result.update(hold_metrics(result, HORIZON))
            result.update(short_response(result, controller))
            result.update({
                "state_set": ("design" if offset in NEIGHBORS else
                              "prior_extra" if offset in EXTRA_STATES else "additional_grid"),
                "pitch_offset_rad": offset[0],
                "angular_y_offset_rad_s": offset[1],
                "initial_state_key": details[offset]["key"],
                "feedback_active_steps": controller.active_steps,
                "feedback_saturated_steps": controller.saturated_steps,
                "feedback_deadband_steps": controller.deadband_steps,
                "feedback_peak_abs": controller.peak_feedback_abs,
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
        "feedback": {"gain": GAIN, "pitch_rate_target_gain_s_inv": PITCH_RATE_TARGET_GAIN_S_INV,
                     "normalized_action_limit": ACTION_LIMIT,
                     "rate_deadband_rad_s": RATE_DEADBAND,
                     "pitch_deadband_rad": PITCH_DEADBAND},
        "design_states": [details[key] for key in NEIGHBORS],
        "prior_extra_states": [details[key] for key in EXTRA_STATES],
        "additional_grid_states": [details[key] for key in ADDITIONAL_GRID_STATES],
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} cases: {output}", flush=True)


if __name__ == "__main__":
    main()
