"""Compare a pre-declared damped rate guard on five design and four extra states.

The frozen Day 9 simulation and original 10-step knee pulse are unchanged.
The new guard uses a small rate-proportional residual for at most five steps.
Its four extra states were in the Phase 3 grid but not used to design the
previous five-state guard; they are not a claim of broad generalization.
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
from evaluate_early_rate_guard import (
    ACTION_STEPS, CONSECUTIVE_STEPS, EarlyRateGuard, HORIZON,
    MONITOR_FROM, PITCH_GATE, RATE_THRESHOLD, post_trigger_metrics,
)
from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, NEIGHBORS, PULSE_ACTION, PULSE_STEPS,
    ZERO_ACTION, hold_metrics,
)
from evaluate_trajectory_branches import DEFAULT_DATA, run_case, unique_output, verify_reference
from record_instability_trajectories import sha256


GAIN = 2.0  # normalized knee action per rad/s of observed body-local angular-y
ACTION_LIMIT = 0.02  # 12.5 times smaller than the prior 0.25 pulse, at most
EXTRA_STATES = ((-0.001, 0.0), (0.001, 0.0), (0.0, -0.005), (0.0, 0.005))
CASES = ("pulse_then_pd", "strong_rate_guard", "damped_rate_guard")


class DampedRateGuard:
    """One five-step, state-triggered, rate-proportional knee correction."""

    def __init__(self) -> None:
        self.streak = 0
        self.fired_at: int | None = None
        self.trigger_pitch: float | None = None
        self.trigger_rate: float | None = None
        self.peak_guard_action_abs = 0.0

    def __call__(self, elapsed: int, obs: np.ndarray) -> np.ndarray:
        if elapsed < PULSE_STEPS:
            return PULSE_ACTION.copy()

        pitch = float(obs[59])
        rate = float(obs[62])  # body-local angular-y, not Euler pitch derivative
        if self.fired_at is None and elapsed >= MONITOR_FROM:
            self.streak = self.streak + 1 if (
                abs(rate) >= RATE_THRESHOLD and abs(pitch) <= PITCH_GATE
            ) else 0
            if self.streak >= CONSECUTIVE_STEPS:
                self.fired_at = elapsed
                self.trigger_pitch = pitch
                self.trigger_rate = rate

        if self.fired_at is not None and elapsed < self.fired_at + ACTION_STEPS:
            value = float(np.clip(GAIN * rate, -ACTION_LIMIT, ACTION_LIMIT))
            self.peak_guard_action_abs = max(self.peak_guard_action_abs, abs(value))
            action = ZERO_ACTION.copy()
            action[[1, 4]] = value
            return action
        return ZERO_ACTION.copy()


def load_states(directory: Path, fine_dir: Path, checkpoint: Path):
    source = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    verify_reference(source, directory, checkpoint)
    fine = json.loads((fine_dir / "results.json").read_text(encoding="utf-8"))
    if (fine["branch_step"] != BRANCH_STEP
            or fine["source_summary_sha256"] != sha256(directory / "summary.json")
            or fine["checkpoint_sha256"] != sha256(checkpoint)):
        raise RuntimeError("Phase 3 source provenance mismatch")
    metadata = json.loads((fine_dir / "initial_states.json").read_text(encoding="utf-8"))
    selected = NEIGHBORS + EXTRA_STATES
    states, details = {}, {}
    with np.load(fine_dir / "initial_states.npz") as arrays:
        for offset in selected:
            matches = [item for item in metadata
                       if (item["pitch_offset_rad"], item["rate_offset_rad_s"]) == offset]
            if len(matches) != 1:
                raise RuntimeError(f"Missing unique saved state: {offset}")
            detail = matches[0]
            snapshot = arrays[detail["key"]].copy()
            if hashlib.sha256(snapshot.tobytes()).hexdigest() != detail["state_sha256"]:
                raise RuntimeError(f"Saved integration-state hash mismatch: {offset}")
            states[offset], details[offset] = snapshot, detail
    return source, states, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "damped_rate_guard")
    parser.add_argument("--design-only", action="store_true",
                        help="Run only the original five states for a smoke check.")
    args = parser.parse_args()

    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source, states, details = load_states(directory, fine_dir, checkpoint)
    quiet_pitch = float(source["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    selected = NEIGHBORS if args.design_only else NEIGHBORS + EXTRA_STATES
    output = unique_output(args.output_root, "damped_guard")
    records = []
    print("=== DAMPED RATE GUARD: FIXED NINE-STATE COMPARISON ===", flush=True)
    print(f"states={len(selected)} cases={len(CASES)} horizon={HORIZON}", flush=True)
    for offset in selected:
        dp, dq = offset
        group = []
        for name in CASES:
            controller = (EarlyRateGuard(0) if name == "pulse_then_pd" else
                          EarlyRateGuard(1) if name == "strong_rate_guard" else
                          DampedRateGuard())
            result = run_case(states[offset], BRANCH_STEP, quiet_pitch,
                              {"name": name}, None, HORIZON,
                              action_override=controller)
            result.update(hold_metrics(result, HORIZON))
            result.update(post_trigger_metrics(result, controller))
            result.update({
                "set": "design" if offset in NEIGHBORS else "extra_validation",
                "pitch_offset_rad": dp,
                "rate_offset_rad_s": dq,
                "initial_pitch_rad": details[offset]["pitch_rad"],
                "initial_body_local_angular_y_rad_s":
                    details[offset]["body_local_angular_y_rad_s"],
                "peak_guard_action_abs": (
                    controller.peak_guard_action_abs if name == "damped_rate_guard"
                    else 0.25 if name == "strong_rate_guard" and controller.fired_at is not None
                    else 0.0
                ),
            })
            group.append(result)
            records.append(result)
        print(f"{group[0]['set']} pitch={dp:+.4f} angular-y={dq:+.4f}: "
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
        "source_summary_sha256": sha256(directory / "summary.json"),
        "phase3_results_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "branch_step": BRANCH_STEP,
        "horizon_steps": HORIZON,
        "first_pulse": {"action": PULSE_ACTION.tolist(), "steps": PULSE_STEPS},
        "trigger": {"monitor_from_step": MONITOR_FROM,
                    "abs_body_local_angular_y_threshold_rad_s": RATE_THRESHOLD,
                    "abs_pitch_gate_rad": PITCH_GATE,
                    "consecutive_steps": CONSECUTIVE_STEPS,
                    "one_shot_action_steps": ACTION_STEPS},
        "damped_policy": {"gain_action_per_rad_s": GAIN,
                          "normalized_action_limit": ACTION_LIMIT,
                          "rule": "clip(gain * observed angular-y, +/- limit) each pulse step"},
        "design_states": [details[key] for key in NEIGHBORS],
        "extra_validation_states": [details[key] for key in EXTRA_STATES],
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} paired cases: {output}", flush=True)


if __name__ == "__main__":
    main()
