"""Compare fixed post-pulse controllers on the Phase 3 center and nearby states.

This is a diagnostic action search. It reuses saved full integration states,
the frozen Day 9 environment, and the existing Day 10 feedback controller.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.controllers.recovery import RecoveryStandController
from g1_control.controllers.residual import PPOStandController

from evaluate_trajectory_branches import (
    DEFAULT_DATA, RECOVERY_STEPS, HOLD_STEPS, run_case,
    unique_output, verify_reference,
)
from record_instability_trajectories import sha256


BRANCH_STEP = 145
FINE_GRID = config.CHECKPOINT_DIR / "recoverability_maps" / "phase3_20260930T010926Z"
NEIGHBORS = ((0.0, 0.0), (-.0005, 0.0), (.0005, 0.0),
             (0.0, -.002), (0.0, .002))
PULSE_STEPS = 10
PULSE_ACTION = np.array([0.0, .5, 0.0, 0.0, .5, 0.0], dtype=np.float32)
ZERO_ACTION = np.zeros(config.ACTION_DIM, dtype=np.float32)


def candidate_specs() -> tuple[dict, ...]:
    specs = [
        {"name": "no_pulse_pd", "after": "pd", "pulse": False},
        {"name": "no_pulse_ppo", "after": "ppo", "pulse": False},
        {"name": "pulse_pd", "after": "pd", "pulse": True},
        {"name": "pulse_ppo", "after": "ppo", "pulse": True},
    ]
    for scale in (.25, .5, 1.0):
        specs.append({"name": f"pulse_feedback_{scale:.2f}", "after": "feedback",
                      "pulse": True, "scale": scale})
    for scale in (.5, 1.0):
        for feedback_steps in (25, 50, 100):
            specs.append({"name": f"pulse_feedback_{scale:.2f}_{feedback_steps}then_pd",
                          "after": "feedback_then_pd", "pulse": True,
                          "scale": scale, "feedback_steps": feedback_steps})
    return tuple(specs)


def action_provider(spec: dict, ppo: PPOStandController,
                    feedback: RecoveryStandController):
    def act(elapsed: int, obs: np.ndarray) -> np.ndarray:
        if spec["pulse"] and elapsed < PULSE_STEPS:
            return PULSE_ACTION
        after_elapsed = elapsed - PULSE_STEPS if spec["pulse"] else elapsed
        mode = spec["after"]
        if mode == "feedback_then_pd" and after_elapsed >= spec["feedback_steps"]:
            mode = "pd"
        if mode == "pd":
            return ZERO_ACTION
        if mode == "ppo":
            return ppo.act(obs).action
        if mode in ("feedback", "feedback_then_pd"):
            return (float(spec["scale"]) * feedback.act(obs).action).astype(np.float32)
        raise RuntimeError(f"Unknown post-pulse controller: {mode}")
    return act


def hold_metrics(result: dict, horizon: int) -> dict:
    trace = result["trace"]
    # A 4 s pass is the exact Phase 3 criterion applied to the prefix.
    passed_4s = (
        len(trace) >= RECOVERY_STEPS + HOLD_STEPS
        and result["entry_step"] is not None
        and result["entry_step"] < RECOVERY_STEPS
        and all(item["hold"] for item in trace[RECOVERY_STEPS:RECOVERY_STEPS + HOLD_STEPS])
    )
    continuous = 0
    longest = 0
    for item in trace:
        continuous = continuous + 1 if item["hold"] else 0
        longest = max(longest, continuous)
    sustained_after_2s = (
        len(trace) == horizon
        and result["entry_step"] is not None
        and result["entry_step"] < RECOVERY_STEPS
        and all(item["hold"] for item in trace[RECOVERY_STEPS:])
    )
    return {"passed_4s": bool(passed_4s),
            "passed_final_2s_at_horizon": bool(result["success"]),
            "held_continuously_after_2s": bool(sustained_after_2s),
            "longest_hold_steps": longest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "handoff_evaluations")
    parser.add_argument("--horizon-steps", type=int, default=500)
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    if args.horizon_steps < RECOVERY_STEPS + HOLD_STEPS:
        parser.error("horizon-steps must be at least 200")
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    verify_reference(source, directory, checkpoint)
    fine = json.loads((fine_dir / "results.json").read_text(encoding="utf-8"))
    if (fine["branch_step"] != BRANCH_STEP
            or fine["source_summary_sha256"] != sha256(directory / "summary.json")
            or fine["checkpoint_sha256"] != sha256(checkpoint)):
        raise RuntimeError("Phase 3 state-grid provenance mismatch")
    metadata = json.loads((fine_dir / "initial_states.json").read_text(encoding="utf-8"))
    selected = (NEIGHBORS[:1] if args.center_only else NEIGHBORS)
    states = {}
    details = {}
    with np.load(fine_dir / "initial_states.npz") as arrays:
        for dp, dq in selected:
            matches = [m for m in metadata if m["pitch_offset_rad"] == dp
                       and m["rate_offset_rad_s"] == dq]
            if len(matches) != 1:
                raise RuntimeError(f"Missing unique state for offset {(dp, dq)}")
            detail = matches[0]
            state = arrays[detail["key"]].copy()
            if hashlib.sha256(state.tobytes()).hexdigest() != detail["state_sha256"]:
                raise RuntimeError("Saved Phase 3 state hash mismatch")
            states[(dp, dq)] = state
            details[(dp, dq)] = detail
    ppo = PPOStandController.from_checkpoint(checkpoint, torch.device("cpu"))
    feedback = RecoveryStandController()
    quiet_pitch = float(source["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    candidates = candidate_specs()
    output = unique_output(args.output_root, "phase4")
    records = []
    print("=== POST-PULSE HANDOFF EVALUATION ===", flush=True)
    print(f"states={len(selected)} candidates={len(candidates)} "
          f"horizon={args.horizon_steps} policy steps", flush=True)
    for dp, dq in selected:
        group = []
        for spec in candidates:
            result = run_case(states[(dp, dq)], BRANCH_STEP, quiet_pitch,
                              {"name": spec["name"]}, ppo, args.horizon_steps,
                              action_override=action_provider(spec, ppo, feedback))
            result.update(hold_metrics(result, args.horizon_steps))
            result.update({"pitch_offset_rad": dp, "rate_offset_rad_s": dq,
                           "initial_pitch_rad": details[(dp, dq)]["pitch_rad"],
                           "initial_body_local_angular_y_rad_s":
                           details[(dp, dq)]["body_local_angular_y_rad_s"]})
            records.append(result)
            group.append(result)
        if dp == 0 and dq == 0:
            baseline = next(r for r in group if r["candidate"] == "pulse_pd")
            if baseline["steps"] != 347 or not baseline["terminated"]:
                raise RuntimeError("Prior Phase 3 center continuation did not reproduce")
        ordered = sorted(group, key=lambda r: r["steps"], reverse=True)
        print(f"offset pitch={dp:+.4f} angular-y={dq:+.4f}: "
              f"4s={sum(r['passed_4s'] for r in group)}/{len(group)} "
              f"10s-hold={sum(r['held_continuously_after_2s'] for r in group)}/{len(group)} "
              f"longest={ordered[0]['candidate']}:{ordered[0]['steps']}", flush=True)

    fields = [key for key in records[0] if key != "trace"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)
    (output / "results.json").write_text(json.dumps({
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase3_grid": str(fine_dir),
        "phase3_grid_results_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "horizon_steps": args.horizon_steps,
        "pulse": {"action": PULSE_ACTION.tolist(), "steps": PULSE_STEPS,
                  "handoff_timing": "post-pulse starts at elapsed policy step 10"},
        "feedback": {"class": "RecoveryStandController", "scales": [.25, .5, 1.0],
                     "source": "src/g1_control/controllers/recovery.py"},
        "candidate_specs": candidates,
        "states": [details[key] for key in selected],
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} paired continuations: {output}", flush=True)


if __name__ == "__main__":
    main()
