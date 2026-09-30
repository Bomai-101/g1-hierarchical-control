"""Compare one bounded second correction after the verified knee-pulse stand.

All candidates start from the five exact Phase 3 integration states. The Day 9
environment and the first 10-step knee pulse are frozen; this script changes
only the optional diagnostic residual after that pulse.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.controllers.residual import PPOStandController
from g1_control.envs.balance_env import G1Env

from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, NEIGHBORS, PULSE_ACTION, PULSE_STEPS,
    ZERO_ACTION, hold_metrics,
)
from evaluate_trajectory_branches import DEFAULT_DATA, STATE_SPEC, run_case, unique_output, verify_reference
from record_instability_trajectories import sha256


HORIZON = 500
RATE_THRESHOLD = .005
TRIGGER_MIN_STEP = 50
TRIGGER_CONSECUTIVE = 3
PITCH_GATE = .05


def candidate_specs() -> tuple[dict, ...]:
    specs = [{"name": "pulse_then_pd", "kind": "baseline"}]
    for start in (180, 200, 220):
        for amplitude in (.25, .5):
            for duration in (5, 10):
                specs.append({"name": f"fixed_{start}_knee_{amplitude:.2f}_{duration}",
                              "kind": "fixed", "start": start,
                              "amplitude": amplitude, "duration": duration})
    for amplitude in (.25, .5):
        for duration in (5, 10):
            specs.append({"name": f"trigger_knee_{amplitude:.2f}_{duration}",
                          "kind": "trigger", "amplitude": amplitude,
                          "duration": duration})
    return tuple(specs)


class CorrectionPolicy:
    """One extra short pulse; trigger can use observed angular-y, once only."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.fired_at: int | None = None
        self.trigger_sign = 0
        self.streak = 0

    def __call__(self, elapsed: int, obs: np.ndarray) -> np.ndarray:
        if elapsed < PULSE_STEPS:
            return PULSE_ACTION
        if self.spec["kind"] == "baseline":
            return ZERO_ACTION
        if self.spec["kind"] == "fixed":
            start = int(self.spec["start"])
            if start <= elapsed < start + int(self.spec["duration"]):
                self.fired_at = start
                action = ZERO_ACTION.copy()
                action[[1, 4]] = float(self.spec["amplitude"])
                return action
            return ZERO_ACTION

        if self.spec["kind"] != "trigger":
            raise RuntimeError("Unknown correction candidate")
        if self.fired_at is None and elapsed >= TRIGGER_MIN_STEP:
            rate = float(obs[62])
            if abs(rate) >= RATE_THRESHOLD and abs(float(obs[59])) <= PITCH_GATE:
                self.streak += 1
            else:
                self.streak = 0
            if self.streak >= TRIGGER_CONSECUTIVE:
                self.fired_at = elapsed
                self.trigger_sign = 1 if rate > 0 else -1
        if self.fired_at is not None and elapsed < self.fired_at + int(self.spec["duration"]):
            action = ZERO_ACTION.copy()
            action[[1, 4]] = self.trigger_sign * float(self.spec["amplitude"])
            return action
        return ZERO_ACTION


def load_sources(directory: Path, fine_dir: Path, checkpoint: Path):
    source = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    verify_reference(source, directory, checkpoint)
    fine = json.loads((fine_dir / "results.json").read_text(encoding="utf-8"))
    if (fine["branch_step"] != BRANCH_STEP
            or fine["source_summary_sha256"] != sha256(directory / "summary.json")
            or fine["checkpoint_sha256"] != sha256(checkpoint)):
        raise RuntimeError("Phase 3 source provenance mismatch")
    metadata = json.loads((fine_dir / "initial_states.json").read_text(encoding="utf-8"))
    states = {}
    details = {}
    with np.load(fine_dir / "initial_states.npz") as arrays:
        for dp, dq in NEIGHBORS:
            matches = [item for item in metadata if item["pitch_offset_rad"] == dp
                       and item["rate_offset_rad_s"] == dq]
            if len(matches) != 1:
                raise RuntimeError(f"Missing exact state {(dp, dq)}")
            detail = matches[0]
            snapshot = arrays[detail["key"]].copy()
            if hashlib.sha256(snapshot.tobytes()).hexdigest() != detail["state_sha256"]:
                raise RuntimeError("Initial-state checksum mismatch")
            states[(dp, dq)] = snapshot
            details[(dp, dq)] = detail
    return source, states, details


def record_center(snapshot: np.ndarray, spec: dict, expected: dict) -> tuple[dict, dict]:
    env = G1Env()
    env.reset()
    mujoco.mj_setState(env.model, env.data, snapshot, STATE_SPEC)
    env.episode_step = BRANCH_STEP
    controller = CorrectionPolicy(spec)
    qpos = [env.data.qpos.copy()]
    qvel = [env.data.qvel.copy()]
    actions = []
    terminated = truncated = False
    for elapsed in range(HORIZON):
        action = np.asarray(controller(elapsed, env.get_observation()), dtype=np.float32)
        observation, _, terminated, truncated = env.step(action)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        actions.append(action.copy())
        if terminated or truncated:
            break
    if (len(actions) != expected["steps"]
            or bool(terminated) != bool(expected["terminated"])
            or abs(float(observation[59]) - expected["final_pitch"]) > 1e-6):
        raise RuntimeError(f"Video trajectory did not match evaluation: {spec['name']}")
    return {"qpos": np.stack(qpos), "qvel": np.stack(qvel),
            "action": np.stack(actions)}, {"fired_at": controller.fired_at}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "correction_experiments")
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    source, states, details = load_sources(directory, fine_dir, checkpoint)
    selected = NEIGHBORS[:1] if args.center_only else NEIGHBORS
    ppo = PPOStandController.from_checkpoint(checkpoint, torch.device("cpu"))
    quiet_pitch = float(source["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    specs = candidate_specs()
    output = unique_output(args.output_root, "correction")
    records = []
    print("=== EARLY CORRECTION: PAIRED EXACT-STATE TEST ===", flush=True)
    print(f"states={len(selected)} candidates={len(specs)} horizon={HORIZON}", flush=True)
    for dp, dq in selected:
        group = []
        for spec in specs:
            controller = CorrectionPolicy(spec)
            result = run_case(states[(dp, dq)], BRANCH_STEP, quiet_pitch,
                              {"name": spec["name"]}, ppo, HORIZON,
                              action_override=controller)
            result.update(hold_metrics(result, HORIZON))
            result.update({"pitch_offset_rad": dp, "rate_offset_rad_s": dq,
                           "correction_step": controller.fired_at,
                           "correction_sign": controller.trigger_sign
                           if spec["kind"] == "trigger" else (1 if controller.fired_at is not None else 0)})
            group.append(result)
            records.append(result)
        if dp == 0 and dq == 0:
            baseline = next(row for row in group if row["candidate"] == "pulse_then_pd")
            if baseline["steps"] != 347 or not baseline["terminated"]:
                raise RuntimeError("Center pulse-PD baseline did not reproduce")
        best = max(group, key=lambda row: row["steps"])
        print(f"pitch={dp:+.4f} angular-y={dq:+.4f}: "
              f"4s={sum(r['passed_4s'] for r in group)}/{len(group)} "
              f"continuous-10s={sum(r['held_continuously_after_2s'] for r in group)}/{len(group)} "
              f"longest={best['candidate']}:{best['steps']}", flush=True)

    # Exploratory selection is explicit; the same five states are the only
    # measured domain, so no claim of generalization is made.
    nonbaseline = specs[1:]
    def rank(spec: dict):
        group = [r for r in records if r["candidate"] == spec["name"]]
        return (sum(r["held_continuously_after_2s"] for r in group),
                sum(r["passed_4s"] for r in group),
                min(r["steps"] for r in group),
                sum(r["steps"] for r in group))
    winner = max(nonbaseline, key=rank)
    center = [r for r in records if r["pitch_offset_rad"] == 0 and r["rate_offset_rad_s"] == 0]
    baseline_result = next(r for r in center if r["candidate"] == "pulse_then_pd")
    winner_result = next(r for r in center if r["candidate"] == winner["name"])
    for name, spec, result in (("baseline_center", specs[0], baseline_result),
                               ("selected_center", winner, winner_result)):
        arrays, timing = record_center(states[(0.0, 0.0)], spec, result)
        np.savez_compressed(output / f"{name}.npz", **arrays)
        if timing["fired_at"] != result["correction_step"]:
            raise RuntimeError("Selected trajectory correction timing mismatch")

    fields = [key for key in records[0] if key != "trace"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)
    (output / "results.json").write_text(json.dumps({
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase3_fine_grid": str(fine_dir),
        "phase3_fine_results_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "branch_step": BRANCH_STEP, "horizon_steps": HORIZON,
        "first_pulse": {"action": PULSE_ACTION.tolist(), "steps": PULSE_STEPS},
        "trigger": {"abs_body_local_angular_y_threshold_rad_s": RATE_THRESHOLD,
                    "earliest_elapsed_step": TRIGGER_MIN_STEP,
                    "consecutive_steps": TRIGGER_CONSECUTIVE,
                    "abs_pitch_gate_rad": PITCH_GATE,
                    "one_shot": True},
        "candidate_specs": specs,
        "states": [details[key] for key in selected],
        "selection_rank": "count continuous 10s, count pass 4s, minimum survival, total survival",
        "selected_candidate": winner["name"],
        "selected_rank": rank(winner),
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Selected exploratory candidate: {winner['name']} rank={rank(winner)}", flush=True)
    print(f"Saved result and render states: {output}", flush=True)


if __name__ == "__main__":
    main()
