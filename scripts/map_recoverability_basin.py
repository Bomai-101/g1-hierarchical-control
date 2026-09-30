"""Phase 3: map a fixed early recovery pulse over nearby pitch and angular-y states.

The map perturbs only the floating-base Euler pitch and its body-local angular-y
velocity at the saved PPO step-145 state. Other qpos/qvel components remain
fixed. Results are diagnostic for this conditional slice, not a global basin.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.controllers.residual import PPOStandController
from g1_control.envs.balance_env import G1Env

from evaluate_trajectory_branches import (
    DEFAULT_DATA, HOLD_STEPS, RECOVERY_STEPS, STATE_SPEC,
    run_case, unique_output, verify_reference,
)
from record_instability_trajectories import foot_geoms, sha256


BRANCH_STEP = 145
PHASE2_RESULT = config.CHECKPOINT_DIR / "branch_recovery" / "phase2_20260930T005253Z" / "results.json"
PITCH_OFFSETS = (-.020, -.010, -.005, 0.0, .005, .010, .020)
RATE_OFFSETS = (-.100, -.050, -.025, 0.0, .025, .050, .100)
CANDIDATES = (
    {"name": "pulse_then_pd", "kind": "pulse", "group": "knee",
     "amplitude": .5, "pulse_steps": 10},
    {"name": "zero_pd", "kind": "zero_pd"},
    {"name": "original_ppo", "kind": "original"},
)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return MuJoCo's wxyz quaternion for extrinsic yaw-pitch-roll."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array((cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr), dtype=np.float64)


def measure_initial_contacts(env: G1Env) -> dict:
    floor, geoms, _ = foot_geoms(env.model)
    contacts = {"left": 0, "right": 0}
    min_dist = None
    for cid in range(env.data.ncon):
        contact = env.data.contact[cid]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor not in pair:
            continue
        other = next(iter(pair - {floor}), -1)
        for side, ids in geoms.items():
            if other in ids:
                contacts[side] += 1
                min_dist = float(contact.dist) if min_dist is None else min(min_dist, float(contact.dist))
    return {"left_foot_contacts": contacts["left"],
            "right_foot_contacts": contacts["right"],
            "minimum_contact_distance_m": min_dist}


def perturbed_state(env: G1Env, original: np.ndarray, base_rpy: np.ndarray,
                    base_rate: float, pitch_delta: float, rate_delta: float) -> tuple[np.ndarray, dict]:
    env.reset()
    mujoco.mj_setState(env.model, env.data, original, STATE_SPEC)
    env.episode_step = BRANCH_STEP
    if pitch_delta or rate_delta:
        target_pitch = float(base_rpy[1]) + pitch_delta
        env.data.qpos[3:7] = rpy_to_quat(float(base_rpy[0]), target_pitch, float(base_rpy[2]))
        env.data.qvel[4] = base_rate + rate_delta
        # The source solver warmstart belongs to the unperturbed state.
        env.data.qacc_warmstart[:] = 0.0
    # Refresh derived positions/contacts after changing the floating base.
    # For the exact center we retain the original integration state itself.
    mujoco.mj_forward(env.model, env.data)
    obs = env.get_observation()
    expected_pitch = float(base_rpy[1]) + pitch_delta
    expected_rate = base_rate + rate_delta
    if abs(float(obs[59]) - expected_pitch) > 2e-6 or abs(float(obs[62]) - expected_rate) > 2e-6:
        raise RuntimeError("Constructed state does not match requested pitch/angular-y")
    fallen, timeout = env.check_termination()
    if fallen or timeout:
        raise RuntimeError("Perturbation starts in a terminal state")
    contacts = measure_initial_contacts(env)
    if pitch_delta or rate_delta:
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC), dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    else:
        state = original.copy()
    metadata = {"pitch_rad": float(obs[59]),
                "body_local_angular_y_rad_s": float(obs[62]),
                "roll_rad": float(obs[58]), "yaw_rad": float(obs[60]),
                "base_height_m": float(env.data.qpos[2]),
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                **contacts}
    return state, metadata


def svg_heatmap(path: Path, rows: list[dict], pitches: tuple[float, ...],
                rates: tuple[float, ...], center_pitch: float, center_rate: float,
                horizon_steps: int) -> None:
    """Create a dependency-free, labelled vector map of the fixed 4 s criterion."""
    by_key = {(r["candidate"], r["pitch_offset_rad"], r["rate_offset_rad_s"]): r
              for r in rows}
    cell = 47
    left = 95
    top = 152
    gap = 65
    panel_width = len(pitches) * cell
    width = left + 3 * panel_width + 2 * gap + 55
    height = top + len(rates) * cell + 132
    panels = (("pulse_then_pd", "Knee pulse → PD"),
              ("zero_pd", "Zero residual PD"),
              ("original_ppo", "Original PPO"))
    bits = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<text x="40" y="42" font-size="25" font-family="sans-serif" fill="#172235">'
            'Early-state recoverability around PPO step 145</text>',
            '<text x="40" y="70" font-size="14" font-family="sans-serif" fill="#45546a">'
            f'Entry within 2 s + final 2 s continuous hold of a {horizon_steps * .02:.1f} s horizon; '
            'base pitch and angular-y varied</text>',
            '<rect x="40" y="84" width="14" height="14" fill="#4f9b65"/>',
            '<text x="60" y="96" font-size="12" font-family="sans-serif">success</text>',
            '<rect x="142" y="84" width="14" height="14" fill="#d37970"/>',
            '<text x="162" y="96" font-size="12" font-family="sans-serif">failure</text>',
            '<circle cx="247" cy="91" r="5" fill="#172235"/>',
            '<text x="258" y="96" font-size="12" font-family="sans-serif">recorded center</text>']
    for index, (name, label) in enumerate(panels):
        x0 = left + index * (panel_width + gap)
        count = sum(bool(by_key[(name, p, q)]["success"]) for p in pitches for q in rates)
        bits.append(f'<text x="{x0}" y="131" font-size="17" font-family="sans-serif" '
                    f'font-weight="600" fill="#172235">{escape(label)} · {count}/{len(pitches)*len(rates)}</text>')
        for iy, dq in enumerate(reversed(rates)):
            for ix, dp in enumerate(pitches):
                row = by_key[(name, dp, dq)]
                x = x0 + ix * cell
                y = top + iy * cell
                fill = "#4f9b65" if row["success"] else "#d37970"
                bits.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}"/>')
                if dp == 0.0 and dq == 0.0:
                    bits.append(f'<circle cx="{x+cell/2-1}" cy="{y+cell/2-1}" r="6" '
                                'fill="#172235"/>')
        for ix, dp in enumerate(pitches):
            bits.append(f'<text x="{x0+ix*cell+cell/2-1}" y="{top+len(rates)*cell+18}" '
                        f'text-anchor="middle" font-size="10" font-family="sans-serif" fill="#45546a">'
                        f'{center_pitch+dp:+.3f}</text>')
        bits.append(f'<text x="{x0+panel_width/2}" y="{top+len(rates)*cell+47}" '
                    'text-anchor="middle" font-size="13" font-family="sans-serif" fill="#172235">'
                    'Initial pitch (rad)</text>')
        if index == 0:
            for iy, dq in enumerate(reversed(rates)):
                bits.append(f'<text x="{x0-9}" y="{top+iy*cell+cell/2+3}" '
                            f'text-anchor="end" font-size="10" font-family="sans-serif" fill="#45546a">'
                            f'{center_rate+dq:+.3f}</text>')
    bits.append(f'<text transform="translate(22 {top+len(rates)*cell/2}) rotate(-90)" '
                'text-anchor="middle" font-size="13" font-family="sans-serif" fill="#172235">'
                'Initial body-local angular-y (rad/s)</text>')
    bits.append(f'<text x="40" y="{height-31}" font-size="11" font-family="sans-serif" '
                'fill="#45546a">Source: Phase 1 PPO step 145, MuJoCo 3.14.0 · '
                f'cell color applies only to the tested {horizon_steps * .02:.1f} s horizon</text>')
    bits.append('</svg>')
    path.write_text("\n".join(bits), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--phase2-result", type=Path, default=PHASE2_RESULT)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "recoverability_maps")
    parser.add_argument("--pitch-offsets", nargs="+", type=float, default=PITCH_OFFSETS)
    parser.add_argument("--rate-offsets", nargs="+", type=float, default=RATE_OFFSETS)
    parser.add_argument("--horizon-steps", type=int, default=RECOVERY_STEPS + HOLD_STEPS,
                        help="At least 200; final 100 steps must meet the hold envelope")
    args = parser.parse_args()
    if args.horizon_steps < RECOVERY_STEPS + HOLD_STEPS:
        parser.error("horizon-steps must be at least 200")
    pitches = tuple(sorted(set(args.pitch_offsets)))
    rates = tuple(sorted(set(args.rate_offsets)))
    if not pitches or not rates or 0.0 not in pitches or 0.0 not in rates:
        parser.error("Both offset axes must include 0.0")
    if any(not math.isfinite(x) for x in pitches + rates):
        parser.error("Offsets must be finite")
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    phase2_path = args.phase2_result.expanduser().resolve(strict=True)
    meta = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    verify_reference(meta, directory, checkpoint)
    phase2 = json.loads(phase2_path.read_text(encoding="utf-8"))
    proven = [r for r in phase2["results"] if r["policy"] == "best_ppo"
              and r["branch_step"] == BRANCH_STEP
              and r["candidate"] == "knee_+0.50_10step" and r["success"]]
    if len(proven) != 1 or phase2["source_summary_sha256"] != sha256(directory / "summary.json"):
        raise RuntimeError("Phase 2 successful source branch cannot be verified")
    with np.load(directory / "best_ppo.npz") as arrays:
        original = arrays["integration_state"][BRANCH_STEP].copy()
        next_qpos = arrays["qpos"][BRANCH_STEP + 1].copy()
        next_qvel = arrays["qvel"][BRANCH_STEP + 1].copy()
    original_controller = PPOStandController.from_checkpoint(checkpoint, torch.device("cpu"))
    env = G1Env()
    env.reset()
    mujoco.mj_setState(env.model, env.data, original, STATE_SPEC)
    base_rpy = env.quaternion_to_rpy(env.data.qpos[3:7])
    base_rate = float(env.data.qvel[4])
    # A warmstart-only control separates true state sensitivity from the
    # bookkeeping change required for off-center physical perturbations.
    env.data.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    recomputed = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, recomputed, STATE_SPEC)
    warmstart_control = run_case(
        recomputed, BRANCH_STEP,
        float(meta["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"]),
        CANDIDATES[0], original_controller, args.horizon_steps,
    )
    if args.horizon_steps == RECOVERY_STEPS + HOLD_STEPS and not warmstart_control["success"]:
        raise RuntimeError("Warmstart-only center lost success; shifted-state map is confounded")
    output = unique_output(args.output_root, "phase3")
    records = []
    state_meta = []
    state_arrays = {}
    quiet_pitch = float(meta["policies"]["best_ppo"]["events"]["quiet_pitch_reference_rad"])
    print("=== PHASE 3 LOCAL RECOVERABILITY MAP ===", flush=True)
    print(f"PPO branch step {BRANCH_STEP}; grid {len(pitches)} x {len(rates)}; "
          "three fixed controllers", flush=True)
    for dq in rates:
        row_successes = 0
        for dp in pitches:
            state, detail = perturbed_state(env, original, base_rpy, base_rate, dp, dq)
            state_key = f"p{pitches.index(dp):02d}_r{rates.index(dq):02d}"
            state_arrays[state_key] = state
            detail.update({"key": state_key, "pitch_offset_rad": dp,
                           "rate_offset_rad_s": dq})
            state_meta.append(detail)
            for candidate in CANDIDATES:
                expected = (next_qpos, next_qvel) if dp == 0.0 and dq == 0.0 and candidate["kind"] == "original" else None
                result = run_case(state, BRANCH_STEP, quiet_pitch, candidate,
                                  original_controller, args.horizon_steps, expected)
                result.update({"pitch_offset_rad": dp, "rate_offset_rad_s": dq,
                               "initial_left_foot_contacts": detail["left_foot_contacts"],
                               "initial_right_foot_contacts": detail["right_foot_contacts"],
                               "initial_min_contact_distance_m": detail["minimum_contact_distance_m"]})
                records.append(result)
                if candidate["name"] == "pulse_then_pd":
                    row_successes += int(result["success"])
        print(f"angular-y offset {dq:+.3f}: pulse successes "
              f"{row_successes}/{len(pitches)}", flush=True)

    center = next(r for r in records if r["candidate"] == "pulse_then_pd"
                  and r["pitch_offset_rad"] == 0.0 and r["rate_offset_rad_s"] == 0.0)
    if args.horizon_steps == RECOVERY_STEPS + HOLD_STEPS and (
        not center["success"] or center["steps"] != args.horizon_steps
    ):
        raise RuntimeError("Recorded Phase 2 center success did not reproduce")
    np.savez_compressed(output / "initial_states.npz", **state_arrays)
    (output / "initial_states.json").write_text(json.dumps(state_meta, indent=2), encoding="utf-8")
    columns = [k for k in records[0] if k != "trace"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in records)
    (output / "results.json").write_text(json.dumps({
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(directory),
        "source_summary_sha256": sha256(directory / "summary.json"),
        "phase2_result": str(phase2_path), "phase2_result_sha256": sha256(phase2_path),
        "checkpoint_sha256": sha256(checkpoint), "branch_step": BRANCH_STEP,
        "warmstart_only_center_success": warmstart_control["success"],
        "warmstart_only_center_steps": warmstart_control["steps"],
        "center_pitch_rad": float(base_rpy[1]),
        "center_body_local_angular_y_rad_s": base_rate,
        "pitch_offsets_rad": pitches, "rate_offsets_rad_s": rates,
        "perturbation": "qpos free-base quaternion Euler pitch and qvel[4] body-local angular-y only; "
                        "solver warmstart zeroed for changed states, mj_forward recomputed",
        "success_rule": f"entry within first 100 steps and continuous hold during final 100 of {args.horizon_steps} steps",
        "horizon_steps": args.horizon_steps,
        "results": records,
    }, indent=2), encoding="utf-8")
    svg_heatmap(output / "recoverability_map.svg", records, pitches, rates,
                float(base_rpy[1]), base_rate, args.horizon_steps)
    for name, _ in ((c["name"], c) for c in CANDIDATES):
        subset = [r for r in records if r["candidate"] == name]
        print(f"{name}: {sum(r['success'] for r in subset)}/{len(subset)} successes", flush=True)
    print(f"Saved map and per-case data: {output}", flush=True)


if __name__ == "__main__":
    main()
