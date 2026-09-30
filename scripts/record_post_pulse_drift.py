"""Record complete post-pulse drift trajectories from the saved PPO step-145 state.

Four fixed controllers are replayed from an identical MuJoCo integration state.
The script logs mechanics for diagnosis; it does not change environment or
controller parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.config.recovery import DEFAULT_RECOVERY_CRITERIA
from g1_control.controllers.recovery import RecoveryStandController
from g1_control.controllers.residual import PPOStandController
from g1_control.envs.balance_env import G1Env
from g1_control.evaluation.recovery_protocol import inside_hold_envelope, measure_recovery_state

from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, PULSE_STEPS, action_provider, candidate_specs,
)
from evaluate_trajectory_branches import DEFAULT_DATA, STATE_SPEC, unique_output, verify_reference
from record_instability_trajectories import (
    foot_geoms, integration_state, new_contact_accumulator,
    read_contacts, sha256, summarize_contacts,
)


HANDOFF = config.CHECKPOINT_DIR / "handoff_evaluations" / "phase4_20260930T012210Z" / "results.json"
LABELS = ("no_pulse_pd", "pulse_pd", "pulse_ppo", "pulse_feedback_0.25")
JOINT_LABELS = ("left_hip", "left_knee", "left_ankle",
                "right_hip", "right_knee", "right_ankle")


def write_comparison_svg(path: Path, timelines: dict[str, list[dict]]) -> None:
    """Plot the early drift; full unclipped values remain in the CSV files."""
    colors = {"no_pulse_pd": "#6b7280", "pulse_pd": "#198754",
              "pulse_ppo": "#315bb7", "pulse_feedback_0.25": "#c54137"}
    labels = {"no_pulse_pd": "No pulse → PD", "pulse_pd": "Knee pulse → PD",
              "pulse_ppo": "Knee pulse → PPO",
              "pulse_feedback_0.25": "Knee pulse → feedback 0.25"}
    panels = (
        ("pitch_rad", "Base pitch (rad)", -.2, .2, (-.2, -.1, 0, .1, .2)),
        ("pitch_rate_body_y_rad_s", "Body-local angular-y (rad/s)", -.5, .5,
         (-.5, -.25, 0, .25, .5)),
        ("com_to_front_contact_x_m", "COM to front contact x (m)", -.08, .22,
         (-.08, 0, .08, .16, .22)),
        ("com_to_rear_contact_x_m", "COM to rear contact x (m)", -.08, .22,
         (-.08, 0, .08, .16, .22)),
    )
    width, height = 1180, 1040
    left, right, top, panel_h, gap = 225, 35, 155, 166, 40
    plot_w = width - left - right
    max_time = 7.0
    bits = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<text x="40" y="43" font-size="26" font-family="sans-serif" fill="#182237">'
            'Post-pulse drift from the same G1 state</text>',
            '<text x="40" y="70" font-size="14" font-family="sans-serif" fill="#4b5870">'
            'PPO trajectory step 145; identical first 10-step knee pulse for the three pulse curves'
            '</text>']
    for i, name in enumerate(LABELS):
        x = 42 + i * 265
        bits.append(f'<line x1="{x}" y1="100" x2="{x+27}" y2="100" '
                    f'stroke="{colors[name]}" stroke-width="4"/>')
        bits.append(f'<text x="{x+35}" y="104" font-size="12" '
                    f'font-family="sans-serif" fill="#182237">{escape(labels[name])}</text>')
    for i, (key, ylabel, low, high, ticks) in enumerate(panels):
        y0 = top + i * (panel_h + gap)
        bits.append(f'<text x="40" y="{y0+panel_h/2}" font-size="13" '
                    f'font-family="sans-serif" fill="#182237">{escape(ylabel)}</text>')
        for tick in ticks:
            y = y0 + panel_h * (high - tick) / (high - low)
            bits.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" '
                        'stroke="#dce3ea" stroke-width="1"/>')
            bits.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
                        f'font-size="11" font-family="sans-serif" fill="#526078">{tick:+.2f}</text>')
        for second in range(8):
            x = left + plot_w * second / max_time
            bits.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" '
                        f'y2="{y0+panel_h}" stroke="#ecf0f4" stroke-width="1"/>')
            if i == len(panels) - 1:
                bits.append(f'<text x="{x:.1f}" y="{y0+panel_h+20}" text-anchor="middle" '
                            f'font-size="11" font-family="sans-serif" fill="#526078">{second}</text>')
        bits.append(f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{panel_h}" '
                    'fill="none" stroke="#8090a0" stroke-width="1"/>')
        if i == len(panels) - 1:
            bits.append(f'<text x="{left+plot_w/2}" y="{y0+panel_h+49}" '
                        'text-anchor="middle" font-size="13" font-family="sans-serif" '
                        'fill="#182237">Time after branch (s)</text>')
        for name in LABELS:
            coords = []
            for row in timelines[name]:
                value = row[key]
                if value is None:
                    continue
                x = left + plot_w * min(float(row["time_s"]), max_time) / max_time
                y = y0 + panel_h * (high - min(max(float(value), low), high)) / (high - low)
                coords.append(f"{x:.2f},{y:.2f}")
            bits.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2.6" '
                        f'points="{" ".join(coords)}"/>')
    bits.append('<text x="40" y="1005" font-size="11" font-family="sans-serif" '
                'fill="#526078">Static COM/contact margins are diagnostic proxies, not ZMP. '
                'Curves are clipped to the displayed y-ranges; CSV retains full values.</text>')
    bits.append('</svg>')
    path.write_text("\n".join(bits), encoding="utf-8")


def record_one(snapshot: np.ndarray, spec: dict, ppo, feedback,
               expected: dict, horizon: int) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    env = G1Env()
    env.reset()
    mujoco.mj_setState(env.model, env.data, snapshot, STATE_SPEC)
    env.episode_step = BRANCH_STEP
    model, data = env.model, env.data
    floor, geoms, bodies = foot_geoms(model)
    act = action_provider(spec, ppo, feedback)
    selected = env.POLICY_JOINT_INDICES
    rows: list[dict] = []
    states = [integration_state(model, data)]
    qpos = [data.qpos.copy()]
    qvel = [data.qvel.copy()]
    actions = []
    target_q = []
    torques = []
    terminated = truncated = False

    for elapsed in range(horizon):
        observation = env.get_observation()
        action = np.asarray(act(elapsed, observation), dtype=np.float32)
        if action.shape != (config.ACTION_DIM,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise RuntimeError("Invalid 6D action from replay controller")
        target = env.action_to_target_q(action)
        accumulator = new_contact_accumulator()
        raw_torques = []
        applied_torques = []
        for _ in range(env.decimation):
            q, dq = env.get_joint_state()
            raw = env.kp * (target - q) - env.kd * dq
            applied = env.compute_pd_torque(target)
            raw_torques.append(raw.copy())
            applied_torques.append(applied.copy())
            data.ctrl[:] = applied
            mujoco.mj_step(model, data)
            read_contacts(model, data, floor, geoms, bodies, accumulator)
        env.episode_step += 1
        observation = env.get_observation()
        terminated, truncated = env.check_termination()
        state = measure_recovery_state(observation, float(data.qpos[2]), env.default_q)
        contact = summarize_contacts(accumulator, env.decimation)
        positions = [pos for side in accumulator.values() for pos in side["positions"]]
        contact_x = [float(pos[0]) for pos in positions]
        com = data.subtree_com[1].copy()
        front_x = max(contact_x) if contact_x else None
        rear_x = min(contact_x) if contact_x else None
        raw_arr = np.asarray(raw_torques)
        applied_arr = np.asarray(applied_torques)
        bounds = env.torque_limits
        clipped = (raw_arr < bounds[:, 0] - 1e-7) | (raw_arr > bounds[:, 1] + 1e-7)
        near = np.abs(applied_arr) >= .95 * np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1]))
        q, dq = env.get_joint_state()
        row = {
            "elapsed": elapsed + 1,
            "time_s": (elapsed + 1) * model.opt.timestep * env.decimation,
            "mode": ("pulse" if spec["pulse"] and elapsed < PULSE_STEPS else spec["after"]),
            "roll_rad": float(observation[58]),
            "pitch_rad": float(observation[59]),
            "pitch_rate_body_y_rad_s": float(observation[62]),
            "gyro_norm_rad_s": float(np.linalg.norm(observation[61:64])),
            "height_m": float(data.qpos[2]),
            "base_vx_m_s": float(data.qvel[0]),
            "base_vz_m_s": float(data.qvel[2]),
            "com_x_m": float(com[0]),
            "com_z_m": float(com[2]),
            "contact_rear_x_m": rear_x,
            "contact_front_x_m": front_x,
            "com_to_front_contact_x_m": None if front_x is None else front_x - float(com[0]),
            "com_to_rear_contact_x_m": None if rear_x is None else float(com[0]) - rear_x,
            "left_contact_occupancy": contact["left"]["contact_occupancy"],
            "right_contact_occupancy": contact["right"]["contact_occupancy"],
            "left_normal_force_N": contact["left"]["mean_normal_force_N"],
            "right_normal_force_N": contact["right"]["mean_normal_force_N"],
            "left_contact_speed_p95_m_s": contact["left"]["contact_velocity_xy_p95_m_s"],
            "right_contact_speed_p95_m_s": contact["right"]["contact_velocity_xy_p95_m_s"],
            "action_norm": float(np.linalg.norm(action)),
            "torque_clip_count": int(clipped.sum()),
            "torque_near_limit_count": int(near.sum()),
            "in_hold_envelope": inside_hold_envelope(state, DEFAULT_RECOVERY_CRITERIA),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        for index, label in enumerate(JOINT_LABELS):
            joint = int(selected[index])
            row[f"{label}_q_rad"] = float(q[joint])
            row[f"{label}_dq_rad_s"] = float(dq[joint])
            row[f"{label}_target_rad"] = float(target[joint])
            row[f"{label}_torque_mean_Nm"] = float(applied_arr[:, joint].mean())
            row[f"{label}_action"] = float(action[index])
        rows.append(row)
        states.append(integration_state(model, data))
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        actions.append(action.copy())
        target_q.append(target.copy())
        torques.append(applied_arr)
        if terminated or truncated:
            break

    if (len(rows) != expected["steps"] or bool(terminated) != bool(expected["terminated"])
            or abs(rows[-1]["pitch_rad"] - expected["final_pitch"]) > 1e-6
            or abs(rows[-1]["pitch_rate_body_y_rad_s"] - expected["final_rate"]) > 1e-6):
        raise RuntimeError(f"Replay mismatch for {spec['name']}")
    arrays = {
        "integration_state": np.stack(states),
        "qpos": np.stack(qpos),
        "qvel": np.stack(qvel),
        "action": np.stack(actions),
        "target_q": np.stack(target_q),
        "applied_pd_torque": np.stack(torques),
    }
    summary = {
        "candidate": spec["name"], "steps": len(rows),
        "fall_time_s_after_branch": rows[-1]["time_s"] if terminated else None,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "first_hold_exit_after_40": next((row["elapsed"] for row in rows[40:]
                                           if not row["in_hold_envelope"]), None),
        "torque_clip_total": sum(row["torque_clip_count"] for row in rows),
        "torque_near_limit_total": sum(row["torque_near_limit_count"] for row in rows),
        "initial_state_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
    }
    return rows, arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--handoff-results", type=Path, default=HANDOFF)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "drift_diagnostics")
    args = parser.parse_args()
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    handoff_path = args.handoff_results.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    phase1 = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    verify_reference(phase1, directory, checkpoint)
    phase3 = json.loads((fine_dir / "results.json").read_text(encoding="utf-8"))
    phase4 = json.loads(handoff_path.read_text(encoding="utf-8"))
    if (phase3["branch_step"] != BRANCH_STEP
            or phase3["source_summary_sha256"] != sha256(directory / "summary.json")
            or phase4["phase3_grid_results_sha256"] != sha256(fine_dir / "results.json")
            or phase4["checkpoint_sha256"] != sha256(checkpoint)):
        raise RuntimeError("Phase 1/3/4 provenance mismatch")
    metadata = json.loads((fine_dir / "initial_states.json").read_text(encoding="utf-8"))
    matches = [item for item in metadata if item["pitch_offset_rad"] == 0
               and item["rate_offset_rad_s"] == 0]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one saved center state")
    with np.load(fine_dir / "initial_states.npz") as states:
        snapshot = states[matches[0]["key"]].copy()
    if hashlib.sha256(snapshot.tobytes()).hexdigest() != matches[0]["state_sha256"]:
        raise RuntimeError("Center integration-state hash mismatch")
    previous = {row["candidate"]: row for row in phase4["results"]
                if row["pitch_offset_rad"] == 0 and row["rate_offset_rad_s"] == 0}
    ppo = PPOStandController.from_checkpoint(checkpoint, torch.device("cpu"))
    feedback = RecoveryStandController()
    specs = {spec["name"]: spec for spec in candidate_specs()}
    output = unique_output(args.output_root, "phase4_drift")
    summaries = {}
    timelines = {}
    print("=== CENTER POST-PULSE DRIFT DIAGNOSIS ===", flush=True)
    for label in LABELS:
        rows, arrays, summary = record_one(snapshot, specs[label], ppo, feedback,
                                           previous[label], int(phase4["horizon_steps"]))
        np.savez_compressed(output / f"{label}.npz", **arrays)
        with (output / f"{label}_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries[label] = summary
        timelines[label] = rows
        print(f"{label:24} steps={summary['steps']:3d} "
              f"hold_exit={summary['first_hold_exit_after_40']} "
              f"torque_clips={summary['torque_clip_total']}", flush=True)
    (output / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "source_handoff_results": str(handoff_path),
        "source_handoff_sha256": sha256(handoff_path),
        "source_fine_grid_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "branch_step": BRANCH_STEP,
        "center_state_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
        "com_contact_note": "contact x interval is an approximate static support indicator, not ZMP or a dynamic stability certificate",
        "policies": summaries,
    }, indent=2), encoding="utf-8")
    write_comparison_svg(output / "drift_comparison.svg", timelines)
    print(f"Saved detailed replay: {output}", flush=True)


if __name__ == "__main__":
    main()
