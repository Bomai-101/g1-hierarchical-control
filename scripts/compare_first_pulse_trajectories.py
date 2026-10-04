"""Replay matched early states with and without the first knee pulse.

Diagnostic-only telemetry: actions, joint targets, applied PD torque, pitch,
angular-y rate, and left/right floor contact are logged at every policy step.
No controller, observation, reward, or physics parameters are changed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.config.recovery import DEFAULT_RECOVERY_CRITERIA
from g1_control.controllers.residual import PPOStandController
from g1_control.envs.balance_env import G1Env
from g1_control.evaluation.recovery_protocol import (
    inside_entry_envelope, inside_hold_envelope, measure_recovery_state,
)
from evaluate_damped_rate_guard import load_states
from evaluate_early_rate_guard import EarlyRateGuard, HORIZON
from evaluate_post_pulse_handoffs import (
    BRANCH_STEP, FINE_GRID, NEIGHBORS, PULSE_ACTION, PULSE_STEPS, ZERO_ACTION,
)
from evaluate_trajectory_branches import DEFAULT_DATA, STATE_SPEC, unique_output
from record_instability_trajectories import (
    foot_geoms, new_contact_accumulator, read_contacts, sha256,
    summarize_contacts,
)


CASES = ("no_pulse_pd", "no_pulse_ppo", "pulse_then_pd", "pulse_strong_guard")
CONTROLLED = tuple(config.POLICY_JOINT_INDICES)
PLOT_COLORS = {
    "no_pulse_pd": "#718096",
    "no_pulse_ppo": "#315cce",
    "pulse_then_pd": "#25914b",
    "pulse_strong_guard": "#cc3b31",
}


def choose_action(mode: str, elapsed: int, obs: np.ndarray,
                  ppo: PPOStandController, guard: EarlyRateGuard) -> np.ndarray:
    if mode not in CASES:
        raise ValueError(f"Unknown comparison mode: {mode}")
    if mode == "no_pulse_ppo":
        return ppo.act(obs).action
    if mode == "pulse_strong_guard":
        return guard(elapsed, obs)
    if mode == "pulse_then_pd" and elapsed < PULSE_STEPS:
        return PULSE_ACTION.copy()
    return ZERO_ACTION.copy()


def first_sustained(rows: list[dict], predicate, required: int = 5) -> int | None:
    start = None
    for row in rows:
        if predicate(row):
            if start is None:
                start = row["local_step"]
            if row["local_step"] - start + 1 >= required:
                return start
        else:
            start = None
    return None


def summarize(rows: list[dict], guard: EarlyRateGuard) -> dict:
    return {
        "post_branch_steps": len(rows),
        "from_reset_steps": BRANCH_STEP + len(rows),
        "terminated": rows[-1]["terminated"],
        "truncated": rows[-1]["truncated"],
        "first_outward_pitch_step": first_sustained(
            rows, lambda r: abs(r["pitch_rad"]) >= .02
            and r["pitch_rad"] * r["angular_y_rad_s"] > 0),
        "first_abs_pitch_005_step": first_sustained(
            rows, lambda r: abs(r["pitch_rad"]) >= .05),
        "first_abs_rate_015_step": first_sustained(
            rows, lambda r: abs(r["angular_y_rad_s"]) >= .15),
        "first_exit_hold_step": next((r["local_step"] for r in rows
                                      if not r["inside_hold"]), None),
        "first_left_contact_lost_step": next((r["local_step"] for r in rows
                                              if r["left_contact_occupancy"] == 0), None),
        "first_right_contact_lost_step": next((r["local_step"] for r in rows
                                               if r["right_contact_occupancy"] == 0), None),
        "max_abs_pitch_rad": max(abs(r["pitch_rad"]) for r in rows),
        "max_abs_angular_y_rad_s": max(abs(r["angular_y_rad_s"]) for r in rows),
        "max_abs_normal_force_imbalance_N": max(
            abs(r["left_normal_force_N"] - r["right_normal_force_N"]) for r in rows),
        "max_contact_speed_m_s": max(max(r["left_contact_speed_max_m_s"] or 0.0,
                                         r["right_contact_speed_max_m_s"] or 0.0)
                                     for r in rows),
        "torque_clip_count": sum(r["torque_clip_count"] for r in rows),
        "torque_near_limit_count": sum(r["torque_near_limit_count"] for r in rows),
        "guard_trigger_step": guard.fired_at,
    }


def replay(snapshot: np.ndarray, mode: str, ppo: PPOStandController) -> tuple[list[dict], dict]:
    env = G1Env()
    env.reset()
    mujoco.mj_setState(env.model, env.data, snapshot, STATE_SPEC)
    env.episode_step = BRANCH_STEP
    floor, geoms, bodies = foot_geoms(env.model)
    guard = EarlyRateGuard(1)
    rows = []
    for elapsed in range(HORIZON):
        obs = env.get_observation()
        action = np.asarray(choose_action(mode, elapsed, obs, ppo, guard), dtype=np.float32)
        if action.shape != (config.ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Controller returned an invalid 6D action")
        target = env.action_to_target_q(action)
        contacts = new_contact_accumulator()
        applied_torques = []
        torque_clips = torque_near = 0
        for _ in range(env.decimation):
            q, dq = env.get_joint_state()
            raw = env.kp * (target - q) - env.kd * dq
            limits = env.torque_limits
            torque_clips += int(np.count_nonzero(
                (raw < limits[:, 0] - 1e-7) | (raw > limits[:, 1] + 1e-7)))
            applied = env.compute_pd_torque(target)
            bound = np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))
            torque_near += int(np.count_nonzero(np.abs(applied) >= .95 * bound))
            applied_torques.append(applied[list(CONTROLLED)].copy())
            env.data.ctrl[:] = applied
            mujoco.mj_step(env.model, env.data)
            read_contacts(env.model, env.data, floor, geoms, bodies, contacts)
        env.episode_step += 1
        obs = env.get_observation()
        terminated, truncated = env.check_termination()
        state = measure_recovery_state(obs, float(env.data.qpos[2]), env.default_q)
        contact = summarize_contacts(contacts, env.decimation)
        mean_torque = np.mean(applied_torques, axis=0)
        q, dq = env.get_joint_state()
        row = {
            "local_step": elapsed + 1,
            "from_reset_step": BRANCH_STEP + elapsed + 1,
            "time_s_after_branch": (elapsed + 1) * env.decimation * env.model.opt.timestep,
            "pitch_rad": float(obs[59]),
            "angular_y_rad_s": float(obs[62]),
            "roll_rad": float(obs[58]),
            "height_m": float(env.data.qpos[2]),
            "base_vx_m_s": float(env.data.qvel[0]),
            "base_vy_m_s": float(env.data.qvel[1]),
            "inside_entry": bool(inside_entry_envelope(state, DEFAULT_RECOVERY_CRITERIA)),
            "inside_hold": bool(inside_hold_envelope(state, DEFAULT_RECOVERY_CRITERIA)),
            "action_6d": action.tolist(),
            "target_q_6d_rad": target[list(CONTROLLED)].tolist(),
            "actual_q_6d_rad": q[list(CONTROLLED)].tolist(),
            "actual_dq_6d_rad_s": dq[list(CONTROLLED)].tolist(),
            "mean_applied_torque_6d": mean_torque.tolist(),
            "peak_abs_applied_torque_6d": np.max(np.abs(applied_torques), axis=0).tolist(),
            "torque_clip_count": torque_clips,
            "torque_near_limit_count": torque_near,
            "left_contact_occupancy": contact["left"]["contact_occupancy"],
            "right_contact_occupancy": contact["right"]["contact_occupancy"],
            "left_normal_force_N": contact["left"]["mean_normal_force_N"],
            "right_normal_force_N": contact["right"]["mean_normal_force_N"],
            "left_contact_speed_max_m_s": contact["left"]["contact_velocity_xy_max_m_s"],
            "right_contact_speed_max_m_s": contact["right"]["contact_velocity_xy_max_m_s"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        rows.append(row)
        if terminated or truncated:
            break
    return rows, summarize(rows, guard)


def write_center_svg(output: Path, cases: list[dict]) -> None:
    """Render the center-state comparison without introducing plot dependencies."""
    center = {case["mode"]: case for case in cases
              if case["pitch_offset_rad"] == 0 and case["angular_y_offset_rad_s"] == 0}
    if set(center) != set(CASES):
        return
    width, height = 1120, 945
    x0, x1 = 105, 1055
    panels = (
        ("Pitch", "rad", 130, 300, -.9, .9, lambda r: r["pitch_rad"], (-.8, 0, .8)),
        ("Body-local angular-y", "rad/s", 330, 500, -2.5, 2.5,
         lambda r: r["angular_y_rad_s"], (-2, 0, 2)),
        ("Total foot normal force", "N", 530, 700, 0, 440,
         lambda r: r["left_normal_force_N"] + r["right_normal_force_N"], (0, 200, 400)),
        ("Mean knee residual action", "normalized", 730, 900, -.1, .55,
         lambda r: (r["action_6d"][1] + r["action_6d"][4]) / 2, (0, .25, .5)),
    )
    def x(seconds: float) -> float:
        return x0 + (x1 - x0) * seconds / (HORIZON * .02)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="105" y="38" font-family="Arial,sans-serif" font-size="23" '
           'font-weight="bold" fill="#182035">First knee pulse: matched center-state trajectories</text>',
           '<text x="105" y="64" font-family="Arial,sans-serif" font-size="12" '
           'fill="#526178">All four cases start from the same saved PPO state at reset step 145; '
           'curves stop at termination.</text>']
    legend_x = (105, 335, 570, 805)
    labels = ("PD, no pulse", "PPO, no pulse", "First pulse → PD", "First pulse → knee guard")
    for mode, px, label in zip(CASES, legend_x, labels, strict=True):
        color = PLOT_COLORS[mode]
        svg.append(f'<line x1="{px}" y1="91" x2="{px+32}" y2="91" '
                   f'stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{px+39}" y="95" font-family="Arial,sans-serif" '
                   f'font-size="12" fill="#263246">{label}</text>')
    for title, unit, top, bottom, minimum, maximum, metric, ticks in panels:
        def y(value: float) -> float:
            return bottom - (bottom - top) * (value - minimum) / (maximum - minimum)
        svg.append(f'<rect x="{x0}" y="{top}" width="{x1-x0}" height="{bottom-top}" '
                   'fill="#ffffff" stroke="#d1d9e4"/>')
        svg.append(f'<rect x="{x(0):.1f}" y="{top}" width="{x(.2)-x(0):.1f}" '
                   f'height="{bottom-top}" fill="#f2f4f8"/>')
        for tick in ticks:
            py = y(tick)
            svg.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x1}" y2="{py:.1f}" '
                       'stroke="#e5e9f0"/>')
            svg.append(f'<text x="{x0-10}" y="{py+4:.1f}" text-anchor="end" '
                       f'font-family="Arial,sans-serif" font-size="11" fill="#526178">{tick:g}</text>')
        svg.append(f'<text x="{x0}" y="{top-10}" font-family="Arial,sans-serif" '
                   f'font-size="13" font-weight="bold" fill="#263246">{title} ({unit})</text>')
        for mode in CASES:
            points = " ".join(f'{x(row["time_s_after_branch"]):.1f},{y(metric(row)):.1f}'
                              for row in center[mode]["steps"])
            svg.append(f'<polyline points="{points}" fill="none" '
                       f'stroke="{PLOT_COLORS[mode]}" stroke-width="2"/>')
        if top == 730:
            for second in range(0, 11, 2):
                svg.append(f'<text x="{x(second):.1f}" y="{bottom+18}" text-anchor="middle" '
                           f'font-family="Arial,sans-serif" font-size="11" fill="#526178">{second}</text>')
    svg.append('<text x="580" y="935" text-anchor="middle" font-family="Arial,sans-serif" '
               'font-size="12" fill="#526178">Time after branch (s)</text>')
    svg.append('</svg>')
    (output / "center_timeline.svg").write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "first_pulse_timelines")
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    _, states, details = load_states(directory, fine_dir, checkpoint)
    selected = NEIGHBORS[:1] if args.center_only else NEIGHBORS
    ppo = PPOStandController.from_checkpoint(checkpoint, torch.device("cpu"))
    output = unique_output(args.output_root, "first_pulse")
    cases = []
    print("=== MATCHED FIRST-PULSE TRAJECTORIES ===", flush=True)
    print(f"states={len(selected)} cases/state={len(CASES)} horizon={HORIZON}", flush=True)
    for offset in selected:
        for mode in CASES:
            rows, summary = replay(states[offset], mode, ppo)
            label = f"pitch_{offset[0]:+.4f}_rate_{offset[1]:+.4f}_{mode}"
            with (output / f"{label}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            cases.append({
                "pitch_offset_rad": offset[0],
                "angular_y_offset_rad_s": offset[1],
                "initial_state_key": details[offset]["key"],
                "mode": mode,
                "csv": f"{label}.csv",
                "summary": summary,
                "steps": rows,
            })
            print(f"pitch={offset[0]:+.4f} angular-y={offset[1]:+.4f} "
                  f"{mode}: {summary['post_branch_steps']} steps, "
                  f"outward={summary['first_outward_pitch_step']}, "
                  f"rate>=.15={summary['first_abs_rate_015_step']}", flush=True)
    (output / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_sha256": sha256(directory / "summary.json"),
        "phase3_results_sha256": sha256(fine_dir / "results.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "branch_step": BRANCH_STEP,
        "horizon_steps": HORIZON,
        "first_pulse": {"action": PULSE_ACTION.tolist(), "steps": PULSE_STEPS},
        "states": [details[key] for key in selected],
        "cases": [{key: val for key, val in case.items() if key != "steps"} for case in cases],
    }, indent=2), encoding="utf-8")
    write_center_svg(output, cases)
    print(f"Saved matched timelines: {output}", flush=True)


if __name__ == "__main__":
    main()
