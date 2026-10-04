"""Identify short signed joint-response directions from saved early-instability states.

Diagnostic only: each exact-state branch receives a 5-step normalized pulse
followed by zero-residual PD for 15 steps. No PPO training or controller edit.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.envs.balance_env import G1Env
from evaluate_damped_rate_guard import load_states, EXTRA_STATES
from evaluate_post_pulse_handoffs import BRANCH_STEP, FINE_GRID, NEIGHBORS
from evaluate_trajectory_branches import DEFAULT_DATA, STATE_SPEC, contact_cost, unique_output
from record_instability_trajectories import foot_geoms, sha256


AMPLITUDE = 0.10
PULSE_STEPS = 5
HORIZON = 20
CHANNELS = {
    "left_hip": (0,), "left_knee": (1,), "left_ankle": (2,),
    "right_hip": (3,), "right_knee": (4,), "right_ankle": (5,),
    "hip_pair": (0, 3), "knee_pair": (1, 4), "ankle_pair": (2, 5),
}


def action_for(channel: str, sign: int) -> np.ndarray:
    action = np.zeros(config.ACTION_DIM, dtype=np.float32)
    if channel == "zero":
        if sign != 0:
            raise ValueError("The zero channel requires sign 0")
        return action
    if channel not in CHANNELS or sign not in (-1, 1):
        raise ValueError("Unknown channel or sign")
    action[list(CHANNELS[channel])] = sign * AMPLITUDE
    return action


def sample(obs: np.ndarray, env: G1Env) -> dict[str, float]:
    return {
        "roll_rad": float(obs[58]),
        "pitch_rad": float(obs[59]),
        "angular_y_rad_s": float(obs[62]),
        "base_height_m": float(env.data.qpos[2]),
    }


def trial(snapshot: np.ndarray, pulse: np.ndarray) -> dict:
    env = G1Env()
    env.reset()
    mujoco.mj_setState(env.model, env.data, snapshot, STATE_SPEC)
    env.episode_step = BRANCH_STEP
    floor, geoms, bodies = foot_geoms(env.model)
    start = sample(env.get_observation(), env)
    slip = []
    torque_clips = 0
    near_limits = 0
    peak_torque_fraction = 0.0
    at_pulse_end = at_horizon = None
    terminated = truncated = False
    completed = 0
    for elapsed in range(HORIZON):
        action = pulse if elapsed < PULSE_STEPS else np.zeros(config.ACTION_DIM, dtype=np.float32)
        target = env.action_to_target_q(action)
        for _ in range(env.decimation):
            q, dq = env.get_joint_state()
            raw = env.kp * (target - q) - env.kd * dq
            limits = env.torque_limits
            torque_clips += int(np.count_nonzero(
                (raw < limits[:, 0] - 1e-7) | (raw > limits[:, 1] + 1e-7)))
            applied = env.compute_pd_torque(target)
            bound = np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))
            fractions = np.divide(np.abs(applied), bound, out=np.zeros_like(applied), where=bound > 0)
            peak_torque_fraction = max(peak_torque_fraction, float(np.max(fractions)))
            near_limits += int(np.count_nonzero(fractions >= 0.95))
            env.data.ctrl[:] = applied
            mujoco.mj_step(env.model, env.data)
            slip.append(contact_cost(env.model, env.data, floor, geoms, bodies))
        env.episode_step += 1
        completed = elapsed + 1
        obs = env.get_observation()
        if completed == PULSE_STEPS:
            at_pulse_end = sample(obs, env)
        terminated, truncated = env.check_termination()
        if terminated or truncated:
            break
    if completed == HORIZON:
        at_horizon = sample(env.get_observation(), env)
    return {
        "start": start,
        "pulse_end": at_pulse_end,
        "horizon_end": at_horizon,
        "completed_steps": completed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "torque_clip_count": torque_clips,
        "torque_near_limit_count": near_limits,
        "peak_torque_limit_fraction": peak_torque_fraction,
        "foot_contact_speed_p95_m_s": float(np.percentile(slip, 95)) if slip else 0.0,
        "foot_contact_speed_max_m_s": max(slip, default=0.0),
    }


def case_specs() -> list[tuple[str, int]]:
    return [("zero", 0)] + [
        (channel, sign) for channel in CHANNELS for sign in (-1, 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fine-grid", type=Path, default=FINE_GRID)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=config.CHECKPOINT_DIR / "joint_pulse_response")
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    fine_dir = args.fine_grid.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    _, states, details = load_states(directory, fine_dir, checkpoint)
    selected = NEIGHBORS[:1] if args.center_only else NEIGHBORS + EXTRA_STATES
    output = unique_output(args.output_root, "joint_response")
    records = []
    print("=== EXACT-STATE JOINT PULSE RESPONSE ===", flush=True)
    print(f"states={len(selected)} cases/state={len(case_specs())} "
          f"amplitude={AMPLITUDE} pulse={PULSE_STEPS} horizon={HORIZON}", flush=True)
    for offset in selected:
        baseline = trial(states[offset], action_for("zero", 0))
        for channel, sign in case_specs():
            result = baseline if channel == "zero" else trial(states[offset], action_for(channel, sign))
            row = {
                "state_set": "design" if offset in NEIGHBORS else "extra",
                "pitch_offset_rad": offset[0],
                "angular_y_offset_rad_s": offset[1],
                "initial_state_key": details[offset]["key"],
                "channel": channel,
                "sign": sign,
                "normalized_amplitude": 0.0 if channel == "zero" else sign * AMPLITUDE,
                **result,
            }
            for moment in ("pulse_end", "horizon_end"):
                for field in ("roll_rad", "pitch_rad", "angular_y_rad_s", "base_height_m"):
                    own = result[moment]
                    base = baseline[moment]
                    row[f"delta_{moment}_{field}"] = (
                        own[field] - base[field] if own is not None and base is not None else None)
            records.append(row)
        print(f"state pitch={offset[0]:+.4f} angular-y={offset[1]:+.4f}: "
              f"{len(case_specs())} exact-state trials", flush=True)
    fields = [key for key in records[0] if key not in ("start", "pulse_end", "horizon_end")]
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
        "normalized_amplitude": AMPLITUDE,
        "joint_target_offset_rad": config.ACTION_SCALE * AMPLITUDE,
        "pulse_steps": PULSE_STEPS,
        "horizon_steps": HORIZON,
        "channels": {name: list(indices) for name, indices in CHANNELS.items()},
        "design_states": [details[key] for key in NEIGHBORS],
        "extra_states": [details[key] for key in EXTRA_STATES],
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} trials: {output}", flush=True)


if __name__ == "__main__":
    main()
