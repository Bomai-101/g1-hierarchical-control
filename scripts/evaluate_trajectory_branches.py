"""Phase 2: test bounded pulses followed by zero-residual PD from saved states.

This diagnostic does not train or alter the standing environment. Every
candidate starts from the same mjSTATE_INTEGRATION snapshot for its branch.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.config.recovery import DEFAULT_RECOVERY_CRITERIA
from g1_control.controllers.residual import PDStandController, PPOStandController
from g1_control.envs.balance_env import G1Env
from g1_control.evaluation.recovery_protocol import (
    inside_entry_envelope,
    inside_hold_envelope,
    measure_recovery_state,
)

from record_instability_trajectories import foot_geoms, contact_velocity_xy, sha256


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
DEFAULT_DATA = config.CHECKPOINT_DIR / "instability_trajectories" / "phase1_20260930T002805Z"
GROUPS = {"hip": (0, 3), "knee": (1, 4), "ankle": (2, 5)}
RECOVERY_STEPS = 100  # 2 s at 50 Hz
HOLD_STEPS = 100      # an additional 2 s


def unique_output(root: Path, label: str = "phase2") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime(f"{label}_%Y%m%dT%H%M%SZ")
    for i in range(1000):
        path = root / (prefix if i == 0 else f"{prefix}_{i:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Could not create Phase 2 output directory")


def branch_steps(summary: dict) -> dict[str, int]:
    events = summary["events"]
    a = int(events["T_deviation_step"])
    c = int(events["T_acceleration_step"])
    t = int(events["T_termination_step"])
    if not 0 < a < c < t:
        raise ValueError("Invalid ordered instability markers")
    # B samples the observed transition; D is late, but still before falling.
    return {"P_pre_40": a - 40, "P_pre_20": a - 20,
            "A_deviation": a, "B_between": (a + c) // 2,
            "C_acceleration": c, "D_late": t - 20}


def make_candidates(amplitudes: tuple[float, ...], pulse_steps: tuple[int, ...]) -> list[dict]:
    cases = [{"name": "original", "kind": "original"},
             {"name": "zero_pd", "kind": "zero_pd"}]
    for duration in pulse_steps:
        for group in GROUPS:
            for amplitude in amplitudes:
                for sign in (-1, 1):
                    cases.append({"name": f"{group}_{sign * amplitude:+.2f}_{duration}step",
                                  "kind": "pulse", "group": group,
                                  "amplitude": sign * amplitude,
                                  "pulse_steps": duration})
    return cases


def action_for(candidate: dict, elapsed: int, obs: np.ndarray, original) -> np.ndarray:
    if candidate["kind"] == "original":
        return original.act(obs).action
    action = np.zeros(config.ACTION_DIM, dtype=np.float32)
    if candidate["kind"] == "pulse" and elapsed < candidate["pulse_steps"]:
        action[list(GROUPS[candidate["group"]])] = candidate["amplitude"]
    # The pulse candidate deliberately hands off to zero-residual PD here.
    # It does not return to the original PPO controller after the pulse.
    return action


def contact_cost(model, data, floor: int, foot_geoms: dict, foot_bodies: dict) -> float:
    speeds = []
    for cid in range(data.ncon):
        contact = data.contact[cid]
        first, second = int(contact.geom1), int(contact.geom2)
        if first != floor and second != floor:
            continue
        other = second if first == floor else first
        for side, ids in foot_geoms.items():
            if other in ids:
                speeds.append(contact_velocity_xy(model, data, foot_bodies[side], contact.pos))
    return max(speeds, default=0.0)


def run_case(snapshot: np.ndarray, branch: int, quiet_pitch: float,
             candidate: dict, original, horizon: int,
             expected_next: tuple[np.ndarray, np.ndarray] | None = None,
             action_override: Callable[[int, np.ndarray], np.ndarray] | None = None) -> dict:
    env = G1Env()
    env.reset()
    mujoco.mj_setState(env.model, env.data, snapshot, STATE_SPEC)
    env.episode_step = branch
    # Reference height is set by reset and is identical to Phase 1.
    start_obs = env.get_observation()
    started_inside_entry = inside_entry_envelope(
        measure_recovery_state(start_obs, float(env.data.qpos[2]), env.default_q),
        DEFAULT_RECOVERY_CRITERIA,
    )
    start_rate = float(start_obs[62])
    outward_sign = np.sign(float(start_obs[59]) - quiet_pitch)
    if outward_sign == 0:
        outward_sign = np.sign(start_rate) or 1.0
    initial_outward_rate = outward_sign * start_rate
    floor, geoms, bodies = foot_geoms(env.model)

    trace = []
    torque_clipped = 0
    near_limit = 0
    slip_speeds = []
    max_action = 0.0
    max_joint_speed = 0.0
    entry_start = None
    entry_run = 0
    left_entry = not started_inside_entry
    reentered_after_exit = False
    braking_at = None
    terminated = truncated = False
    # Braking means the outward angular speed is reduced to <= 0 for 3
    # consecutive samples, not simply a momentary derivative sign change.
    brake_run = 0
    for elapsed in range(horizon):
        obs = env.get_observation()
        action = (action_override(elapsed, obs) if action_override is not None
                  else action_for(candidate, elapsed, obs, original))
        action = np.asarray(action, dtype=np.float32)
        if (action.shape != (config.ACTION_DIM,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1.0)):
            raise ValueError("Branch controller returned invalid 6D action")
        max_action = max(max_action, float(np.max(np.abs(action))))
        target = env.action_to_target_q(action)
        for _ in range(env.decimation):
            q, dq = env.get_joint_state()
            raw = env.kp * (target - q) - env.kd * dq
            limits = env.torque_limits
            torque_clipped += int(np.count_nonzero((raw < limits[:, 0] - 1e-7)
                                                     | (raw > limits[:, 1] + 1e-7)))
            applied = env.compute_pd_torque(target)
            near_limit += int(np.count_nonzero(np.abs(applied) >=
                              .95 * np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))))
            env.data.ctrl[:] = applied
            mujoco.mj_step(env.model, env.data)
            slip_speeds.append(contact_cost(env.model, env.data, floor, geoms, bodies))
        env.episode_step += 1
        if elapsed == 0 and expected_next is not None:
            pos_error = float(np.max(np.abs(env.data.qpos - expected_next[0])))
            vel_error = float(np.max(np.abs(env.data.qvel - expected_next[1])))
            if pos_error > 1e-10 or vel_error > 1e-10:
                raise RuntimeError(f"Exact-state replay failed at branch {branch}: "
                                   f"qpos={pos_error:.3g}, qvel={vel_error:.3g}")
        obs = env.get_observation()
        terminated, truncated = env.check_termination()
        state = measure_recovery_state(obs, float(env.data.qpos[2]), env.default_q)
        in_entry = inside_entry_envelope(state, DEFAULT_RECOVERY_CRITERIA)
        in_hold = inside_hold_envelope(state, DEFAULT_RECOVERY_CRITERIA)
        if not in_entry:
            left_entry = True
        elif left_entry:
            reentered_after_exit = True
        entry_run = entry_run + 1 if in_entry else 0
        if entry_start is None and entry_run >= DEFAULT_RECOVERY_CRITERIA.entry_steps:
            entry_start = elapsed + 1 - DEFAULT_RECOVERY_CRITERIA.entry_steps
        outward_rate = outward_sign * float(obs[62])
        brake_run = brake_run + 1 if outward_rate <= 0 else 0
        if braking_at is None and brake_run >= 3:
            braking_at = elapsed + 1 - 3
        max_joint_speed = max(max_joint_speed, float(np.max(np.abs(obs[29:58]))))
        trace.append({"elapsed": elapsed + 1, "pitch": float(obs[59]),
                      "pitch_rate": float(obs[62]), "outward_rate": outward_rate,
                      "height": state.height, "entry": in_entry, "hold": in_hold,
                      "terminated": bool(terminated), "truncated": bool(truncated)})
        if terminated or truncated:
            break

    # The entry must happen within 2 s. The last 2 s must continuously meet
    # the wider hold envelope. These are two distinct requirements.
    entered_in_window = entry_start is not None and entry_start < RECOVERY_STEPS
    held_last_two_seconds = len(trace) == horizon and all(
        item["hold"] for item in trace[-HOLD_STEPS:]
    )
    return {
        "branch_step": branch, "candidate": candidate["name"],
        "steps": len(trace), "terminated": bool(terminated), "truncated": bool(truncated),
        "initial_pitch": float(start_obs[59]), "initial_rate": start_rate,
        "initial_outward_rate": initial_outward_rate,
        "started_inside_entry": started_inside_entry,
        "reentered_after_exit": reentered_after_exit,
        "braking_step": braking_at, "entry_step": entry_start,
        "entered_within_2s": entered_in_window,
        "held_final_2s": held_last_two_seconds,
        "success": bool(not terminated and not truncated and entered_in_window and held_last_two_seconds),
        "final_pitch": trace[-1]["pitch"], "final_rate": trace[-1]["pitch_rate"],
        "max_action_abs": max_action, "max_joint_speed": max_joint_speed,
        "torque_clip_count": torque_clipped, "torque_near_limit_count": near_limit,
        "foot_contact_speed_p95_m_s": float(np.percentile(slip_speeds, 95)),
        "foot_contact_speed_max_m_s": float(max(slip_speeds)),
        "trace": trace,
    }


def verify_reference(meta: dict, directory: Path, checkpoint: Path) -> None:
    model_ref = meta["model_reference"]
    scene = Path.home() / "robotics/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    for path, expected, label in (
        (scene, model_ref["scene_sha256"], "scene"),
        (scene.with_name("g1_29dof.xml"), model_ref["robot_xml_sha256"], "robot"),
        (checkpoint, model_ref["checkpoint_sha256"], "checkpoint"),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"{label} differs from Phase 1 reference")
    if mujoco.__version__ != model_ref["mujoco_version"]:
        raise RuntimeError("MuJoCo version differs from Phase 1")
    if int(model_ref["state_spec_value"]) != int(STATE_SPEC):
        raise RuntimeError("MuJoCo state specification differs from Phase 1")
    if not meta["initial_integration_state_identical"]:
        raise RuntimeError("Phase 1 policies did not share the initial state")
    for policy in ("zero_pd", "best_ppo"):
        if not (directory / f"{policy}.npz").is_file():
            raise FileNotFoundError(f"Missing Phase 1 array for {policy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=config.CHECKPOINT_DIR / "branch_recovery")
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=(.5, 1.0))
    parser.add_argument("--pulse-steps", nargs="+", type=int, default=(10,))
    parser.add_argument("--policies", nargs="+", choices=("zero_pd", "best_ppo"),
                        default=("zero_pd", "best_ppo"))
    parser.add_argument("--stages", nargs="+", choices=("P_pre_40", "P_pre_20",
                        "A_deviation", "B_between", "C_acceleration", "D_late"),
                        default=("A_deviation", "B_between", "C_acceleration", "D_late"))
    args = parser.parse_args()
    if any(not 0 < a <= 1 for a in args.amplitudes) or any(d < 1 for d in args.pulse_steps):
        parser.error("amplitudes must be in (0,1] and pulse-steps must be positive")
    directory = args.trajectory_dir.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    meta = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    verify_reference(meta, directory, checkpoint)
    original_controllers = {
        "zero_pd": PDStandController(),
        "best_ppo": PPOStandController.from_checkpoint(checkpoint, torch.device("cpu")),
    }
    candidates = make_candidates(tuple(args.amplitudes), tuple(args.pulse_steps))
    output = unique_output(args.output_root)
    records = []
    branch_states = {}
    branch_metadata = []
    horizon = RECOVERY_STEPS + HOLD_STEPS
    print("=== PHASE 2 EXACT-STATE BRANCH RECOVERY ===", flush=True)
    print(f"Recovery: 2 s; final hold: 2 s; candidates: {len(candidates)}", flush=True)
    for policy in args.policies:
        summary = meta["policies"][policy]
        steps = branch_steps(summary)
        with np.load(directory / f"{policy}.npz") as arrays:
            snapshots = arrays["integration_state"]
            qpos = arrays["qpos"]
            qvel = arrays["qvel"]
            for stage in args.stages:
                branch = steps[stage]
                key = f"{policy}_{stage}"
                snapshot = snapshots[branch].copy()
                branch_states[key] = snapshot
                branch_metadata.append({"key": key, "policy": policy, "stage": stage,
                                        "step": branch, "time_s": branch * .02,
                                        "pitch": float(arrays["observation"][branch, 59]),
                                        "pitch_rate": float(arrays["observation"][branch, 62]),
                                        "qpos_sha256": hashlib.sha256(qpos[branch].tobytes()).hexdigest(),
                                        "qvel_sha256": hashlib.sha256(qvel[branch].tobytes()).hexdigest(),
                                        "integration_state_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest()})
                group = []
                for candidate in candidates:
                    result = run_case(snapshot, branch,
                                      float(summary["events"]["quiet_pitch_reference_rad"]),
                                      candidate, original_controllers[policy], horizon,
                                      (qpos[branch + 1], qvel[branch + 1])
                                      if candidate["kind"] == "original" else None)
                    result["policy"] = policy
                    result["stage"] = stage
                    group.append(result)
                    records.append(result)
                original = group[0]
                remaining = int(summary["length"]) - branch
                if original["steps"] != remaining or not original["terminated"]:
                    raise RuntimeError(f"Original replay mismatch at {key}: {original['steps']} vs {remaining}")
                best = [r for r in group if r["success"]]
                braked = sum(r["braking_step"] is not None for r in group)
                print(f"{key:27} pitch={group[0]['initial_pitch']:+.3f} "
                      f"rate={group[0]['initial_rate']:+.3f} "
                      f"original={original['steps']:3d} successes={len(best):2d} "
                      f"braked={braked:2d}/{len(group)}", flush=True)

    np.savez_compressed(output / "branch_states.npz", **branch_states)
    (output / "branch_states.json").write_text(json.dumps(branch_metadata, indent=2), encoding="utf-8")
    fields = [key for key in records[0] if key != "trace"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in records)
    (output / "results.json").write_text(json.dumps({
        "schema_version": 1,
        "source_directory": str(directory),
        "source_summary_sha256": sha256(directory / "summary.json"),
        "criteria": {"recovery_steps": RECOVERY_STEPS, "hold_steps": HOLD_STEPS,
                     "entry_steps": DEFAULT_RECOVERY_CRITERIA.entry_steps,
                     "pulse_steps": args.pulse_steps, "amplitudes": args.amplitudes},
        "branches": branch_metadata,
        "results": records,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} cases: {output}", flush=True)


if __name__ == "__main__":
    main()
