"""Record matched zero-PD and deterministic PPO trajectories through termination.

Phase 1 diagnostic only. The sampled integration states are saved at policy
boundaries so later branching can restore identical simulation inputs.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.config import balance as config
from g1_control.controllers.residual import PDStandController, PPOStandController
from g1_control.envs.balance_env import G1Env


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def foot_geoms(model: mujoco.MjModel) -> tuple[int, dict[str, set[int]], dict[str, int]]:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise RuntimeError("Cannot find named floor geom")
    sides = {}
    bodies = {}
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        if bid < 0:
            raise RuntimeError(f"Cannot find {side} ankle-roll body")
        bodies[side] = bid
        sides[side] = {
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) == bid
            and int(model.geom_contype[gid]) != 0
            and int(model.geom_conaffinity[gid]) != 0
        }
        if len(sides[side]) != 4:
            raise RuntimeError(f"Expected four colliding contact geoms on {side} foot")
    return floor, sides, bodies


def contact_velocity_xy(model: mujoco.MjModel, data: mujoco.MjData, body: int, pos) -> float:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, pos, body)
    return float(np.linalg.norm((jacp @ data.qvel)[:2]))


def new_contact_accumulator() -> dict:
    return {
        side: {"count": 0, "physics_with_contact": 0, "normal_force_sum": 0.0,
               "tangent_force_sum": 0.0, "velocity_xy": [], "positions": []}
        for side in ("left", "right")
    }


def read_contacts(model, data, floor, geoms, bodies, accumulator) -> None:
    seen = set()
    for cid in range(data.ncon):
        contact = data.contact[cid]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor not in pair:
            continue
        other = next(iter(pair - {floor}), -1)
        for side, ids in geoms.items():
            if other not in ids:
                continue
            seen.add(side)
            record = accumulator[side]
            record["count"] += 1
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, cid, wrench)
            record["normal_force_sum"] += abs(float(wrench[0]))
            record["tangent_force_sum"] += float(np.linalg.norm(wrench[1:3]))
            record["velocity_xy"].append(contact_velocity_xy(model, data, bodies[side], contact.pos))
            record["positions"].append(np.array(contact.pos).copy())
    for side in seen:
        accumulator[side]["physics_with_contact"] += 1


def summarize_contacts(accumulator: dict, decimation: int) -> dict:
    result = {}
    for side, raw in accumulator.items():
        positions = np.asarray(raw["positions"], dtype=np.float64)
        velocities = raw["velocity_xy"]
        result[side] = {
            "contact_count": raw["count"],
            "contact_occupancy": raw["physics_with_contact"] / decimation,
            "mean_normal_force_N": raw["normal_force_sum"] / decimation,
            "mean_tangential_force_N": raw["tangent_force_sum"] / decimation,
            "contact_velocity_xy_p95_m_s": float(np.percentile(velocities, 95)) if velocities else None,
            "contact_velocity_xy_max_m_s": float(max(velocities)) if velocities else None,
            "contact_centroid_world_m": positions.mean(axis=0).tolist() if len(positions) else None,
        }
    return result


def integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def metrics(row: dict) -> dict:
    keys = (
        "step", "time_s", "roll_rad", "pitch_rad", "roll_rate_rad_s", "pitch_rate_rad_s",
        "height_m", "base_vx_m_s", "base_vy_m_s", "gyro_norm_rad_s",
        "action_norm", "reward", "torque_clip_count", "torque_near_limit_count",
        "left_contact_occupancy", "right_contact_occupancy",
        "left_normal_force_N", "right_normal_force_N",
        "left_contact_velocity_p95_m_s", "right_contact_velocity_p95_m_s",
    )
    return {key: row[key] for key in keys}


def record_one(label: str, controller, model_reference: dict) -> tuple[dict, dict]:
    env = G1Env()
    model, data = env.model, env.data
    floor, geoms, bodies = foot_geoms(model)
    observation = env.reset()
    initial_state = integration_state(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    steps = []
    boundary_states = [initial_state]
    qpos = [initial_qpos]
    qvel = [initial_qvel]
    actions = []
    target_q = []
    raw_torque = []
    applied_torque = []
    total_reward = 0.0
    terminated = truncated = False

    for step in range(1, env.max_episode_steps + 1):
        output = controller.act(observation)
        action = output.action.astype(np.float32, copy=True)
        target = env.action_to_target_q(action)
        contacts = new_contact_accumulator()
        physics_raw = []
        physics_applied = []
        for _ in range(env.decimation):
            q, dq = env.get_joint_state()
            raw = env.kp * (target - q) - env.kd * dq
            applied = env.compute_pd_torque(target)
            physics_raw.append(raw.copy())
            physics_applied.append(applied.copy())
            data.ctrl[:] = applied
            mujoco.mj_step(model, data)
            read_contacts(model, data, floor, geoms, bodies, contacts)

        env.episode_step += 1
        observation = env.get_observation()
        reward = env.compute_reward(action)
        terminated, truncated = env.check_termination()
        if terminated:
            reward -= 5.0
        total_reward += reward
        contact = summarize_contacts(contacts, env.decimation)
        raw_arr = np.asarray(physics_raw)
        applied_arr = np.asarray(physics_applied)
        bounds = env.torque_limits
        clipped = (raw_arr < bounds[:, 0] - 1e-7) | (raw_arr > bounds[:, 1] + 1e-7)
        near = np.abs(applied_arr) >= 0.95 * np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1]))
        row = {
            "step": step,
            "time_s": float(step * model.opt.timestep * env.decimation),
            "roll_rad": float(observation[58]),
            "pitch_rad": float(observation[59]),
            "yaw_rad": float(observation[60]),
            "roll_rate_rad_s": float(observation[61]),
            "pitch_rate_rad_s": float(observation[62]),
            "yaw_rate_rad_s": float(observation[63]),
            "gyro_norm_rad_s": float(np.linalg.norm(observation[61:64])),
            "height_m": float(data.qpos[2]),
            "base_vx_m_s": float(data.qvel[0]),
            "base_vy_m_s": float(data.qvel[1]),
            "base_vz_m_s": float(data.qvel[2]),
            "com_world_m": data.subtree_com[1].tolist(),
            "left_foot_world_m": data.xpos[bodies["left"]].tolist(),
            "right_foot_world_m": data.xpos[bodies["right"]].tolist(),
            "action_norm": output.action_l2_norm,
            "reward": reward,
            "torque_clip_count": int(clipped.sum()),
            "torque_near_limit_count": int(near.sum()),
            "left_contact_occupancy": contact["left"]["contact_occupancy"],
            "right_contact_occupancy": contact["right"]["contact_occupancy"],
            "left_normal_force_N": contact["left"]["mean_normal_force_N"],
            "right_normal_force_N": contact["right"]["mean_normal_force_N"],
            "left_contact_velocity_p95_m_s": contact["left"]["contact_velocity_xy_p95_m_s"],
            "right_contact_velocity_p95_m_s": contact["right"]["contact_velocity_xy_p95_m_s"],
            "contact": contact,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        steps.append(row)
        boundary_states.append(integration_state(model, data))
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        actions.append(action)
        target_q.append(target.copy())
        raw_torque.append(raw_arr)
        applied_torque.append(applied_arr)
        if terminated or truncated:
            break

    arrays = {
        "integration_state": np.stack(boundary_states),
        "qpos": np.stack(qpos),
        "qvel": np.stack(qvel),
        "action": np.stack(actions),
        "target_q": np.stack(target_q),
        "raw_pd_torque": np.stack(raw_torque),
        "applied_pd_torque": np.stack(applied_torque),
        "observation": np.stack([np.concatenate((q[7:36], v[6:35],
                          env.quaternion_to_rpy(q[3:7]), v[3:6]))
                          for q, v in zip(qpos, qvel, strict=True)]),
    }
    summary = {
        "policy": label,
        "length": len(steps),
        "duration_s_after_settling": len(steps) * model.opt.timestep * env.decimation,
        "return": float(total_reward),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "max_abs_pitch_rad": max(abs(row["pitch_rad"]) for row in steps),
        "max_abs_pitch_rate_rad_s": max(abs(row["pitch_rate_rad_s"]) for row in steps),
        "min_height_m": min(row["height_m"] for row in steps),
        "torque_clip_count": sum(row["torque_clip_count"] for row in steps),
        "torque_near_limit_count": sum(row["torque_near_limit_count"] for row in steps),
        "max_action_norm": max(row["action_norm"] for row in steps),
        "initial_qpos_sha256": hashlib.sha256(initial_qpos.tobytes()).hexdigest(),
        "initial_qvel_sha256": hashlib.sha256(initial_qvel.tobytes()).hexdigest(),
        "model_reference": model_reference,
    }
    return {"summary": summary, "steps": steps}, arrays


def first_sustained(rows: list[dict], predicate, duration: int) -> int | None:
    run_start = None
    run_length = 0
    for row in rows:
        if predicate(row):
            if run_start is None:
                run_start = row["step"]
            run_length += 1
            if run_length >= duration:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def event_markers(rows: list[dict]) -> dict:
    # The early 0.4-1.6 s interval estimates this trajectory's quiet pitch.
    # These are reproducible operational thresholds, not a recoverability law.
    quiet = [r["pitch_rad"] for r in rows if 20 <= r["step"] <= 80]
    reference = float(np.median(quiet)) if quiet else float(rows[0]["pitch_rad"])
    dt = float(rows[0]["time_s"])
    derivatives = [0.0] + [
        (rows[i]["pitch_rate_rad_s"] - rows[i - 1]["pitch_rate_rad_s"]) / dt
        for i in range(1, len(rows))
    ]

    def deviation(r):
        error = r["pitch_rad"] - reference
        return abs(error) >= 0.015 and error * r["pitch_rate_rad_s"] > 0

    for i, row in enumerate(rows):
        error = row["pitch_rad"] - reference
        outward_acceleration = np.sign(error) * float(np.mean(derivatives[max(0, i - 4):i + 1]))
        row["outward_pitch_accel_5step_rad_s2"] = outward_acceleration

    def acceleration(r):
        error = r["pitch_rad"] - reference
        return (
            abs(error) >= 0.03
            and np.sign(error) * r["pitch_rate_rad_s"] >= 0.15
            and r["outward_pitch_accel_5step_rad_s2"] >= 0.20
        )

    deviation_step = first_sustained(rows, deviation, 10)
    acceleration_step = first_sustained(rows, acceleration, 5)
    termination_step = next((r["step"] for r in rows if r["terminated"]), None)
    return {
        "quiet_pitch_reference_rad": reference,
        "T_deviation_step": deviation_step,
        "T_acceleration_step": acceleration_step,
        "T_termination_step": termination_step,
        "definitions": {
            "T_deviation": "|pitch-reference| >= 0.015 rad and pitch error * pitch rate > 0 for 10 consecutive policy steps; marker is first step of run",
            "T_acceleration": "|pitch-reference| >= 0.03 rad, outward pitch rate >= 0.15 rad/s and outward five-step mean angular acceleration >= 0.20 rad/s^2 for 5 consecutive policy steps; marker is first step of run",
            "T_termination": "first step triggering unchanged G1Env.check_termination()",
        },
    }


def output_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("phase1_%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        path = root / (prefix if suffix == 0 else f"{prefix}_{suffix:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Could not create unique trajectory directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=config.CHECKPOINT_DIR / "instability_trajectories")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("obs_dim", -1)) != config.OBS_DIM or int(payload.get("action_dim", -1)) != config.ACTION_DIM:
        raise ValueError("Checkpoint observation/action dimensions do not match frozen Day9")
    if int(payload.get("update", -1)) != 1:
        raise ValueError("Expected the historical update_001 deterministic checkpoint")
    checkpoint_config = payload.get("config", {})
    if abs(float(checkpoint_config.get("current_log_std_mean", checkpoint_config.get("log_std_mean", 0))) + 3.0) > 1e-6:
        raise ValueError("Checkpoint is not the frozen Day9 log_std=-3 policy")

    scene = Path.home() / "robotics/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    robot = scene.with_name("g1_29dof.xml")
    model_reference = {
        "mujoco_version": mujoco.__version__,
        "scene_sha256": sha256(scene),
        "robot_xml_sha256": sha256(robot),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_update": int(payload["update"]),
        "checkpoint_total_env_steps": int(payload.get("total_env_steps", -1)),
        "state_spec": "mjSTATE_INTEGRATION",
        "state_spec_value": int(STATE_SPEC),
        "array_alignment": "qpos/qvel/integration_state/observation index 0 is after reset settling; index k is after policy step k; actions/targets/torques index k-1 produced state k",
    }
    phase0_path = config.CHECKPOINT_DIR / "environment_audits" / "phase0_environment_audit.json"
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))["static"]
    for key in ("scene_sha256", "robot_xml_sha256", "mujoco_version"):
        if phase0[key] != model_reference[key]:
            raise RuntimeError(f"Environment Reference V1 mismatch: {key}")
    model_reference["phase0_audit_sha256"] = sha256(phase0_path)
    controller = PPOStandController.from_checkpoint(checkpoint, torch.device("cpu"))
    output = output_directory(args.output_root)
    summaries = {}
    initial_state_hashes = []
    for label, policy in (("zero_pd", PDStandController()), ("best_ppo", controller)):
        result, arrays = record_one(label, policy, model_reference)
        events = event_markers(result["steps"])
        result["summary"]["events"] = events
        summaries[label] = result["summary"]
        initial_state_hashes.append(arrays["integration_state"][0].tobytes())
        np.savez_compressed(output / f"{label}.npz", **arrays)
        (output / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (output / f"{label}_timeline.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(metrics(result["steps"][0])))
            writer.writeheader()
            writer.writerows(metrics(row) for row in result["steps"])
        print(f"{label}: {result['summary']['length']} steps; return={result['summary']['return']:.2f}; events={events}")

    if initial_state_hashes[0] != initial_state_hashes[1]:
        raise RuntimeError("Zero and PPO trajectories did not start from identical MuJoCo integration state")
    if summaries["zero_pd"]["length"] != 215 or summaries["best_ppo"]["length"] != 241:
        raise RuntimeError("Historical 215/241-step reference was not reproduced")
    (output / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_integration_state_identical": True,
        "model_reference": model_reference,
        "policies": summaries,
    }, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
