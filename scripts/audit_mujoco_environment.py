"""Read-only Phase 0 audit of the exact MuJoCo G1 standing environment.

Writes only a JSON report to the already ignored checkpoint tree. It does not
change the model, standing pose, controller, reward, or termination rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.config import balance as config
from g1_control.envs.balance_env import G1Env


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geom_record(model: mujoco.MjModel, geom_id: int) -> dict:
    body_id = int(model.geom_bodyid[geom_id])
    return {
        "id": geom_id,
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "type": mujoco.mjtGeom(int(model.geom_type[geom_id])).name,
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
        "condim": int(model.geom_condim[geom_id]),
        "friction": model.geom_friction[geom_id].tolist(),
        "size": model.geom_size[geom_id].tolist(),
    }


def body_has_ancestor(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def physics_and_model(env: G1Env, scene: Path, robot: Path) -> tuple[dict, int, dict[str, set[int]]]:
    model = env.model
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise RuntimeError("The scene has no named floor geom")

    feet: dict[str, set[int]] = {}
    for side in ("left", "right"):
        ankle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
        )
        if ankle_id < 0:
            raise RuntimeError(f"Missing {side} ankle roll body")
        feet[side] = {
            gid for gid in range(model.ngeom)
            if body_has_ancestor(model, int(model.geom_bodyid[gid]), ankle_id)
            and int(model.geom_contype[gid]) != 0
            and int(model.geom_conaffinity[gid]) != 0
        }

    option = model.opt
    option_record = {
        "timestep_s": float(option.timestep),
        "integrator": mujoco.mjtIntegrator(int(option.integrator)).name,
        "solver": mujoco.mjtSolver(int(option.solver)).name,
        "iterations": int(option.iterations),
        "tolerance": float(option.tolerance),
        "gravity_m_s2": option.gravity.tolist(),
        "disableflags": int(option.disableflags),
        "enableflags": int(option.enableflags),
        "policy_decimation": int(env.decimation),
        "policy_dt_s": float(option.timestep * env.decimation),
    }

    joints = []
    for jid in range(model.njnt):
        joints.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            "damping": float(model.dof_damping[int(model.jnt_dofadr[jid])]),
            "armature": float(model.dof_armature[int(model.jnt_dofadr[jid])]),
            "frictionloss": float(model.dof_frictionloss[int(model.jnt_dofadr[jid])]),
            "limited": bool(model.jnt_limited[jid]),
            "range": model.jnt_range[jid].tolist(),
            "actuator_force_limited_at_joint": bool(model.jnt_actfrclimited[jid]),
            "joint_actuator_force_range": model.jnt_actfrcrange[jid].tolist(),
        })

    actuators = []
    for aid in range(model.nu):
        actuators.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
            "ctrl_limited": bool(model.actuator_ctrllimited[aid]),
            "ctrlrange": model.actuator_ctrlrange[aid].tolist(),
            "force_limited": bool(model.actuator_forcelimited[aid]),
            "forcerange": model.actuator_forcerange[aid].tolist(),
        })

    roots = (ET.parse(scene).getroot(), ET.parse(robot).getroot())
    xml_support = []
    xml_contact_pairs = []
    for source, root in zip((scene, robot), roots, strict=True):
        for element in root.iter():
            if element.tag in {"weld", "connect", "equality", "tendon", "spatial", "fixed"}:
                xml_support.append({"source": source.name, "tag": element.tag, "attributes": element.attrib})
            if element.tag == "pair":
                xml_contact_pairs.append({"source": source.name, "attributes": element.attrib})
            if element.tag == "body" and element.get("mocap") == "true":
                xml_support.append({"source": source.name, "tag": "mocap_body", "attributes": element.attrib})

    report = {
        "mujoco_version": mujoco.__version__,
        "scene": str(scene),
        "scene_sha256": digest(scene),
        "robot_xml": str(robot),
        "robot_xml_sha256": digest(robot),
        "model_dims": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "ngeom": model.ngeom, "npair": model.npair, "neq": model.neq},
        "physics": option_record,
        "floor": geom_record(model, floor_id),
        "feet": {side: [geom_record(model, gid) for gid in sorted(ids)] for side, ids in feet.items()},
        "floor_foot_collision_allowed": {
            side: [
                bool((int(model.geom_contype[floor_id]) & int(model.geom_conaffinity[gid]))
                     or (int(model.geom_contype[gid]) & int(model.geom_conaffinity[floor_id])))
                for gid in sorted(ids)
            ]
            for side, ids in feet.items()
        },
        "explicit_contact_pairs": xml_contact_pairs,
        "joint_passive_and_limits": joints,
        "actuators": actuators,
        "external_support_xml": xml_support,
        "external_support_runtime": {
            "equality_constraints": int(model.neq),
            "tendons": int(model.ntendon),
            "mocap_bodies": int(model.nmocap),
            "applied_force_nonzero_at_reset": bool(np.any(env.data.xfrc_applied)),
        },
        "controller": {
            "pose": {"hip": config.HIP_PITCH, "knee": config.KNEE, "ankle": config.ANKLE_PITCH},
            "action_scale": float(env.action_scale),
            "kp": env.kp.tolist(), "kd": env.kd.tolist(),
            "torque_clip_uses_ctrlrange": True,
        },
    }
    return report, floor_id, feet


def contact_point_speed(env: G1Env, body_id: int, position: np.ndarray) -> float:
    model, data = env.model, env.data
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, position, body_id)
    return float(np.linalg.norm((jacp @ data.qvel)[:2]))


def run_zero_policy(env: G1Env, floor_id: int, feet: dict[str, set[int]]) -> dict:
    model, data = env.model, env.data
    env.reset()
    contact_counts = Counter()
    foot_contact_steps = Counter()
    contact_speed: dict[str, list[float]] = {side: [] for side in feet}
    tangential_to_normal: list[float] = []
    raw_torque_max = np.zeros(model.nu)
    clip_counts = np.zeros(model.nu, dtype=np.int64)
    near_limit_counts = np.zeros(model.nu, dtype=np.int64)
    physics_steps = 0
    samples: list[dict] = []
    terminated = truncated = False

    for policy_step in range(env.max_episode_steps):
        target = env.default_q
        for _ in range(env.decimation):
            q, dq = env.get_joint_state()
            raw = env.kp * (target - q) - env.kd * dq
            bounds = env.torque_limits
            raw_torque_max = np.maximum(raw_torque_max, np.abs(raw))
            clip_counts += ((raw < bounds[:, 0] - 1e-7) | (raw > bounds[:, 1] + 1e-7)).astype(np.int64)
            actual = np.clip(raw, bounds[:, 0], bounds[:, 1])
            near_limit_counts += (np.abs(actual) >= 0.95 * np.maximum(np.abs(bounds[:, 0]), np.abs(bounds[:, 1]))).astype(np.int64)
            data.ctrl[:] = actual
            mujoco.mj_step(model, data)
            physics_steps += 1
            sides_seen = set()
            for cid in range(data.ncon):
                contact = data.contact[cid]
                pair = {int(contact.geom1), int(contact.geom2)}
                if floor_id not in pair:
                    continue
                other = next(iter(pair - {floor_id}), floor_id)
                for side, ids in feet.items():
                    if other not in ids:
                        continue
                    sides_seen.add(side)
                    contact_counts[side] += 1
                    body_id = int(model.geom_bodyid[other])
                    contact_speed[side].append(contact_point_speed(env, body_id, contact.pos))
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, cid, wrench)
                    if abs(wrench[0]) > 1e-6:
                        tangential_to_normal.append(float(np.linalg.norm(wrench[1:3]) / abs(wrench[0])))
            for side in sides_seen:
                foot_contact_steps[side] += 1

        env.episode_step += 1
        observation = env.get_observation()
        terminated, truncated = env.check_termination()
        samples.append({
            "step": env.episode_step,
            "time_s_after_settling": round(env.episode_step * model.opt.timestep * env.decimation, 6),
            "pitch_rad": float(observation[59]),
            "pitch_rate_rad_s": float(observation[62]),
            "roll_rad": float(observation[58]),
            "height_m": float(data.qpos[2]),
            "base_vxy_m_s": data.qvel[:2].tolist(),
            "ncon": int(data.ncon),
        })
        if terminated or truncated:
            break

    def stats(values: list[float]) -> dict:
        return {
            "count": len(values),
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": float(np.max(values)) if values else None,
        }

    return {
        "policy_steps": len(samples), "physics_steps": physics_steps,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "duration_s_after_settling": float(len(samples) * model.opt.timestep * env.decimation),
        "contact_events": dict(contact_counts),
        "foot_contact_physics_steps": dict(foot_contact_steps),
        "foot_contact_velocity_xy_m_s": {side: stats(values) for side, values in contact_speed.items()},
        "contact_tangential_to_normal_force_ratio": stats(tangential_to_normal),
        "raw_pd_torque_abs_max": raw_torque_max.tolist(),
        "torque_clip_count": clip_counts.tolist(),
        "torque_near_limit_count": near_limit_counts.tolist(),
        "torque_clip_fraction_any": float(np.sum(clip_counts) / (physics_steps * model.nu)),
        "torque_near_limit_fraction_any": float(np.sum(near_limit_counts) / (physics_steps * model.nu)),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.CHECKPOINT_DIR / "environment_audits" / "phase0_environment_audit.json")
    args = parser.parse_args()
    scene = Path.home() / "robotics/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    robot = scene.with_name("g1_29dof.xml")
    env = G1Env()
    static, floor_id, feet = physics_and_model(env, scene, robot)
    data = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "static": static,
        "zero_policy_run": run_zero_policy(env, floor_id, feet),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    run = data["zero_policy_run"]
    print(f"MuJoCo {mujoco.__version__}; dt={env.model.opt.timestep}; gravity={env.model.opt.gravity.tolist()}")
    print(f"floor-foot contact enabled: {static['floor_foot_collision_allowed']}")
    print(f"zero policy: {run['policy_steps']} steps; terminated={run['terminated']}")
    print(f"contact events: {run['contact_events']}; contact physics steps: {run['foot_contact_physics_steps']}")
    print(f"foot horizontal contact speed: {run['foot_contact_velocity_xy_m_s']}")
    print(f"PD torque clipping fraction: {run['torque_clip_fraction_any']:.6f}")
    print(f"JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
