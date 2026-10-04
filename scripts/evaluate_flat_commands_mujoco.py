"""Match the Isaac flat command suite using original MuJoCo actor/PD code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--local-policy", type=Path, required=True)
    parser.add_argument("--comparison-policy", type=Path)
    parser.add_argument("--comparison-label", default="supplied")
    parser.add_argument("--local-label", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30)
    args = parser.parse_args()
    if args.comparison_label == args.local_label:
        parser.error("Actor labels must differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ref = args.reference_root.resolve()
    sys.path.insert(0, str(ref / "mujoco"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import mujoco
    from mujoco_eval import run_grid

    model_path = ref / "mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml"
    metadata = ref / "mujoco/policies/isaac_metadata.json"
    policies = {args.comparison_label: args.comparison_policy.resolve() if args.comparison_policy else ref / "mujoco/policies/baseline/policy_actor.npz",
                args.local_label: args.local_policy.resolve()}
    xml = ET.parse(model_path).getroot()
    meshdir = xml.find("compiler").get("meshdir", "")
    files = [model_path, metadata, Path(__file__).resolve(), *policies.values()]
    files += [model_path.parent / meshdir / mesh.get("file") for mesh in xml.findall("asset/mesh")]
    files += [ref / "mujoco/mujoco_eval" / name for name in ("run_grid.py", "policy.py", "terrains.py")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    cases = []
    for speed in (0.5, 1.0):
        for rate in (0.0, -0.2, 0.2):
            cases.append(dict(mode="rate", speed=speed, target=rate, start_yaw=0.0))
        for target in (0.0, -math.pi / 2, math.pi / 2):
            cases.append(dict(mode="heading", speed=speed, target=target,
                              start_yaw=0.5 if target == 0 else 0.0))
    (args.output_dir / "protocol.json").write_text(json.dumps(dict(
        cases=cases, duration=args.duration, heading_gain=0.5, yaw_limit=1,
        timestep=0.001, friction=0.8, seed=42, terrain="plane", pd_changes=False,
        body="pelvis", policy_input="original 123D assembly; command indices 9:12 checked",
        limitations="Isaac and MuJoCo assets/contact/solver remain different; plane cases are deterministic",
        python=sys.version, numpy=np.__version__, mujoco=mujoco.__version__), indent=2))
    summary = []
    for actor, policy_path in policies.items():
        for case_id, case in enumerate(cases):
            cfg = SimpleNamespace(model=str(model_path), policy=str(policy_path), metadata=str(metadata),
                command_x=case["speed"], command_y=0., command_yaw=case["target"] if case["mode"] == "rate" else 0.,
                timestep=.001, duration=args.duration, seed=42, device="cpu", allow_missing_joints=False,
                initial_base_height=.74, min_base_height=.35, base_body="pelvis", control_mode="pd")
            initialized = False
            current_data = None
            rows = []
            original_forward = mujoco.mj_forward
            original_obs = run_grid.build_policy_observation

            def forward(model, data, *values, **kwargs):
                nonlocal initialized, current_data
                if not initialized:
                    data.qpos[3:7] = [math.cos(case["start_yaw"] / 2), 0, 0, math.sin(case["start_yaw"] / 2)]
                    initialized = True
                    current_data = data
                    snap = dict(qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                        compiled_arrays_sha256={name: hashlib.sha256(getattr(model, name).tobytes()).hexdigest()
                            for name in ("body_mass", "body_inertia", "dof_armature", "geom_friction", "actuator_ctrlrange")})
                    (args.output_dir / f"{actor}_{case_id:02d}_initial.json").write_text(json.dumps(snap, indent=2))
                return original_forward(model, data, *values, **kwargs)

            def observation(quat, lin_vel, ang_vel, command, *values, **kwargs):
                w, x, y, z = quat
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                yaw_error = float(np.arctan2(np.sin(case["target"] - yaw), np.cos(case["target"] - yaw)))
                yaw_command = float(np.clip(.5 * yaw_error, -1, 1)) if case["mode"] == "heading" else case["target"]
                command[:] = [case["speed"], 0, yaw_command]
                cfg.command_yaw = yaw_command
                obs = original_obs(quat, lin_vel, ang_vel, command, *values, **kwargs)
                if not np.array_equal(obs[9:12], command.astype(obs.dtype)):
                    raise RuntimeError("Policy input command mismatch")
                body_v = run_grid.quat_rotate_inverse(quat, lin_vel)
                body_w = run_grid.quat_rotate_inverse(quat, ang_vel)
                gravity = run_grid.quat_rotate_inverse(quat, np.array([0., 0., -1.]))
                rows.append(dict(time_s=float(current_data.time), heading=yaw,
                    heading_error=yaw_error, body_vx=float(body_v[0]), body_wz=float(body_w[2]),
                    world_wz=float(ang_vel[2]), command_vx=case["speed"], command_wz=yaw_command,
                    tilt_deg=float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1)))),
                    world_x=float(current_data.qpos[0]), world_y=float(current_data.qpos[1])))
                return obs

            with patch.object(mujoco, "mj_forward", forward), patch.object(run_grid, "build_policy_observation", observation):
                metrics = run_grid.run_once(cfg, "plane", .8)
            if not rows or abs(rows[0]["heading"] - case["start_yaw"]) > 1e-6:
                raise RuntimeError("Initial heading mismatch")
            trace_path = args.output_dir / f"{actor}_{case_id:02d}.csv"
            with trace_path.open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            tail = [r for r in rows if r["time_s"] >= min(2, args.duration / 2)]
            row = dict(actor=actor, case_id=case_id, **case,
                       elapsed_s=metrics["elapsed_s"], fall=metrics["fell"],
                       mean_body_vx=float(np.mean([r["body_vx"] for r in tail])),
                       forward_rmse=float(np.sqrt(np.mean([(r["body_vx"] - r["command_vx"]) ** 2 for r in tail]))),
                       body_yaw_rmse=float(np.sqrt(np.mean([(r["body_wz"] - r["command_wz"]) ** 2 for r in tail]))),
                       world_yaw_rmse=float(np.sqrt(np.mean([(r["world_wz"] - r["command_wz"]) ** 2 for r in tail]))),
                       mean_world_wz=float(np.mean([r["world_wz"] for r in tail])),
                       final_heading_error_abs=abs(rows[-1]["heading_error"]) if case["mode"] == "heading" else None,
                       max_tilt_deg=max(r["tilt_deg"] for r in tail))
            (args.output_dir / f"{actor}_{case_id:02d}.json").write_text(json.dumps(dict(summary=row, reference_metrics=metrics), indent=2))
            summary.append(row)
            with (args.output_dir / "summary.csv").open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(summary)
            print("MUJOCO_FLAT_CASE " + json.dumps(row), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("Source changed")
    (args.output_dir / "complete.json").write_text(json.dumps(dict(cases=len(summary), inputs_unchanged=True)))


if __name__ == "__main__":
    main()
