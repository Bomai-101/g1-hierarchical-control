"""Isolated ankle force-limit experiment; preserve source models and policy/PD."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_root / "mujoco"))
    from mujoco_eval import run_grid
    metadata_path = args.reference_root / "mujoco/policies/isaac_metadata.json"
    model_path = args.reference_root / "mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml"
    metadata = json.loads(metadata_path.read_text())
    inputs = [metadata_path, model_path, args.policy, Path(__file__).resolve(),
              args.reference_root / "mujoco/mujoco_eval/run_grid.py", args.reference_root / "mujoco/mujoco_eval/policy.py"]
    hashes = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.output / "inputs.json").write_text(json.dumps(hashes, indent=2))
    results = []
    for limit in (50., 20.):
        for yaw in (0., -.2, .2):
            stem = f"limit{limit:g}_yaw{yaw:+.1f}"
            cfg = SimpleNamespace(model=str(model_path), policy=str(args.policy.resolve()), metadata=str(metadata_path),
                command_x=1., command_y=0., command_yaw=yaw, timestep=.001, duration=30., seed=42,
                device="cpu", allow_missing_joints=False, initial_base_height=.74, min_base_height=.35,
                base_body="pelvis", control_mode="pd")
            original_forward, original_step = mujoco.mj_forward, mujoco.mj_step
            initialized = False
            joint_ids = qvel_addresses = ankle_positions = None
            physical_steps = 0
            over20 = np.zeros(37, dtype=int)
            force_sum = np.zeros(37)
            times, states, commands = [], [], []
            changes = []
            def forward(model, data, *values, **kwargs):
                nonlocal initialized, joint_ids, qvel_addresses, ankle_positions
                if not initialized:
                    joint_ids = np.array([run_grid.resolve_joint_id(model, name)[0] for name in metadata["joint_names"]])
                    if np.any(joint_ids < 0) or len(set(joint_ids)) != 37:
                        raise RuntimeError("Joint mapping is incomplete or not one-to-one")
                    qvel_addresses = model.jnt_dofadr[joint_ids]
                    ankle_positions = np.array([i for i,name in enumerate(metadata["joint_names"]) if "ankle" in name])
                    before = model.jnt_actfrcrange.copy()
                    for index in ankle_positions:
                        jid = joint_ids[index]
                        if not model.jnt_actfrclimited[jid] or not np.array_equal(before[jid], [-50.,50.]):
                            raise RuntimeError("Unexpected source ankle force limit")
                        model.jnt_actfrcrange[jid] = [-limit, limit]
                        changes.append(dict(joint=metadata["joint_names"][index], original=before[jid].tolist(), effective=[-limit,limit]))
                    changed = np.flatnonzero(np.any(before != model.jnt_actfrcrange,axis=1))
                    expected = joint_ids[ankle_positions] if limit == 20 else np.array([],dtype=int)
                    if set(changed) != set(expected):
                        raise RuntimeError("Unexpected parameter change")
                    initialized = True
                result = original_forward(model, data, *values, **kwargs)
                if not states:
                    states.append(np.r_[data.qpos,data.qvel].copy()); times.append(float(data.time))
                    mujoco.mj_saveModel(model,str(args.output / f"{stem}.mjb"))
                return result
            def step(model, data, *values, **kwargs):
                nonlocal physical_steps
                result = original_step(model, data, *values, **kwargs)
                force = np.abs(data.qfrc_actuator[qvel_addresses])
                over20[:] += force > 20. + 1e-7
                force_sum[:] += force
                physical_steps += 1
                if physical_steps % 20 == 0:
                    states.append(np.r_[data.qpos,data.qvel].copy()); times.append(float(data.time))
                return result
            original_act = run_grid.TorchActorPolicy.act
            def act(actor, obs):
                if not np.array_equal(obs[9:12], np.array([1.,0.,yaw], dtype=obs.dtype)):
                    raise RuntimeError("Command mismatch")
                commands.append(obs[9:12].copy())
                return original_act(actor, obs)
            with patch.object(mujoco,"mj_forward",forward), patch.object(mujoco,"mj_step",step), patch.object(run_grid.TorchActorPolicy,"act",act):
                metrics = run_grid.run_once(cfg,"plane",.8)
            np.savez_compressed(args.output / f"{stem}.npz", time=times,states=states,commands=commands)
            row = dict(ankle_limit_nm=limit,yaw_command=yaw,metrics=metrics,physical_steps=physical_steps,
                ankle_force_over20_fraction={metadata["joint_names"][i]:float(over20[i]/physical_steps) for i in ankle_positions},
                ankle_mean_abs_force={metadata["joint_names"][i]:float(force_sum[i]/physical_steps) for i in ankle_positions},
                model_changes=changes)
            (args.output / f"{stem}.json").write_text(json.dumps(row,indent=2))
            results.append(row)
            print("ANKLE_PROBE",limit,yaw,metrics["elapsed_s"],metrics["fell"],metrics["yaw_rate_tracking_rmse_rps"],flush=True)
    for path,expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source input changed")
    (args.output / "complete.json").write_text(json.dumps(dict(results=results,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__),indent=2))


if __name__ == "__main__":
    main()
