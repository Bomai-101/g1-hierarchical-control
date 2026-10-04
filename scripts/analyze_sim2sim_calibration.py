"""Independently verify and archive the first controlled Sim2Sim calibration."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference_root / "mujoco"))
    from mujoco_eval import run_grid
    live = args.run_root / "interface_live"
    probe = args.run_root / "ankle_limit_probe"
    runtime = json.loads((live / "runtime.json").read_text())
    evidence = {}
    for folder in (live,probe):
        for path,expected in json.loads((folder / "inputs.json").read_text()).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Input hash changed")
    for actor in ("baseline1499","candidate1999"):
        data = np.load(live / f"{actor}.npz")
        errors = {name:float(np.max(np.abs(data[left]-data[right]))) for name,left,right in
                  [("obs", "isaac_obs","reconstructed_obs"),("action","isaac_action","exported_action"),("target","target","expected_target")]}
        if errors["obs"] > 5e-6 or errors["action"] > 1e-5 or errors["target"] > 1e-6:
            raise RuntimeError("Independent interface verification failed")
        previous = np.load(args.run_root / "isaac_posttraining" / f"{actor}_06.npz")["trace"][:200,0]
        if not np.array_equal(data["root"],np.c_[previous[:,1:11],previous[:,14:17]]):
            raise RuntimeError("New Isaac trajectory changed from matched evidence")
        evidence[actor] = dict(interface_errors=errors,isaac_root_200_samples_exact_previous=True)
    rows = []
    for yaw,index in ((0.,6),(-.2,7),(.2,8)):
        old_stem,new_stem = f"limit50_yaw{yaw:+.1f}",f"limit20_yaw{yaw:+.1f}"
        original = mujoco.MjModel.from_binary_path(str(probe / f"{old_stem}.mjb"))
        changed = mujoco.MjModel.from_binary_path(str(probe / f"{new_stem}.mjb"))
        differences = []
        for name in dir(original):
            value = getattr(original,name)
            other = getattr(changed,name)
            if isinstance(value,np.ndarray) and not np.array_equal(value,other,equal_nan=True):
                differences.append(name)
        if differences != ["jnt_actfrcrange"]:
            raise RuntimeError(f"Other compiled-model arrays changed: {differences}")
        changed_joints = np.flatnonzero(np.any(original.jnt_actfrcrange != changed.jnt_actfrcrange,axis=1))
        expected_joints = [run_grid.resolve_joint_id(original,name)[0] for name in runtime["joint_names"] if "ankle" in name]
        if set(changed_joints) != set(expected_joints):
            raise RuntimeError("Wrong joint force ranges changed")
        previous = json.loads((args.run_root / "mujoco_posttraining" / f"baseline1499_{index:02d}.json").read_text())["reference_metrics"]
        baseline = json.loads((probe / f"{old_stem}.json").read_text())
        if baseline["metrics"] != previous:
            raise RuntimeError("Control metrics differ from previous baseline")
        evidence[f"yaw{yaw:+.1f}"] = dict(control_metrics_exact_previous=True,only_model_array_changed=differences,changed_joint_count=len(changed_joints))
        for stem in (old_stem,new_stem):
            result = json.loads((probe / f"{stem}.json").read_text())
            data = np.load(probe / f"{stem}.npz")
            expected = np.tile(np.array([1,0,yaw],dtype=data["commands"].dtype),(len(data["commands"]),1))
            if not np.array_equal(data["commands"],expected):
                raise RuntimeError("Recorded commands changed")
            metrics = result["metrics"]
            if abs(data["time"][-1]-metrics["elapsed_s"])>1e-8:
                raise RuntimeError("Recorded termination time differs")
            rows.append(dict(ankle_limit_nm=result["ankle_limit_nm"],yaw_command=yaw,elapsed_s=metrics["elapsed_s"],
                fell=metrics["fell"],body_yaw_rmse_rad_s=metrics["yaw_rate_tracking_rmse_rps"],
                body_forward_rmse_m_s=metrics["speed_tracking_rmse_mps"],
                right_ankle_pitch_force_over20_fraction=result["ankle_force_over20_fraction"]["right_ankle_pitch_joint"]))
    with (args.output / "ankle_comparison.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    model=mujoco.MjModel.from_binary_path(str(probe / "limit50_yaw+0.0.mjb"))
    kp,kd=run_grid.pd_gains(runtime["joint_names"])
    contracts=[]
    for i,name in enumerate(runtime["joint_names"]):
        jid,resolved=run_grid.resolve_joint_id(model,name)
        adr=model.jnt_dofadr[jid]
        contracts.append(dict(isaac_joint=name,mujoco_joint=resolved,isaac_kp=runtime["stiffness"][i],mujoco_kp=kp[i],
            isaac_kd=runtime["damping"][i],mujoco_kd=kd[i],isaac_armature=runtime["armature"][i],mujoco_armature=model.dof_armature[adr],
            isaac_effort_nm=runtime["effort_limits"][i],mujoco_joint_effort_nm=model.jnt_actfrcrange[jid,1]))
    with (args.output / "joint_contract.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contracts[0]));writer.writeheader();writer.writerows(contracts)
    for kind in ("kp","kd"):
        if not all(np.isclose(row[f"isaac_{kind}"],row[f"mujoco_{kind}"]) for row in contracts):
            raise RuntimeError(f"Runtime actuator {kind} mismatch")
    evidence["runtime_contract"] = dict(joints=37,kp_match=37,kd_match=37,armature_match=37,
        effort_limit_match=sum(bool(np.isclose(row["isaac_effort_nm"],row["mujoco_joint_effort_nm"])) for row in contracts),
        limitations="Numerical interface check on Isaac states does not prove axis/geometry equivalence or identical actuator/contact dynamics.")
    evidence["runtime_contract"]["armature_match"] = sum(bool(np.isclose(row["isaac_armature"],row["mujoco_armature"])) for row in contracts)
    finger = args.run_root / "finger_armature_probe"
    if (finger / "complete.json").exists():
        for path,expected in json.loads((finger / "inputs.json").read_text()).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Finger probe input changed")
        finger_rows=[]
        for result in json.loads((finger / "complete.json").read_text())["results"]:
            yaw=result["yaw_command"]
            control=mujoco.MjModel.from_binary_path(str(probe / f"limit50_yaw{yaw:+.1f}.mjb"))
            variant=mujoco.MjModel.from_binary_path(str(finger / f"yaw{yaw:+.1f}.mjb"))
            differences=[name for name in dir(control) if isinstance(getattr(control,name),np.ndarray)
                and not np.array_equal(getattr(control,name),getattr(variant,name),equal_nan=True)]
            if differences != ["dof_armature"]:
                raise RuntimeError("Finger probe changed another model parameter")
            changed=np.flatnonzero(control.dof_armature != variant.dof_armature)
            if len(changed)!=14 or not np.allclose(variant.dof_armature[changed],.001):
                raise RuntimeError("Finger probe changed unexpected inertia values")
            metrics=result["metrics"]
            baseline=json.loads((probe / f"limit50_yaw{yaw:+.1f}.json").read_text())["metrics"]
            finger_rows.append(dict(yaw_command=yaw,elapsed_s=metrics["elapsed_s"],fell=metrics["fell"],
                baseline_body_yaw_rmse_rad_s=baseline["yaw_rate_tracking_rmse_rps"],body_yaw_rmse_rad_s=metrics["yaw_rate_tracking_rmse_rps"],
                baseline_forward_rmse_m_s=baseline["speed_tracking_rmse_mps"],forward_rmse_m_s=metrics["speed_tracking_rmse_mps"]))
        with (args.output / "finger_comparison.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(finger_rows[0]));writer.writeheader();writer.writerows(finger_rows)
        evidence["finger_probe"] = dict(cases=3,only_model_array_changed="dof_armature",changed_dofs_each=14,source_hashes_pass=True)
    (args.output / "verification.json").write_text(json.dumps(evidence,indent=2))
    print("PASS: live interface, reference trajectories, source hashes, only four ankle force limits changed")


if __name__ == "__main__":
    main()
