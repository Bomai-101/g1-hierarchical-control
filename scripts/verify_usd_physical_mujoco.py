"""Verify contact filtering, restored kinematics, torque semantics and small responses."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from analyze_model_geometry import rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True);parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=args.run_root;props=json.loads((root/"usd_matched_properties/properties.json").read_text());meta=json.loads(args.metadata.read_text())
    contact_source=json.loads((root/"usd_contact_meshes/contacts.json").read_text())
    manifest=json.loads((root/"usd_physical_models/build_manifest.json").read_text())
    for path,expected in manifest["inputs"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError("Build input changed")
    kinematic=mujoco.MjModel.from_binary_path(str(root/"usd_matched_kinematic/g1_usd_matched_kinematic.mjb"))
    models={mode:mujoco.MjModel.from_binary_path(str(root/"usd_physical_models"/f"g1_usd_physical_{mode}.mjb")) for mode in ("explicit","position")}
    differences=[name for name in dir(models["explicit"]) if isinstance(getattr(models["explicit"],name),np.ndarray)
        and not np.array_equal(getattr(models["explicit"],name),getattr(models["position"],name),equal_nan=True)]
    allowed={"actuator_biasprm","actuator_gainprm","actuator_biastype"}
    if set(differences)!=allowed:raise RuntimeError(f"Actuator variants change additional arrays: {differences}")
    results={};kp=np.array(props["stiffness"]);kd=np.array(props["damping"]);limits=np.array(props["effort_limits"])
    for mode,model in models.items():
        for name in ("body_pos","body_quat","body_mass","body_ipos","body_iquat","body_inertia","jnt_pos","jnt_axis"):
            if not np.array_equal(getattr(model,name),getattr(kinematic,name)):raise RuntimeError(f"Frozen body/joint property changed: {name}")
        ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in props["joint_names"]]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
        ai=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name) for name in props["joint_names"]])
        if not np.allclose(model.dof_armature[va],props["armature"]) or not np.array_equal(model.actuator_forcerange[ai],np.c_[-limits,limits]):raise RuntimeError("Actuator properties differ")
        geometry_checks=[]
        for item in contact_source["meshes"]:
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"contact_"+item["body"]);mid=model.geom_dataid[gid]
            vertices=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            points=vertices@rotation(model.geom_quat[gid]).T+model.geom_pos[gid]
            expected=np.array(item["vertices"])
            error=float(np.max(np.abs(np.array([points.min(axis=0),points.max(axis=0)])-np.array([expected.min(axis=0),expected.max(axis=0)]))))
            if error>1e-7:raise RuntimeError("Contact mesh bounds differ")
            if model.geom_contype[gid]!=1 or model.geom_conaffinity[gid]!=2:raise RuntimeError("Robot mask differs")
            geometry_checks.append(dict(body=item["body"],local_bounds_max_error_m=error))
        data=mujoco.MjData(model);data.qpos[:]=model.qpos0;data.qpos[qa]=meta["default_joint_pos"]
        # Force all potential robot surfaces close to ground, then verify every generated pair is ground/robot.
        data.qpos[2]=.65;mujoco.mj_forward(model,data);floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
        if not data.ncon:raise RuntimeError("No contact produced by lowered feet")
        for c in data.contact:
            if floor not in (c.geom1,c.geom2):raise RuntimeError("Robot-robot contact was not filtered")
        contacts=int(data.ncon)
        data.qpos[2]=2.;data.qvel[:]=0;data.qvel[va]=np.linspace(-.1,.1,37)
        desired=data.qpos[qa]+.001;expected=kp*.001-kd*data.qvel[va]
        data.ctrl[ai]=expected if mode=="explicit" else desired;mujoco.mj_forward(model,data)
        torque_error=float(np.max(np.abs(data.qfrc_actuator[va]-expected)))
        if torque_error>1e-10:raise RuntimeError("PD torque semantics differ")
        # Large target errors must obey each configured simulation force bound.
        data.ctrl[ai]=1e6 if mode=="explicit" else data.qpos[qa]+1e6;mujoco.mj_forward(model,data)
        limit_error=float(np.max(np.abs(data.qfrc_actuator[va]-limits)))
        if limit_error>1e-9:raise RuntimeError("Force limits not enforced")
        # Response checks suspend the model, disable gravity and keep other joint targets fixed.
        responses=[];model.opt.gravity[:]=0
        for name in ("left_hip_pitch_joint","left_ankle_pitch_joint","left_five_joint"):
            data=mujoco.MjData(model);data.qpos[:]=model.qpos0;data.qpos[2]=2.;data.qpos[qa]=meta["default_joint_pos"]
            desired=data.qpos[qa].copy();index=props["joint_names"].index(name);desired[index]+=.01
            peak_speed=peak_force=0.
            for _ in range(200):
                data.ctrl[ai]=np.clip(kp*(desired-data.qpos[qa])-kd*data.qvel[va],-limits,limits) if mode=="explicit" else desired
                mujoco.mj_step(model,data);peak_speed=max(peak_speed,float(np.max(np.abs(data.qvel[va]))));peak_force=max(peak_force,float(np.max(np.abs(data.qfrc_actuator[va]))))
                if not np.all(np.isfinite(data.qpos)) or any(w.number for w in data.warning):raise RuntimeError("Response numerical failure")
            responses.append(dict(joint=name,target_delta_rad=.01,final_target_error_rad=float(desired[index]-data.qpos[qa[index]]),peak_joint_speed_rad_s=peak_speed,peak_actuator_force_nm=peak_force))
        results[mode]=dict(contacts_when_lowered=contacts,all_contacts_ground_only=True,contact_geometry=geometry_checks,
            pd_torque_max_abs_error_nm=torque_error,force_limit_max_abs_error_nm=limit_error,responses=responses)
    result=dict(actuator_variant_model_array_differences=differences,checks=results,
        caveat="Finite response is not proof of accurate dynamics or stability; explicit PD may have large high-frequency motion. No PhysX equivalence claim.")
    (args.output/"component_verification.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
