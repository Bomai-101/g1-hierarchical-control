"""Check the independent MuJoCo joint tree against recorded Isaac body poses."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from analyze_model_geometry import forward_usd,rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--usd",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.model_dir/"build_manifest.json").read_text())
    for path,expected in manifest["inputs"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError("Build input changed")
    properties_dir=args.run_root/"usd_matched_properties"
    for path,expected in json.loads((properties_dir/"inputs.json").read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError("Isaac property source changed")
    properties=json.loads((properties_dir/"properties.json").read_text())
    model=mujoco.MjModel.from_binary_path(str(args.model_dir/"g1_usd_matched_kinematic.mjb"));data=mujoco.MjData(model)
    xml_model=mujoco.MjModel.from_xml_path(str(args.model_dir/"g1_usd_matched_kinematic.xml"))
    for name in dir(model):
        a,b=getattr(model,name),getattr(xml_model,name)
        if isinstance(a,np.ndarray) and not np.array_equal(a,b,equal_nan=True):raise RuntimeError(f"Binary/XML differ: {name}")
    if model.nu or np.any(model.geom_contype) or np.any(model.geom_conaffinity) or np.any(model.opt.gravity):
        raise RuntimeError("Model must remain kinematic-only")
    names=properties["body_names"]
    body_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name) for name in names])
    joint_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in properties["joint_names"]])
    if min(body_ids)<1 or min(joint_ids)<1 or len(set(joint_ids))!=37:raise RuntimeError("Incomplete named mapping")
    addresses=model.jnt_qposadr[joint_ids]
    mass_error=float(np.max(np.abs(model.body_mass[body_ids]-properties["masses"])))
    com_error=float(np.max(np.abs(model.body_ipos[body_ids]-np.array(properties["coms_xyzw"])[:,:3])))
    inertia_error=0.
    for i,bid in enumerate(body_ids):
        actual=np.asarray(properties["inertias"][i]).reshape(3,3);actual=(actual+actual.T)/2
        basis=rotation(model.body_iquat[bid]);compiled=basis@np.diag(model.body_inertia[bid])@basis.T
        inertia_error=max(inertia_error,float(np.max(np.abs(compiled-actual))))
    if mass_error>1e-12 or com_error>1e-12 or inertia_error>1e-9:raise RuntimeError("Physical property import differs")
    summary={};rows=[]
    for actor in ("baseline1499","candidate1999"):
        trace=np.load(args.run_root/"visual_isaac"/f"{actor}.npz")
        if list(trace["body_names"])!=names or list(trace["joint_names"])!=properties["joint_names"]:raise RuntimeError("Trace ordering differs")
        pelvis=names.index("pelvis");position_errors=np.zeros((len(trace["time"]),44));rotation_errors=np.zeros_like(position_errors)
        for sample in range(len(trace["time"])):
            data.qpos[:3]=trace["body_positions"][sample,pelvis]-trace["origin"]
            data.qpos[3:7]=trace["body_quaternions"][sample,pelvis]
            data.qpos[addresses]=trace["joint_positions"][sample]
            mujoco.mj_forward(model,data)
            position_errors[sample]=np.linalg.norm(data.xpos[body_ids]-(trace["body_positions"][sample]-trace["origin"]),axis=1)
            for i,bid in enumerate(body_ids):
                rotation_errors[sample,i]=np.max(np.abs(data.xmat[bid].reshape(3,3)-rotation(trace["body_quaternions"][sample,i])))
        max_position=float(position_errors.max());max_rotation=float(rotation_errors.max())
        if max_position>5e-6 or max_rotation>1e-5:raise RuntimeError(f"MuJoCo pose mismatch: {actor} {max_position} {max_rotation}")
        summary[actor]=dict(samples=len(trace["time"]),body_pose_pairs=int(position_errors.size),max_position_error_m=max_position,
            max_rotation_matrix_abs_error=max_rotation,initial_foot_position_error_m={side:float(position_errors[0,names.index(f"{side}_ankle_roll_link")]) for side in ("left","right")})
        rows.extend(dict(actor=actor,body=name,max_position_error_m=float(position_errors[:,i].max()),max_rotation_matrix_abs_error=float(rotation_errors[:,i].max())) for i,name in enumerate(names))
    # Random joint configurations give coverage beyond the walking trajectory.
    usd=json.loads(args.usd.read_text());rng=np.random.default_rng(42);random_error=0.
    data.qpos[:3]=0;data.qpos[3:7]=[1,0,0,0]
    for _ in range(64):
        angles=rng.uniform(-.4,.4,37);data.qpos[addresses]=angles;mujoco.mj_forward(model,data)
        expected=forward_usd(usd,dict(zip(properties["joint_names"],angles)))
        for name,bid in zip(names,body_ids):
            random_error=max(random_error,float(np.max(np.abs(data.xpos[bid]-expected[name][:3,3]))),float(np.max(np.abs(data.xmat[bid].reshape(3,3)-expected[name][:3,:3]))))
    if random_error>1e-10:raise RuntimeError("Random configuration reconstruction differs")
    # Ensure a wrong joint-tree transform is rejected by the same recorded-pose criterion.
    altered=mujoco.MjModel.from_binary_path(str(args.model_dir/"g1_usd_matched_kinematic.mjb"))
    knee=mujoco.mj_name2id(altered,mujoco.mjtObj.mjOBJ_BODY,"left_knee_link");altered.body_pos[knee,0]+=.001
    bad=mujoco.MjData(altered);trace=np.load(args.run_root/"visual_isaac/baseline1499.npz");pelvis=names.index("pelvis")
    bad.qpos[:3]=trace["body_positions"][0,pelvis]-trace["origin"];bad.qpos[3:7]=trace["body_quaternions"][0,pelvis]
    bad.qpos[addresses]=trace["joint_positions"][0];mujoco.mj_forward(altered,bad)
    fault_error=float(np.linalg.norm(bad.xpos[knee]-(trace["body_positions"][0,names.index("left_knee_link")]-trace["origin"])))
    if fault_error<=5e-6:raise RuntimeError("Injected 1mm transform fault was not detected")
    summary.update(properties=dict(mass_max_abs_kg=mass_error,com_max_abs_m=com_error,inertia_tensor_max_abs_kg_m2=inertia_error),
        random_configs=64,random_fk_max_abs_error=random_error,injected_1mm_transform_fault_detected=True,fault_error_m=fault_error,
        physics_steps=0,scope="Only compiled kinematics and imported body properties validated; no locomotion claim.")
    with (args.output/"body_pose_errors.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"verification.json").write_text(json.dumps(summary,indent=2))
    paths=[args.model_dir/"g1_usd_matched_kinematic.xml",args.model_dir/"g1_usd_matched_kinematic.mjb",args.usd,properties_dir/"properties.json",Path(__file__).resolve()]
    paths.extend(args.run_root/"visual_isaac"/f"{actor}.npz" for actor in ("baseline1499","candidate1999"))
    (args.output/"verification_inputs.json").write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
