"""Compare original joint frames/geometry and check USD FK against real Isaac poses."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def rotation(quaternion):
    w,x,y,z=np.asarray(quaternion)/np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def transform(position,quaternion):
    value=np.eye(4);value[:3,:3]=rotation(quaternion);value[:3,3]=position;return value


def forward_usd(usd,angles):
    result={"pelvis":np.eye(4)};remaining=list(usd["joints"])
    while remaining:
        progress=False
        for joint in remaining[:]:
            parent,child=joint["frames"]
            if parent["body"] not in result:continue
            spin=np.eye(4)
            if joint["kind"]=="revolute":
                axis=np.eye(3)[{"X":0,"Y":1,"Z":2}[joint["axis_token"]]]
                angle=angles.get(joint["name"],0.)
                spin[:3,:3]=rotation([np.cos(angle/2),*(axis*np.sin(angle/2))])
            result[child["body"]]=result[parent["body"]]@transform(parent["local_pos"],parent["local_quat"])@spin@np.linalg.inv(transform(child["local_pos"],child["local_quat"]))
            remaining.remove(joint);progress=True
        if not progress:raise RuntimeError("USD graph is disconnected or cyclic")
    return result


def angle_between(left,right):
    dot=np.dot(left,right)/(np.linalg.norm(left)*np.linalg.norm(right))
    return float(np.rad2deg(np.arccos(np.clip(dot,-1,1))))


def write_csv(path,rows):
    with path.open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    usd=json.loads(args.usd.read_text());metadata=json.loads(args.metadata.read_text())
    if usd["meters_per_unit"]!=1 or usd["up_axis"]!="Z":raise RuntimeError("Unexpected USD units/axis")
    if hashlib.sha256(Path(usd["asset"]).read_bytes()).hexdigest()!=usd["asset_sha256"]:raise RuntimeError("USD changed")
    joints={j["name"]:j for j in usd["joints"]};bodies={b["name"]:b for b in usd["bodies"]}
    if sum(j["kind"]=="revolute" for j in joints.values())!=37:raise RuntimeError("Joint count differs")
    authored=forward_usd(usd,{})
    zero_position_error=max(np.linalg.norm(authored[name][:3,3]-b["position"]) for name,b in bodies.items())
    zero_rotation_error=max(np.max(np.abs(authored[name][:3,:3]-np.array(b["rotation_rows"]).T)) for name,b in bodies.items())
    if zero_position_error>2e-6 or zero_rotation_error>2e-5:raise RuntimeError("Authored transforms are not zero-joint pose")
    isaac=np.load(args.run_root/"visual_isaac/baseline1499.npz")
    errors=[]
    for index in (0,50,100,200,400):
        fk=forward_usd(usd,dict(zip(isaac["joint_names"],isaac["joint_positions"][index])))
        pelvis=list(isaac["body_names"]).index("pelvis")
        inverse=np.linalg.inv(transform(isaac["body_positions"][index,pelvis],isaac["body_quaternions"][index,pelvis]))
        position_error=rotation_error=0.
        for body,name in enumerate(isaac["body_names"]):
            actual=inverse@transform(isaac["body_positions"][index,body],isaac["body_quaternions"][index,body])
            position_error=max(position_error,float(np.linalg.norm(actual[:3,3]-fk[str(name)][:3,3])))
            rotation_error=max(rotation_error,float(np.max(np.abs(actual[:3,:3]-fk[str(name)][:3,:3]))))
        errors.append(dict(sample=index,max_body_position_error_m=position_error,max_rotation_matrix_abs_error=rotation_error))
    if max(e["max_body_position_error_m"] for e in errors)>2e-4 or max(e["max_rotation_matrix_abs_error"] for e in errors)>2e-3:
        raise RuntimeError(f"USD kinematic reconstruction disagrees with real Isaac: {errors}")
    model=mujoco.MjModel.from_binary_path(str(args.run_root/"visual_mujoco/baseline1499.mjb"));data=mujoco.MjData(model)
    # The alias mapping is copied from the archived reference source, not guessed from body labels.
    aliases={"torso_joint":"waist_yaw_joint","left_elbow_pitch_joint":"left_elbow_joint","right_elbow_pitch_joint":"right_elbow_joint",
        "left_elbow_roll_joint":"left_wrist_roll_joint","right_elbow_roll_joint":"right_wrist_roll_joint"}
    finger={"zero":"thumb_0","one":"thumb_1","two":"thumb_2","three":"index_0","four":"index_1","five":"middle_0","six":"middle_1"}
    for side in ("left","right"):
        aliases.update({f"{side}_{word}_joint":f"{side}_hand_{target}_joint" for word,target in finger.items()})
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,aliases.get(name,name)) for name in metadata["joint_names"]]
    if min(ids)<0 or len(set(ids))!=37:raise RuntimeError("Mapping is incomplete")
    data.qpos[:]=model.qpos0;data.qpos[2]=.74;mujoco.mj_forward(model,data)
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    rows=[]
    for name,jid in zip(metadata["joint_names"],ids):
        joint=joints[name];parent,child=joint["frames"]
        body=model.jnt_bodyid[jid]
        local_axis=rotation(child["local_quat"])@np.eye(3)[{"X":0,"Y":1,"Z":2}[joint["axis_token"]]]
        rows.append(dict(isaac_joint=name,mujoco_joint=aliases.get(name,name),usd_child=child["body"],mujoco_child=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(body)),
            zero_anchor_distance_m=float(np.linalg.norm(np.array(parent["anchor"])-(data.xanchor[jid]-data.xpos[base]))),
            zero_axis_angle_deg=angle_between(parent["axis"],data.xaxis[jid]),
            child_local_axis_angle_deg=angle_between(local_axis,model.jnt_axis[jid]),
            usd_anchor_internal_error_m=float(np.linalg.norm(np.array(parent["anchor"])-child["anchor"])),
            usd_axis_internal_angle_deg=angle_between(parent["axis"],child["axis"]),
            usd_lower_rad=float(np.deg2rad(joint["lower_deg"])),usd_upper_rad=float(np.deg2rad(joint["upper_deg"])),
            mujoco_lower_rad=float(model.jnt_range[jid,0]),mujoco_upper_rad=float(model.jnt_range[jid,1])))
    write_csv(args.output/"joint_frames.csv",rows)
    data.qpos[model.jnt_qposadr[ids]]=metadata["default_joint_pos"];mujoco.mj_forward(model,data)
    usd_default=forward_usd(usd,dict(zip(metadata["joint_names"],metadata["default_joint_pos"])))
    body_rows=[]
    mapping={"pelvis":base};mapping.update({joints[name]["frames"][1]["body"]:int(model.jnt_bodyid[jid]) for name,jid in zip(metadata["joint_names"],ids)})
    for name,bid in mapping.items():
        b=bodies[name]
        actual=transform(data.xpos[bid]-data.xpos[base],data.xquat[bid])
        com=b["com_local"]
        valid_com=all(isinstance(x,(int,float)) for x in com)
        usd_inertia=rotation(b["principal_axes"])@np.diag(b["diagonal_inertia"])@rotation(b["principal_axes"]).T
        mu_inertia=rotation(model.body_iquat[bid])@np.diag(model.body_inertia[bid])@rotation(model.body_iquat[bid]).T
        body_rows.append(dict(usd_body=name,mujoco_body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid),
            default_origin_distance_m=float(np.linalg.norm(usd_default[name][:3,3]-actual[:3,3])),
            default_rotation_difference_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(usd_default[name][:3,:3].T@actual[:3,:3])-1)/2,-1,1)))),
            usd_mass_kg=b["mass"],mujoco_mass_kg=float(model.body_mass[bid]),
            inertia_tensor_frobenius_difference_kg_m2=float(np.linalg.norm(usd_inertia-mu_inertia)),
            com_local_distance_m=float(np.linalg.norm(np.array(com)-model.body_ipos[bid])) if valid_com else None))
    write_csv(args.output/"body_default_pose_and_mass.csv",body_rows)
    feet={}
    for side in ("left","right"):
        name=f"{side}_ankle_roll_link";bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        geoms=[]
        for gid in np.flatnonzero(model.geom_bodyid==bid):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):continue
            if model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_SPHERE:raise RuntimeError("Unexpected foot collider shape")
            geoms.append(dict(center=model.geom_pos[gid].tolist(),radius=float(model.geom_size[gid,0]),
                contype=int(model.geom_contype[gid]),conaffinity=int(model.geom_conaffinity[gid]),margin=float(model.geom_margin[gid])))
        usd_colliders=[c for c in usd["colliders"] if c["body"]==name and c["enabled"]]
        bounds=np.array([c["bounds_local"] for c in usd_colliders])
        mu_lo=np.min([np.array(g["center"])-g["radius"] for g in geoms],axis=0)
        mu_hi=np.max([np.array(g["center"])+g["radius"] for g in geoms],axis=0)
        feet[side]=dict(usd_colliders=usd_colliders,mujoco_spheres=geoms,usd_union_bounds=[bounds[:,0].min(axis=0).tolist(),bounds[:,1].max(axis=0).tolist()],
            mujoco_union_bounds=[mu_lo.tolist(),mu_hi.tolist()])
    summary=dict(authored_zero_position_error_m=float(zero_position_error),authored_zero_rotation_matrix_error=float(zero_rotation_error),
        live_isaac_fk_checks=errors,joints=37,body_comparisons=len(body_rows),
        zero_axis_angle_max_deg=max(r["zero_axis_angle_deg"] for r in rows),zero_anchor_distance_max_m=max(r["zero_anchor_distance_m"] for r in rows),
        child_local_axis_angle_max_deg=max(r["child_local_axis_angle_deg"] for r in rows),
        usd_authored_positive_mass_sum_kg=sum(b["mass"] for b in bodies.values() if b["mass"] is not None),mujoco_mass_sum_kg=float(model.body_mass.sum()),
        usd_collision_prims=len(usd["colliders"]),mujoco_contact_geoms=int(np.sum((model.geom_contype!=0)|(model.geom_conaffinity!=0))),
        mujoco_robot_contact_geoms=int(np.sum(((model.geom_contype!=0)|(model.geom_conaffinity!=0))&(model.geom_bodyid!=0))),
        left_right_foot_bounds_equal=bool(np.allclose(feet["left"]["usd_union_bounds"],feet["right"]["usd_union_bounds"]) and np.allclose(feet["left"]["mujoco_union_bounds"],feet["right"]["mujoco_union_bounds"])),
        pelvis=dict(usd_mass=bodies["pelvis"]["mass"],mujoco_mass=float(model.body_mass[base]),usd_com=bodies["pelvis"]["com_local"],mujoco_com=model.body_ipos[base].tolist()),
        caveat="Authored masses may use automatic USD defaults on fixed tiny bodies; direct per-body mass comparison does not aggregate differently fused fixed links. No causal locomotion repair is demonstrated.")
    (args.output/"feet.json").write_text(json.dumps(feet,indent=2));(args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    chain=["pelvis","left_hip_pitch_link","left_hip_roll_link","left_hip_yaw_link","left_knee_link","left_ankle_pitch_link","left_ankle_roll_link"]
    plot_data=dict(left_leg_names=chain,usd_default_positions=[usd_default[name][:3,3].tolist() for name in chain],
        mujoco_default_positions=[(data.xpos[mapping[name]]-data.xpos[base]).tolist() for name in chain])
    (args.output/"plot_data.json").write_text(json.dumps(plot_data,indent=2))
    files=[args.usd,args.metadata,args.run_root/"visual_isaac/baseline1499.npz",args.run_root/"visual_mujoco/baseline1499.mjb",Path(__file__).resolve()]
    (args.output/"inputs.json").write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
