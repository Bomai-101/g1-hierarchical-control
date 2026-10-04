"""Build a USD-aligned articulated MuJoCo model for kinematic validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from analyze_model_geometry import rotation,transform


def values(array):
    return " ".join(f"{x:.17g}" for x in np.asarray(array).reshape(-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd",type=Path,required=True)
    parser.add_argument("--properties",type=Path,required=True)
    parser.add_argument("--visuals",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    usd=json.loads(args.usd.read_text());props=json.loads(args.properties.read_text())
    visuals=json.loads((args.visuals/"geometry.json").read_text())
    if set(props["body_names"])!={b["name"] for b in usd["bodies"]}:raise RuntimeError("Runtime bodies differ")
    if set(props["joint_names"])!={j["name"] for j in usd["joints"] if j["kind"]=="revolute"}:raise RuntimeError("Runtime joints differ")
    scene=ET.Element("mujoco",model="g1_usd_matched_kinematics_only")
    scene.append(ET.Comment("Kinematic validation only: mj_forward; no contact geometry or actuators; gravity disabled. Not a locomotion deployment model."))
    ET.SubElement(scene,"compiler",angle="radian",fusestatic="false",inertiafromgeom="false",meshdir=str(args.visuals.resolve()))
    ET.SubElement(scene,"option",gravity="0 0 0",timestep="0.001")
    assets=ET.SubElement(scene,"asset")
    world=ET.SubElement(scene,"worldbody")
    elements={"pelvis":ET.SubElement(world,"body",name="pelvis",pos="0 0 0.74")}
    ET.SubElement(elements["pelvis"],"freejoint",name="floating_base_joint")
    remaining=list(usd["joints"])
    while remaining:
        progress=False
        for joint in remaining[:]:
            parent,child=joint["frames"]
            if parent["body"] not in elements:continue
            neutral=transform(parent["local_pos"],parent["local_quat"])@np.linalg.inv(transform(child["local_pos"],child["local_quat"]))
            quat=np.zeros(4);mujoco.mju_mat2Quat(quat,neutral[:3,:3].reshape(-1))
            element=ET.SubElement(elements[parent["body"]],"body",name=child["body"],pos=values(neutral[:3,3]),quat=values(quat))
            elements[child["body"]]=element
            if joint["kind"]=="revolute":
                axis=rotation(child["local_quat"])@np.eye(3)[{"X":0,"Y":1,"Z":2}[joint["axis_token"]]]
                ET.SubElement(element,"joint",name=joint["name"],type="hinge",pos=values(child["local_pos"]),axis=values(axis),
                    limited="true",range=values(np.deg2rad([joint["lower_deg"],joint["upper_deg"]])))
            remaining.remove(joint);progress=True
        if not progress:raise RuntimeError("Disconnected joint graph")
    for name,element in elements.items():
        index=props["body_names"].index(name)
        inertia=np.asarray(props["inertias"][index]).reshape(3,3)
        inertia=(inertia+inertia.T)/2
        if np.min(np.linalg.eigvalsh(inertia))<=0:raise RuntimeError("Non-positive inertia")
        # PhysX get_inertias is already expressed in the link's actor frame.
        # Do not rotate it a second time with the COM principal-axis quaternion.
        full=[inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]
        ET.SubElement(element,"inertial",pos=values(props["coms_xyzw"][index][:3]),mass=values([props["masses"][index]]),fullinertia=values(full))
        for i,mesh in enumerate(visuals["meshes"]):
            if mesh["body"]!=name:continue
            ET.SubElement(assets,"mesh",name=f"visual_{i}",file=mesh["file"])
            ET.SubElement(element,"geom",name=f"visual_{i}",type="mesh",mesh=f"visual_{i}",rgba=values(mesh["rgba"]),contype="0",conaffinity="0",density="0",group="1")
    path=args.output/"g1_usd_matched_kinematic.xml"
    ET.indent(scene);ET.ElementTree(scene).write(path,encoding="utf-8",xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(path))
    if model.nbody!=45 or model.njnt!=38 or model.nu!=0:raise RuntimeError("Compiled structure differs")
    mujoco.mj_saveModel(model,str(args.output/"g1_usd_matched_kinematic.mjb"))
    files=[args.usd,args.properties,args.visuals/"geometry.json",Path(__file__).resolve(),Path(__file__).with_name("analyze_model_geometry.py")]
    files.extend(args.visuals/m["file"] for m in visuals["meshes"])
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/"build_manifest.json").write_text(json.dumps(dict(inputs=hashes,mujoco=mujoco.__version__,bodies=44,revolute_joints=37,
        fixed_joints=6,total_mass_kg=float(model.body_mass.sum()),scope="Kinematics only; no mj_step, policy inference, actuators, contact geometry or gravity.",
        inertia_definition="Actual PhysX COM inertia in link actor frame, symmetrized for float32 roundoff; actual COM xyz; no second quaternion rotation."),indent=2))
    print("BUILT",path,"bodies",model.nbody-1,"joints",model.njnt-1,flush=True)


if __name__=="__main__":main()
