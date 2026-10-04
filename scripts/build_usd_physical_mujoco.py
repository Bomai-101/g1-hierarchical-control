"""Extend the frozen USD-aligned tree with contact and two actuator representations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematic",type=Path,required=True);parser.add_argument("--contacts",type=Path,required=True)
    parser.add_argument("--properties",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    original=ET.parse(args.kinematic).getroot();contacts=json.loads(args.contacts.read_text());props=json.loads(args.properties.read_text())
    for mode in ("explicit","position"):
        scene=copy.deepcopy(original);scene.set("model",f"g1_usd_physical_{mode}")
        scene.find("option").set("gravity","0 0 -9.81")
        # Both actuator variants use the same integrator; only actuator representation differs.
        scene.find("option").set("integrator","implicitfast")
        scene.find("option").set("iterations","100")
        world=scene.find("worldbody");assets=scene.find("asset");elements={b.attrib["name"]:b for b in world.iter("body")}
        for i,mesh in enumerate(contacts["meshes"]):
            if mesh["approximation"]!="convexHull":raise RuntimeError("Only recorded convex-hull colliders supported")
            ET.SubElement(assets,"mesh",name=f"contact_mesh_{i}",file=mesh["file"])
            ET.SubElement(elements[mesh["body"]],"geom",name=f"contact_{mesh['body']}",type="mesh",mesh=f"contact_mesh_{i}",
                contype="1",conaffinity="2",density="0",group="3",friction="0.8 0.02 0.0001",rgba="0.2 0.7 0.2 0.3")
        ET.SubElement(world,"geom",name="floor",type="plane",size="20 20 .1",contype="2",conaffinity="1",friction="0.8 0.02 0.0001",rgba="0.25 0.28 0.3 1")
        ET.SubElement(world,"light",pos="0 -2 5",dir="0 0 -1")
        joints={j.attrib["name"]:j for j in scene.iter("joint")};actuators=ET.SubElement(scene,"actuator")
        for index,name in enumerate(props["joint_names"]):
            limit=props["effort_limits"][index];joints[name].set("armature",str(props["armature"][index]))
            joints[name].set("actuatorfrclimited","true");joints[name].set("actuatorfrcrange",f"{-limit} {limit}")
            attributes=dict(name=name,joint=name,forcelimited="true",forcerange=f"{-limit} {limit}")
            if mode=="explicit":ET.SubElement(actuators,"motor",**attributes)
            else:ET.SubElement(actuators,"position",kp=str(props["stiffness"][index]),kv=str(props["damping"][index]),**attributes)
        ET.indent(scene);path=args.output/f"g1_usd_physical_{mode}.xml";ET.ElementTree(scene).write(path,encoding="utf-8",xml_declaration=True)
        model=mujoco.MjModel.from_xml_path(str(path));mujoco.mj_saveModel(model,str(path.with_suffix(".mjb")))
        if model.nu!=37 or model.nbody!=45 or sum((model.geom_contype!=0)|(model.geom_conaffinity!=0))!=4:raise RuntimeError("Compiled structure differs")
        print("PHYSICAL_MODEL",mode,"actuators",model.nu,flush=True)
    files=[args.kinematic,args.contacts,args.properties,Path(__file__).resolve()]+[Path(m["file"]) for m in contacts["meshes"]]
    (args.output/"build_manifest.json").write_text(json.dumps(dict(inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        mujoco=mujoco.__version__,integrator="implicitfast",physics_dt=.001,policy_dt=.02,
        contact_filter="robot(1,2), ground(2,1): no robot-robot contacts",friction=.8,
        limits="Actual Isaac configured simulation limits, not verified hardware ratings",scope="Derived physical candidates; not deployed. Comparison with original model changes multiple physical properties; explicit vs position is isolated actuator-representation comparison.",
        caveats=["MuJoCo position servo/integrator not identical to PhysX implicit drive","MuJoCo friction0.8 does not exactly represent Isaac static0.8/dynamic0.6","Contact compliance and solver differ"]),indent=2))


if __name__=="__main__":main()
