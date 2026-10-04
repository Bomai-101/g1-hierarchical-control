"""Test only the finger armature discrepancy found in the live runtime audit."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_root/"mujoco"))
    from mujoco_eval import run_grid
    runtime_path=args.run_root/"interface_live/runtime.json"
    runtime=json.loads(runtime_path.read_text())
    metadata=args.reference_root/"mujoco/policies/isaac_metadata.json"
    model_path=args.reference_root/"mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml"
    files=[args.policy,runtime_path,metadata,model_path,Path(__file__).resolve(),args.reference_root/"mujoco/mujoco_eval/run_grid.py"]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/"inputs.json").write_text(json.dumps(hashes,indent=2))
    rows=[]
    for yaw in (0.,-.2,.2):
        stem=f"yaw{yaw:+.1f}"
        cfg=SimpleNamespace(model=str(model_path),policy=str(args.policy.resolve()),metadata=str(metadata),
            command_x=1.,command_y=0.,command_yaw=yaw,timestep=.001,duration=30.,seed=42,device="cpu",
            allow_missing_joints=False,initial_base_height=.74,min_base_height=.35,base_body="pelvis",control_mode="pd")
        original_armature=run_grid.configure_policy_armature
        original_forward=mujoco.mj_forward
        changes=[]
        saved=False
        def armature(model,names,joint_ids):
            original_armature(model,names,joint_ids)
            if names != runtime["joint_names"]:
                raise RuntimeError("Runtime joint order differs")
            for i,jid in enumerate(joint_ids):
                adr=model.jnt_dofadr[jid]
                desired=round(runtime["armature"][i],6)
                if not np.isclose(model.dof_armature[adr],desired):
                    if "_hand_" not in mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(jid)):
                        raise RuntimeError("Non-finger parameter would change")
                    changes.append(dict(joint=names[i],before=float(model.dof_armature[adr]),after=desired))
                    model.dof_armature[adr]=desired
            if len(changes)!=14:
                raise RuntimeError("Expected 14 finger armature changes")
        def forward(model,data,*values,**kwargs):
            nonlocal saved
            result=original_forward(model,data,*values,**kwargs)
            if not saved:
                control=mujoco.MjModel.from_binary_path(str(args.run_root/"ankle_limit_probe"/f"limit50_{stem}.mjb"))
                differences=[name for name in dir(model) if isinstance(getattr(model,name),np.ndarray)
                    and not np.array_equal(getattr(model,name),getattr(control,name),equal_nan=True)]
                if differences != ["dof_armature"]:
                    raise RuntimeError(f"Unexpected compiled model change: {differences}")
                mujoco.mj_saveModel(model,str(args.output/f"{stem}.mjb"))
                saved=True
            return result
        with patch.object(run_grid,"configure_policy_armature",armature),patch.object(mujoco,"mj_forward",forward):
            metrics=run_grid.run_once(cfg,"plane",.8)
        row=dict(yaw_command=yaw,metrics=metrics,changes=changes,only_model_array_changed="dof_armature")
        (args.output/f"{stem}.json").write_text(json.dumps(row,indent=2));rows.append(row)
        print("FINGER_PROBE",yaw,metrics["elapsed_s"],metrics["fell"],metrics["yaw_rate_tracking_rmse_rps"],flush=True)
    for path,expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Source input changed")
    (args.output/"complete.json").write_text(json.dumps(dict(results=rows,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__),indent=2))


if __name__=="__main__":
    main()
