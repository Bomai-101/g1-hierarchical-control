"""Snapshot actual PhysX mass/COM/inertia and actuator arrays for a derived model."""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_root/"isaac_sim"))
    launcher=AppLauncher(args);app=launcher.app;env=None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import g1_walk_sim51
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        cfg=load_cfg_from_registry("G1-Walk-Flat-Sim51-Play-v0","env_cfg_entry_point")
        cfg.scene.num_envs=1;cfg.sim.device=args.device;cfg.seed=42
        env=gym.make("G1-Walk-Flat-Sim51-Play-v0",cfg=cfg);env.reset(seed=42)
        robot=env.unwrapped.scene["robot"]
        def arr(value):return value.detach().cpu().numpy().tolist()
        result=dict(body_names=list(robot.body_names),joint_names=list(robot.joint_names),
            masses=arr(robot.root_physx_view.get_masses()[0]),coms_xyzw=arr(robot.root_physx_view.get_coms()[0]),
            inertias=arr(robot.root_physx_view.get_inertias()[0]),
            stiffness=arr(robot.data.joint_stiffness[0]),damping=arr(robot.data.joint_damping[0]),
            armature=arr(robot.data.joint_armature[0]),effort_limits=arr(robot.data.joint_effort_limits[0]),
            python=sys.version,numpy=np.__version__,torch=torch.__version__,policy_steps=0,
            note="Actual PhysX arrays after environment construction/reset; no policy loaded or executed; COM quaternion is xyzw.")
        (args.output/"properties.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        files=[Path(__file__).resolve(),args.reference_root/"isaac_sim/assets/g1_minimal.usd",args.reference_root/"isaac_sim/g1_walk_sim51/g1_asset_cfg.py",args.reference_root/"isaac_sim/g1_walk_sim51/g1_env_cfg.py"]
        (args.output/"inputs.json").write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
        print("MODEL_PROPERTIES",len(result["body_names"]),sum(result["masses"]),flush=True)
    except BaseException:
        traceback.print_exc();raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=="__main__":main()
