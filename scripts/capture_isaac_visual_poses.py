"""Capture actual Isaac body/link poses, plus native USD visual geometry.

No Isaac graphics renderer is used. The saved poses can be rendered later
without re-simulating Isaac or replacing its kinematics with another robot.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def export_geometry(asset, names, output):
    from pxr import Usd, UsdGeom
    import numpy as np
    stage = Usd.Stage.Open(str(asset))
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
            continue
        body = prim.GetParent()
        while body and body.GetName() not in names:
            body = body.GetParent()
        if not body:
            raise RuntimeError("Visual mesh has no matching recorded body")
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        # USD transforms use row vectors. Geometry stays in its rigid-body frame.
        points = (np.c_[points, np.ones(len(points))] @ np.asarray(transform))[:, :3]
        faces, cursor = [], 0
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = indices[cursor:cursor + count]
            faces.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1))
            cursor += count
        file = output / f"mesh_{len(meshes):02d}.obj"
        with file.open("w") as stream:
            for vertex in points:
                stream.write("v " + " ".join(f"{v:.9g}" for v in vertex) + "\n")
            for face in faces:
                stream.write("f " + " ".join(str(v + 1) for v in face) + "\n")
        color = mesh.GetDisplayColorAttr().Get()
        rgba = list(color[0]) + [1] if color else [0.60, 0.66, 0.72, 1]
        meshes.append(dict(body=body.GetName(), file=file.name, rgba=rgba, usd_path=str(prim.GetPath())))
    if not meshes:
        raise RuntimeError("No USD visual meshes extracted")
    scene = ET.Element("mujoco", model="Isaac_pose_visualization_only")
    ET.SubElement(scene, "compiler", angle="radian", meshdir=str(output.resolve()))
    ET.SubElement(scene, "option", gravity="0 0 0")
    asset_xml = ET.SubElement(scene, "asset")
    ET.SubElement(asset_xml, "texture", name="grid", type="2d", builtin="checker", rgb1="0.17 0.20 0.23", rgb2="0.23 0.27 0.30", width="512", height="512")
    ET.SubElement(asset_xml, "material", name="ground", texture="grid", texrepeat="20 20", reflectance="0.05")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="0 -2 5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .1", material="ground", contype="0", conaffinity="0")
    for name in names:
        body = ET.SubElement(world, "body", name=name)
        ET.SubElement(body, "freejoint", name=f"pose_{name}")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia=".001 .001 .001")
        for i, item in enumerate(meshes):
            if item["body"] == name:
                ET.SubElement(asset_xml, "mesh", name=f"visual_{i}", file=item["file"])
                ET.SubElement(body, "geom", type="mesh", mesh=f"visual_{i}", rgba=" ".join(map(str, item["rgba"])), contype="0", conaffinity="0", density="0")
    ET.ElementTree(scene).write(output / "isaac_visual.xml")
    (output / "geometry.json").write_text(json.dumps(dict(meshes=meshes, body_names=names, meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
         method="native USD mesh vertices transformed into their recorded rigid body frames; no MuJoCo physics or joint mapping"), indent=2))


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_root / "isaac_sim"))
    asset = args.reference_root / "isaac_sim/assets/g1_minimal.usd"
    paths = [args.baseline, args.candidate, asset, Path(__file__).resolve()]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "inputs.json").write_text(json.dumps(hashes, indent=2))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import g1_walk_sim51
        from g1_walk_sim51.ppo_cfg import G1FlatPPORunnerCfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        cfg = load_cfg_from_registry("G1-Walk-Flat-Sim51-Play-v0", "env_cfg_entry_point")
        cfg.scene.num_envs = 16
        cfg.sim.device = args.device
        cfg.seed = 42
        cfg.episode_length_s = 60
        cfg.observations.policy.enable_corruption = False
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_standing_envs = 0
        cfg.commands.base_velocity.resampling_time_range = (10000, 10000)
        cfg.commands.base_velocity.ranges.lin_vel_x = (1, 1)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        cfg.events.physics_material.params["static_friction_range"] = (.8, .8)
        cfg.events.physics_material.params["dynamic_friction_range"] = (.6, .6)
        cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
        env = RslRlVecEnvWrapper(gym.make("G1-Walk-Flat-Sim51-Play-v0", cfg=cfg))
        robot = env.unwrapped.scene["robot"]
        term = env.unwrapped.command_manager.get_term("base_velocity")
        agent = G1FlatPPORunnerCfg()
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
        geometry = args.output_dir / "geometry"
        geometry.mkdir()
        export_geometry(asset, list(robot.body_names), geometry)
        steps = round(args.duration / env.unwrapped.step_dt)
        for label, checkpoint in (("baseline1499", args.baseline), ("candidate1999", args.candidate)):
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
            env.seed(42)
            env.reset()
            positions, quaternions, roots, joints, commands = [], [], [], [], []
            for step in range(steps + 1):
                term._update_command()
                obs = env.get_observations()
                expected = torch.tensor([1, 0, 0], device=args.device).expand(16, 3)
                if not torch.equal(obs["policy"][:, 9:12], expected):
                    raise RuntimeError("Actor does not receive fixed [1,0,0]")
                positions.append(robot.data.body_pos_w[0].cpu().numpy().copy())
                quaternions.append(robot.data.body_quat_w[0].cpu().numpy().copy())
                roots.append(torch.cat((robot.data.root_pos_w, robot.data.root_quat_w, robot.data.root_lin_vel_w,
                                        robot.data.root_ang_vel_w), dim=1)[0].cpu().numpy().copy())
                joints.append(robot.data.joint_pos[0].cpu().numpy().copy())
                commands.append(obs["policy"][0, 9:12].cpu().numpy().copy())
                if step == steps:
                    break
                with torch.inference_mode():
                    action = policy(obs)
                _, _, done, _ = env.step(action)
                if done[0]:
                    raise RuntimeError("Displayed Isaac environment terminated")
            np.savez_compressed(args.output_dir / f"{label}.npz", time=np.arange(steps + 1) * env.unwrapped.step_dt,
                body_positions=positions, body_quaternions=quaternions, root=roots, joint_positions=joints,
                commands=commands, origin=env.unwrapped.scene.env_origins[0].cpu().numpy(),
                body_names=np.array(robot.body_names), joint_names=np.array(robot.joint_names))
            print("ISAAC_POSES " + label + " saved", flush=True)
        for p, expected in hashes.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Input changed")
        (args.output_dir / "complete.json").write_text(json.dumps(dict(command=[1, 0, 0], duration=args.duration,
            seed=42, displayed_environment=0, num_envs=16, pose_source="Isaac GPU physics", rendering="native USD geometry + recorded body poses; not native Isaac screen capture"), indent=2))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
