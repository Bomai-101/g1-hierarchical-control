"""Compare real Isaac policy inputs/targets with the reference MuJoCo interface."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-policy", type=Path, required=True)
    parser.add_argument("--candidate-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_root / "isaac_sim"))
    sys.path.insert(0, str(args.reference_root / "mujoco"))
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
        from mujoco_eval.policy import build_policy_observation, TorchActorPolicy
        from isaaclab.utils.io import dump_yaml
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
        base = env.unwrapped
        robot = base.scene["robot"]
        term = base.command_manager.get_term("base_velocity")
        action_term = base.action_manager.get_term("joint_pos")
        runner = OnPolicyRunner(env, G1FlatPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
        metadata_path = args.reference_root / "mujoco/policies/isaac_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        inputs = [args.baseline, args.candidate, args.baseline_policy, args.candidate_policy, metadata_path, Path(__file__).resolve(),
                  args.reference_root / "mujoco/mujoco_eval/policy.py",
                  args.reference_root / "isaac_sim/g1_walk_sim51/g1_env_cfg.py",
                  args.reference_root / "isaac_sim/g1_walk_sim51/g1_asset_cfg.py",
                  args.reference_root / "isaac_sim/assets/g1_minimal.usd"]
        hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        (args.output / "inputs.json").write_text(json.dumps(hashes, indent=2))
        dump_yaml(str(args.output / "env.yaml"), cfg)
        def array(tensor):
            return tensor.detach().cpu().numpy().copy()
        defaults = array(robot.data.default_joint_pos[0])
        if list(robot.joint_names) != metadata["joint_names"] or not np.allclose(defaults, metadata["default_joint_pos"], atol=1e-7):
            raise RuntimeError("Joint order/defaults mismatch")
        if np.any(array(robot.data.default_joint_vel[0])):
            raise RuntimeError("MuJoCo observation assumes zero default velocity")
        runtime = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
            physics_dt=cfg.sim.dt, policy_dt=base.step_dt, joint_names=robot.joint_names,
            default_joint_pos=defaults.tolist(), stiffness=array(robot.data.joint_stiffness[0]).tolist(),
            damping=array(robot.data.joint_damping[0]).tolist(), armature=array(robot.data.joint_armature[0]).tolist(),
            effort_limits=array(robot.data.joint_effort_limits[0]).tolist(), actuator_classes={k:type(v).__name__ for k,v in robot.actuators.items()})
        (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))
        summaries = {}
        for label, checkpoint, exported in (("baseline1499", args.baseline, args.baseline_policy), ("candidate1999", args.candidate, args.candidate_policy)):
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
            other = TorchActorPolicy(exported)
            env.seed(42)
            env.reset()
            records = {k:[] for k in ("isaac_obs", "reconstructed_obs", "isaac_action", "exported_action", "target", "expected_target", "root", "joint_vel")}
            for step in range(200):
                term._update_command()
                obs = env.get_observations()
                actual = array(obs["policy"][0])
                rebuilt = build_policy_observation(array(robot.data.root_quat_w[0]), array(robot.data.root_lin_vel_w[0]),
                    array(robot.data.root_ang_vel_w[0]), array(term.command[0]), array(robot.data.joint_pos[0]),
                    array(robot.data.joint_vel[0]), defaults, array(base.action_manager.action[0]))
                with torch.inference_mode():
                    action = policy(obs)
                exported_action = other.act(actual)
                expected = defaults + metadata["action_scale"] * array(action[0])
                records["isaac_obs"].append(actual)
                records["reconstructed_obs"].append(rebuilt)
                records["isaac_action"].append(array(action[0]))
                records["exported_action"].append(exported_action)
                records["expected_target"].append(expected)
                records["root"].append(array(robot.data.root_state_w[0]))
                records["joint_vel"].append(array(robot.data.joint_vel[0]))
                _, _, done, _ = env.step(action)
                records["target"].append(array(action_term.processed_actions[0]))
                if done[0]:
                    raise RuntimeError("Audit environment terminated")
            values = {k:np.stack(v) for k,v in records.items()}
            np.savez_compressed(args.output / f"{label}.npz", **values)
            errors = dict(observation_max_abs=float(np.max(np.abs(values["isaac_obs"]-values["reconstructed_obs"]))),
                action_max_abs=float(np.max(np.abs(values["isaac_action"]-values["exported_action"]))),
                target_max_abs=float(np.max(np.abs(values["target"]-values["expected_target"]))), samples=200,
                observation_group_max_abs={name:float(np.max(np.abs(values["isaac_obs"][:,start:stop]-values["reconstructed_obs"][:,start:stop])))
                    for name,start,stop in [("linear_velocity",0,3),("angular_velocity",3,6),("gravity",6,9),("command",9,12),("joint_position",12,49),("joint_velocity",49,86),("previous_action",86,123)]})
            if errors["observation_max_abs"] > 5e-6 or errors["action_max_abs"] > 1e-5 or errors["target_max_abs"] > 1e-6:
                raise RuntimeError(f"Interface mismatch: {errors}")
            summaries[label] = errors
            print("INTERFACE_PASS", label, errors, flush=True)
        for path, expected in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Input changed")
        (args.output / "complete.json").write_text(json.dumps(summaries, indent=2))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
