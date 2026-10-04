"""Evaluate reference/local flat actors with explicit rate and heading tasks.

Run with the existing Isaac Python and IsaacLab source paths. No training,
reference-file changes, reward changes, PD changes or recovery are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-checkpoint", type=Path)
    parser.add_argument("--comparison-label", default="supplied")
    parser.add_argument("--local-label", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.comparison_label == args.local_label:
        parser.error("Actor labels must differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = args.reference_root.resolve()
    sys.path.insert(0, str(reference / "isaac_sim"))
    actor_paths = {
        args.comparison_label: args.comparison_checkpoint.resolve() if args.comparison_checkpoint else reference / "isaac_sim/checkpoints/baseline/model_1499.pt",
        args.local_label: args.local_checkpoint.resolve(),
    }
    sources = [Path(__file__).resolve(),
               reference / "isaac_sim/g1_walk_sim51/g1_env_cfg.py",
               reference / "isaac_sim/g1_walk_sim51/mdp.py",
               reference / "isaac_sim/g1_walk_sim51/ppo_cfg.py",
               reference / "isaac_sim/g1_walk_sim51/g1_asset_cfg.py",
               reference / "isaac_sim/assets/g1_minimal.usd"]
    cases = []
    for speed in (0.5, 1.0):
        for rate in (0.0, -0.2, 0.2):
            cases.append(dict(mode="rate", speed=speed, target=rate, start_yaw=0.0))
        for target in (0.0, -math.pi / 2, math.pi / 2):
            cases.append(dict(mode="heading", speed=speed, target=target,
                              start_yaw=0.5 if target == 0 else 0.0))
    if args.smoke:
        cases = [cases[0], cases[3]]
    protocol = dict(seed=args.seed, num_envs=args.num_envs, duration=args.duration,
                    warmup_s=2.0, cases=cases, terrain="plane", static_friction=0.8,
                    dynamic_friction=0.6, heading_gain=0.5, yaw_limit=1.0,
                    standing_probability=0, observation_noise=False,
                    scoring="first episode only; pre-action samples; post-reset data excluded",
                    replication="same fixed initial state; parallel environments are not independent seeds",
                    outcome="diagnostic comparison; no baseline acceptance claimed",
                    checkpoint_hashes={k: dict(path=str(v), sha256=sha256(v)) for k, v in actor_paths.items()},
                    source_hashes={str(p): sha256(p) for p in sources})
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import g1_walk_sim51  # noqa: F401
        from g1_walk_sim51.ppo_cfg import G1FlatPPORunnerCfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml

        cfg = load_cfg_from_registry("G1-Walk-Flat-Sim51-Play-v0", "env_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.episode_length_s = max(60, args.duration + 10)
        cfg.observations.policy.enable_corruption = False
        cfg.commands.base_velocity.rel_standing_envs = 0
        cfg.commands.base_velocity.resampling_time_range = (10000, 10000)
        cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
        env = RslRlVecEnvWrapper(gym.make("G1-Walk-Flat-Sim51-Play-v0", cfg=cfg))
        base = env.unwrapped
        robot = base.scene["robot"]
        term = base.command_manager.get_term("base_velocity")
        agent_cfg = G1FlatPPORunnerCfg()
        agent_cfg.device = args.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        dump_yaml(str(args.output_dir / "env.yaml"), cfg)
        dump_yaml(str(args.output_dir / "agent.yaml"), agent_cfg)
        (args.output_dir / "runtime.json").write_text(json.dumps(dict(
            python=sys.version, numpy=np.__version__, torch=torch.__version__,
            physics_dt=cfg.sim.dt, policy_dt=base.step_dt, device=args.device), indent=2))
        rows = []
        steps = round(args.duration / base.step_dt)
        for actor, checkpoint in actor_paths.items():
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
            for case_id, case in enumerate(cases):
                term.cfg.heading_command = case["mode"] == "heading"
                term.cfg.ranges.lin_vel_x = (case["speed"], case["speed"])
                term.cfg.ranges.lin_vel_y = (0, 0)
                term.cfg.ranges.ang_vel_z = (-1, 1) if term.cfg.heading_command else (case["target"], case["target"])
                term.cfg.ranges.heading = (case["target"], case["target"]) if term.cfg.heading_command else (0, 0)
                reset_cfg = base.event_manager.get_term_cfg("reset_base")
                reset_cfg.params["pose_range"]["yaw"] = (case["start_yaw"], case["start_yaw"])
                env.seed(args.seed)
                env.reset()
                if not torch.allclose(robot.data.heading_w, torch.full_like(robot.data.heading_w, case["start_yaw"]), atol=1e-4):
                    raise RuntimeError("Initial heading does not match case protocol")
                active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
                failed = torch.zeros_like(active)
                timed_out = torch.zeros_like(active)
                trace = []
                for step in range(steps):
                    # Use original command implementation, with explicit case bounds.
                    term._update_command()
                    obs = env.get_observations()
                    cmd = term.command.clone()
                    if not torch.equal(obs["policy"][:, 9:12], cmd):
                        raise RuntimeError("Actor observation does not contain applied command")
                    if case["mode"] == "heading":
                        error = torch.atan2(torch.sin(case["target"] - robot.data.heading_w),
                                            torch.cos(case["target"] - robot.data.heading_w))
                        expected = torch.clamp(0.5 * error, -1, 1)
                        if not torch.allclose(cmd[:, 2], expected, atol=1e-6):
                            raise RuntimeError("Heading command differs from training feedback law")
                    with torch.inference_mode():
                        actions = policy(obs)
                    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1, 1))
                    # [valid, root pose7, world vel3, body vel3, world omega3,
                    # body omega3, heading, tilt, applied command3]
                    sample = torch.cat((active[:, None], robot.data.root_pos_w,
                                        robot.data.root_quat_w, robot.data.root_lin_vel_w,
                                        robot.data.root_lin_vel_b, robot.data.root_ang_vel_w,
                                        robot.data.root_ang_vel_b, robot.data.heading_w[:, None],
                                        tilt[:, None], cmd), dim=1)
                    trace.append(sample.detach().cpu().numpy())
                    _, _, dones, _ = env.step(actions)
                    new_done = active & dones.bool()
                    failed |= new_done & base.termination_manager.terminated
                    timed_out |= new_done & base.termination_manager.time_outs
                    active &= ~new_done
                    if not active.any():
                        break
                data = np.stack(trace)
                times = np.arange(len(data)) * base.step_dt
                mask = (data[:, :, 0] > 0) & (times[:, None] >= min(2.0, args.duration / 2))
                if not mask.any():
                    raise RuntimeError("No eligible samples")
                body_vx, body_wz, world_wz, commands = data[:, :, 11], data[:, :, 19], data[:, :, 16], data[:, :, 22:25]
                heading_error = np.arctan2(np.sin(case["target"] - data[:, :, 20]), np.cos(case["target"] - data[:, :, 20]))
                final = [data[np.flatnonzero(data[:, e, 0])[-1], e] for e in range(args.num_envs)]
                final = np.stack(final)
                row = dict(actor=actor, case_id=case_id, **case, observed_s=len(data) * base.step_dt,
                           falls=int(failed.sum()), timeouts=int(timed_out.sum()), completed=int(active.sum()),
                           mean_body_vx=float(body_vx[mask].mean()),
                           forward_rmse=float(np.sqrt(((body_vx - commands[:, :, 0])[mask] ** 2).mean())),
                           body_yaw_rmse=float(np.sqrt(((body_wz - commands[:, :, 2])[mask] ** 2).mean())),
                           world_yaw_rmse=float(np.sqrt(((world_wz - commands[:, :, 2])[mask] ** 2).mean())),
                           mean_body_wz=float(body_wz[mask].mean()), mean_world_wz=float(world_wz[mask].mean()),
                           final_heading_error_abs=float(np.abs(np.arctan2(np.sin(case["target"] - final[:, 20]), np.cos(case["target"] - final[:, 20]))).mean()) if case["mode"] == "heading" else None,
                           max_tilt_deg=float(np.rad2deg(data[:, :, 21][mask]).max()),
                           final_heading_mean=float(final[:, 20].mean()),
                           mean_applied_yaw=float(commands[:, :, 2][mask].mean()))
                path = args.output_dir / f"{actor}_{case_id:02d}"
                np.savez_compressed(str(path) + ".npz", time=times, trace=data)
                Path(str(path) + ".json").write_text(json.dumps(row, indent=2))
                rows.append(row)
                with (args.output_dir / "summary.csv").open("w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print("FLAT_CASE " + json.dumps(row), flush=True)
        if any(sha256(p) != digest for p, digest in protocol["source_hashes"].items()):
            raise RuntimeError("Source changed during evaluation")
        if any(sha256(actor_paths[k]) != item["sha256"] for k, item in protocol["checkpoint_hashes"].items()):
            raise RuntimeError("Checkpoint changed during evaluation")
        (args.output_dir / "complete.json").write_text(json.dumps(dict(cases=len(rows), source_hashes_verified=True)))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
