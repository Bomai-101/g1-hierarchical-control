"""Continue the unchanged reference flat PPO task into an isolated run.

The original checkpoint is preserved. Rough terrain and new reward terms
are deliberately outside this continuation experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--additional-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.additional_iters <= 0:
        parser.error("additional iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ref = args.reference_root.resolve()
    checkpoint = args.checkpoint.resolve()
    paths = [checkpoint, Path(__file__).resolve(), ref / "isaac_sim/assets/g1_minimal.usd"]
    paths += [ref / "isaac_sim/g1_walk_sim51" / name for name in
              ("g1_env_cfg.py", "g1_asset_cfg.py", "mdp.py", "ppo_cfg.py")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    sys.path.insert(0, str(ref / "isaac_sim"))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import g1_walk_sim51  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from g1_walk_sim51.ppo_cfg import G1FlatPPORunnerCfg
        from rsl_rl.runners import OnPolicyRunner

        cfg = load_cfg_from_registry("G1-Walk-Flat-Sim51-v0", "env_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.log_dir = str(args.output_dir)
        agent_cfg = G1FlatPPORunnerCfg()
        agent_cfg.device = args.device
        agent_cfg.seed = args.seed
        agent_cfg.max_iterations = args.additional_iters
        dump_yaml(str(args.output_dir / "params/env.yaml"), cfg)
        dump_yaml(str(args.output_dir / "params/agent.yaml"), agent_cfg)
        env = RslRlVecEnvWrapper(gym.make("G1-Walk-Flat-Sim51-v0", cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(args.output_dir), device=args.device)
        runner.load(str(checkpoint))
        # Saved iter is the last completed update, not the next update.
        runner.current_learning_iteration += 1
        start = runner.current_learning_iteration
        protocol = dict(initial_checkpoint=str(checkpoint), seed=args.seed,
                        task="G1-Walk-Flat-Sim51-v0", num_envs=args.num_envs,
                        additional_iterations=args.additional_iters, first_iteration=start,
                        expected_last_iteration=start + args.additional_iters - 1,
                        transitions=args.additional_iters * args.num_envs * agent_cfg.num_steps_per_env,
                        reward_changes=False, command_changes=False, pd_changes=False,
                        optimizer_loaded=True, python=sys.version, torch=torch.__version__,
                        original_preserved=True, candidate_only=True)
        (args.output_dir / "training_protocol.json").write_text(json.dumps(protocol, indent=2))
        print("FLAT_TRAIN_START " + json.dumps(protocol), flush=True)
        runner.learn(num_learning_iterations=args.additional_iters, init_at_random_ep_len=True)
        final = args.output_dir / f"model_{runner.current_learning_iteration}.pt"
        if not final.is_file():
            raise RuntimeError("Expected final checkpoint missing")
        for p, expected in hashes.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Input changed: {p}")
        result = dict(protocol, final_checkpoint=str(final.resolve()),
                      final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(), inputs_unchanged=True)
        (args.output_dir / "complete.json").write_text(json.dumps(result, indent=2))
        print("FLAT_TRAIN_COMPLETE " + json.dumps(result), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
