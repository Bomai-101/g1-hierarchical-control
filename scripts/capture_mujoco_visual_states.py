"""Capture fixed-command reference MuJoCo rollouts for visual replay."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_root / "mujoco"))
    import mujoco
    import numpy as np
    from mujoco_eval import run_grid
    model_path = args.reference_root / "mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml"
    metadata = args.reference_root / "mujoco/policies/isaac_metadata.json"
    files = [model_path, metadata, args.baseline, args.candidate, Path(__file__).resolve()]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output_dir / "inputs.json").write_text(json.dumps(hashes, indent=2))
    for label, policy in (("baseline1499", args.baseline), ("candidate1999", args.candidate)):
        cfg = SimpleNamespace(model=str(model_path), policy=str(policy.resolve()), metadata=str(metadata),
            command_x=1., command_y=0., command_yaw=0., timestep=.001, duration=args.duration,
            seed=42, device="cpu", allow_missing_joints=False, initial_base_height=.74,
            min_base_height=.35, base_body="pelvis", control_mode="pd")
        states, times, commands = [], [], []
        actual_data = None
        steps = 0
        original_forward, original_step = mujoco.mj_forward, mujoco.mj_step
        original_act = run_grid.TorchActorPolicy.act
        def sample(data):
            states.append(np.r_[data.qpos, data.qvel].copy())
            times.append(float(data.time))
        def forward(model, data, *values, **kwargs):
            nonlocal actual_data
            result = original_forward(model, data, *values, **kwargs)
            if actual_data is None:
                actual_data = data
                mujoco.mj_saveModel(model, str(args.output_dir / f"{label}.mjb"))
                sample(data)
            return result
        def step(model, data, *values, **kwargs):
            nonlocal steps
            result = original_step(model, data, *values, **kwargs)
            steps += 1
            if steps % 20 == 0:
                sample(data)
            return result
        def act(policy_object, obs):
            if not np.array_equal(obs[9:12], np.array([1., 0., 0.], dtype=obs.dtype)):
                raise RuntimeError("Actor command mismatch")
            commands.append(obs[9:12].copy())
            return original_act(policy_object, obs)
        with patch.object(mujoco, "mj_forward", forward), patch.object(mujoco, "mj_step", step), patch.object(run_grid.TorchActorPolicy, "act", act):
            metrics = run_grid.run_once(cfg, "plane", .8)
        if times[-1] < actual_data.time - 1e-8:
            sample(actual_data)
        np.savez_compressed(args.output_dir / f"{label}.npz", states=states, time=times, commands=commands)
        (args.output_dir / f"{label}.json").write_text(json.dumps(metrics, indent=2))
        print("MUJOCO_STATES " + label + " " + str(metrics["elapsed_s"]), flush=True)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Input changed")
    (args.output_dir / "complete.json").write_text(json.dumps(dict(command=[1, 0, 0], duration=args.duration,
        mujoco=mujoco.__version__, policy_step=.02, physics_step=.001, no_monitor=True), indent=2))


if __name__ == "__main__":
    main()
