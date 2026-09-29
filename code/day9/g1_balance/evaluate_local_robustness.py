"""Evaluate near-zero perturbations without changing the training environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import mujoco
import numpy as np
import torch

import experiment_config as config
from g1_env import G1Env
from networks import Actor


def snapshot(env):
    obs = env.get_observation()
    return {"qpos": env.data.qpos.tolist(), "qvel": env.data.qvel.tolist(),
            "pitch": float(obs[59]), "pitch_rate": float(obs[62]),
            "height": float(env.data.qpos[2])}


def episode(actor, device, pitch_delta, rate_delta):
    env = G1Env()
    env.reset()
    before = snapshot(env)
    # Relative rotation about the base-local y axis, preserving reset attitude.
    if pitch_delta:
        rotation = np.array([np.cos(pitch_delta / 2), 0.,
                             np.sin(pitch_delta / 2), 0.])
        result = np.empty(4)
        mujoco.mju_mulQuat(result, env.data.qpos[3:7].copy(), rotation)
        env.data.qpos[3:7] = result
    if rate_delta:
        env.data.qvel[4] += rate_delta
    if pitch_delta or rate_delta:
        mujoco.mj_forward(env.model, env.data)
    initial = snapshot(env)
    obs = env.get_observation()
    rows = []
    total = 0.
    for step in range(env.max_episode_steps):
        with torch.no_grad():
            action = (np.zeros(env.ACTION_DIM, dtype=np.float32) if actor is None
                      else torch.tanh(actor(torch.as_tensor(obs, device=device)).mean)
                      .cpu().numpy().astype(np.float32))
        row = {"step": step, "time": float(env.data.time),
               "pitch_before": float(obs[59]), "pitch_rate_before": float(obs[62]),
               "height_before": float(env.data.qpos[2]), "action": action.tolist(),
               "target_offset_rad": (env.action_scale * action).tolist()}
        obs, reward, terminated, truncated = env.step(action)
        total += reward
        row.update(pitch_after=float(obs[59]), pitch_rate_after=float(obs[62]),
                   height_after=float(env.data.qpos[2]), reward=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated))
        rows.append(row)
        if terminated or truncated:
            break
    actions = np.asarray([row["action"] for row in rows])
    return {"length": len(rows), "return": float(total),
            "action_rms": float(np.sqrt(np.mean(actions ** 2))),
            "action_max_abs": float(np.max(np.abs(actions))),
            "reset_state": before, "initial_state": initial, "trace": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.CHECKPOINT_DIR / config.BEST_DETERMINISTIC_CHECKPOINT)
    args = parser.parse_args()
    path = args.checkpoint.resolve(strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(config.OBS_DIM, config.ACTION_DIM).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = path.parent / "evaluations" / ("near_zero_" + stamp + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    cases = [("nominal", 0., 0.)]
    cases += [("pitch", p, 0.) for p in [-.005, -.002, -.001, .001, .002, .005]]
    cases += [("pitch_rate", 0., r) for r in [-.05, -.02, -.01, .01, .02, .05]]
    results = []
    for index, (kind, pitch, rate) in enumerate(cases):
        zero = episode(None, device, pitch, rate)
        ppo = episode(actor, device, pitch, rate)
        assert zero["initial_state"] == ppo["initial_state"], "Unpaired initial states"
        for label, result in [("zero", zero), ("ppo", ppo)]:
            trace_name = f"{index:02d}_{kind}_{label}.json"
            with (output / trace_name).open("x") as stream:
                json.dump(result.pop("trace"), stream, indent=2, allow_nan=False)
            result["trace_file"] = trace_name
        delta = ppo["length"] - zero["length"]
        results.append(dict(kind=kind, pitch_delta=pitch, pitch_rate_delta=rate,
                            zero=zero, ppo=ppo, delta_steps=delta))
        print(f"{kind:10s} pitch={pitch:+.3f} rate={rate:+.3f} "
              f"zero={zero['length']} ppo={ppo['length']} delta={delta:+d} "
              f"action_rms={ppo['action_rms']:.6g}", flush=True)
    deltas = [r["delta_steps"] for r in results if r["kind"] != "nominal"]
    report = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "update": checkpoint.get("update"), "device": str(device), "seed": config.SEED,
              "protocol": "relative base-local pitch rotation and additive qvel[4] after reset; zero leaves reset untouched",
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in [Path(__file__).name, "g1_env.py", "networks.py", "experiment_config.py"]},
              "cases": results, "nonzero_summary": {"cases": len(deltas),
              "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
              "losses": sum(d < 0 for d in deltas)}}
    with (output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print("Saved:", output)


if __name__ == "__main__":
    main()
