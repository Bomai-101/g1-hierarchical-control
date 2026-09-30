"""Verify that saved Phase 1 MuJoCo integration states replay exactly one step."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.envs.balance_env import G1Env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve(strict=True)
    for policy in ("zero_pd", "best_ppo"):
        arrays = np.load(directory / f"{policy}.npz")
        snapshots = arrays["integration_state"]
        checks = sorted({0, 50, 100, len(arrays["action"]) // 2, len(arrays["action"]) - 2})
        for boundary in checks:
            env = G1Env()
            env.reset()
            mujoco.mj_setState(
                env.model, env.data, snapshots[boundary],
                mujoco.mjtState.mjSTATE_INTEGRATION,
            )
            env.episode_step = boundary
            env.step(arrays["action"][boundary])
            qpos_error = float(np.max(np.abs(env.data.qpos - arrays["qpos"][boundary + 1])))
            qvel_error = float(np.max(np.abs(env.data.qvel - arrays["qvel"][boundary + 1])))
            if qpos_error > 1e-10 or qvel_error > 1e-10:
                raise RuntimeError(
                    f"{policy} boundary {boundary} failed replay: "
                    f"qpos={qpos_error:.3g}, qvel={qvel_error:.3g}"
                )
            print(f"{policy} boundary={boundary:3d} qpos_error={qpos_error:.2g} qvel_error={qvel_error:.2g}")
    print("integration-state one-step replay: PASS")


if __name__ == "__main__":
    main()
