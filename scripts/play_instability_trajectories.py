"""Play saved zero-PD and PPO instability runs in a MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from g1_control.envs.balance_env import G1Env


DEFAULT_RUN = (
    PROJECT_ROOT
    / "code/day9/g1_balance/checkpoints/instability_trajectories/phase1_20260930T002805Z"
)


def wait_frame(model, viewer, data, qpos, qvel, frame_period: float) -> bool:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(frame_period)
    return viewer.is_running()


def play(viewer, env: G1Env, run_dir: Path, policy: str, speed: float) -> None:
    arrays = np.load(run_dir / f"{policy}.npz")
    qpos = arrays["qpos"]
    qvel = arrays["qvel"]
    frame_period = env.decimation * env.model.opt.timestep / speed
    print(f"Playing {policy}: {len(qpos) - 1} policy steps at {speed:g}x real time")
    for frame in range(len(qpos)):
        if not wait_frame(env.model, viewer, env.data, qpos[frame], qvel[frame], frame_period):
            return
    print(f"Finished {policy}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--policy", choices=("both", "zero_pd", "best_ppo"), default="both")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback rate; 1 means recorded 50 Hz wall-clock")
    parser.add_argument("--pause-after-first", action="store_true", help="Wait for Enter between zero PD and PPO")
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    selected = ("zero_pd", "best_ppo") if args.policy == "both" else (args.policy,)
    for policy in selected:
        if not (run_dir / f"{policy}.npz").is_file():
            raise FileNotFoundError(run_dir / f"{policy}.npz")

    env = G1Env()
    env.reset()
    pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise RuntimeError("Cannot find pelvis body for tracking camera")

    print(f"Run: {run_dir}")
    print("Close the MuJoCo window to stop playback.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -12
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.sync()

        for index, policy in enumerate(selected):
            if index and args.pause_after_first:
                input("Press Enter to play the PPO trajectory...")
            if not viewer.is_running():
                break
            play(viewer, env, run_dir, policy, args.speed)

        if viewer.is_running():
            input("Playback finished. Press Enter to close the MuJoCo viewer...")


if __name__ == "__main__":
    main()
