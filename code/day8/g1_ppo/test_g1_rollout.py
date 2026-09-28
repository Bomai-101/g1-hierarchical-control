import time

import mujoco.viewer
import numpy as np
import torch

from g1_env import G1Env
from networks import Actor, Critic


OBS_DIM = 64
ACTION_DIM = 29


def main():

    torch.manual_seed(42)

    # -------------------------
    # 1. Environment
    # -------------------------

    env = G1Env()

    # -------------------------
    # 2. Actor / Critic
    # -------------------------

    actor = Actor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM
    )

    critic = Critic(
        obs_dim=OBS_DIM
    )

    # We are NOT training yet.
    actor.eval()
    critic.eval()

    # -------------------------
    # 3. Reset environment
    # -------------------------

    obs = env.reset()

    print(
        "Initial observation shape:",
        obs.shape
    )

    total_reward = 0.0

    # 50 Hz policy
    policy_dt = (
        env.model.opt.timestep
        * env.decimation
    )

    # -------------------------
    # 4. Launch MuJoCo viewer
    # -------------------------

    viewer = mujoco.viewer.launch_passive(
        env.model,
        env.data
    )

    try:

        for step in range(1000):

            if not viewer.is_running():
                break

            step_start = time.time()

            obs_tensor = torch.from_numpy(
                obs
            ).float()

            with torch.no_grad():

                dist = actor(
                    obs_tensor
                )

                raw_action_tensor = (
                    dist.mean
                )

                action_tensor = torch.tanh(
                    raw_action_tensor
                )

                value = critic(
                    obs_tensor
                ).squeeze(-1)

            action = (
                action_tensor
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            (
                next_obs,
                reward,
                terminated,
                truncated
            ) = env.step(
                action
            )

            total_reward += reward

            viewer.sync()

            if step % 10 == 0:

                print(
                    f"step={step:4d} | "
                    f"height={env.data.qpos[2]:.3f} | "
                    f"pitch={next_obs[59]:.3f} | "
                    f"reward={reward:.3f} | "
                    f"value={value.item():.3f}"
                )

            obs = next_obs

            if terminated or truncated:

                print()
                print("Episode ended.")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("episode steps:", step + 1)
                print("total reward:", total_reward)

                viewer.sync()
                time.sleep(0.1)

                break

            # Keep visualization near real time
            elapsed = (
                time.time()
                - step_start
            )

            remaining = (
                policy_dt
                - elapsed
            )

            if remaining > 0:
                time.sleep(
                    remaining
                )

    finally:

        if viewer.is_running():
            viewer.close()

if __name__ == "__main__":
    main()