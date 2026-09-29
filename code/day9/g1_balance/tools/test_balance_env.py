import numpy as np

from g1_env import G1Env


def main():

    env = G1Env()

    obs = env.reset()

    print(
        "=== DAY 9 BALANCE ENV TEST ==="
    )

    print(
        "Observation shape:",
        obs.shape
    )

    print(
        "Policy action dim:",
        env.ACTION_DIM
    )

    print(
        "Policy joint indices:",
        env.POLICY_JOINT_INDICES
    )

    print(
        "Reference height:",
        env.reference_height
    )

    print()

    zero_action = np.zeros(
        env.ACTION_DIM,
        dtype=np.float32
    )

    total_reward = 0.0

    for step in range(
        100
    ):

        (
            obs,
            reward,
            terminated,
            truncated
        ) = env.step(
            zero_action
        )

        total_reward += reward

        if step % 10 == 0:

            print(
                f"step={step:3d} | "
                f"height={env.data.qpos[2]:.3f} | "
                f"pitch={obs[59]:+.3f} | "
                f"gyro_y={obs[62]:+.3f} | "
                f"reward={reward:+.3f}"
            )

        if (
            terminated
            or truncated
        ):

            print()

            print(
                "Episode ended at:",
                step + 1
            )

            print(
                "Total reward:",
                total_reward
            )

            print(
                "terminated:",
                terminated
            )

            break


if __name__ == "__main__":
    main()