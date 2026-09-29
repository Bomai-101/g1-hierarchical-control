import numpy as np

from g1_env import G1Env


ACTION_SCALES = [
    0.10,
    0.20,
    0.25
]


def scripted_balance_action(
    obs
):

    action = np.zeros(
        29,
        dtype=np.float32
    )

    # Observation:
    #
    # obs[58] = roll
    # obs[59] = pitch
    # obs[60] = yaw
    #
    # obs[61:64] = base angular velocity

    pitch = float(
        obs[59]
    )

    pitch_rate = float(
        obs[62]
    )

    # --------------------------------------------------------
    # Very simple pitch-feedback controller.
    #
    # This is NOT the final controller.
    # It is only an action-authority diagnostic.
    # --------------------------------------------------------

    correction = (
        1.5 * pitch
        + 0.15 * pitch_rate
    )

    correction = np.clip(
        correction,
        -1.0,
        1.0
    )

    # Left leg:
    #
    # 0 hip pitch
    # 3 knee
    # 4 ankle pitch

    action[0] = -correction
    action[3] = correction
    action[4] = -correction

    # Right leg:
    #
    # 6 hip pitch
    # 9 knee
    # 10 ankle pitch

    action[6] = -correction
    action[9] = correction
    action[10] = -correction

    return action


def run_test(
    action_scale
):

    env = G1Env()

    env.action_scale = (
        action_scale
    )

    obs = env.reset()

    total_reward = 0.0

    max_pitch = 0.0
    min_height = float(
        env.data.qpos[2]
    )

    for step in range(
        1000
    ):

        action = (
            scripted_balance_action(
                obs
            )
        )

        (
            obs,
            reward,
            terminated,
            truncated
        ) = env.step(
            action
        )

        total_reward += reward

        pitch = abs(
            float(obs[59])
        )

        height = float(
            env.data.qpos[2]
        )

        max_pitch = max(
            max_pitch,
            pitch
        )

        min_height = min(
            min_height,
            height
        )

        if (
            terminated
            or truncated
        ):

            break

    return {
        "action_scale":
            action_scale,

        "length":
            step + 1,

        "return":
            total_reward,

        "max_pitch":
            max_pitch,

        "min_height":
            min_height,
    }


def main():

    print(
        "=== ACTION AUTHORITY TEST ==="
    )

    print()

    for scale in ACTION_SCALES:

        result = run_test(
            scale
        )

        print(
            f"scale={result['action_scale']:.2f} | "
            f"length={result['length']:4d} | "
            f"return={result['return']:8.2f} | "
            f"max_pitch={result['max_pitch']:.3f} | "
            f"min_height={result['min_height']:.3f}"
        )


if __name__ == "__main__":
    main()