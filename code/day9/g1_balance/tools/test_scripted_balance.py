import numpy as np

from g1_env import G1Env


def get_scripted_action(obs):

    # --------------------------------------------------------
    # Observation
    #
    # obs[59] = pitch
    # obs[62] = pitch angular velocity
    # --------------------------------------------------------

    pitch = float(obs[59])
    pitch_rate = float(obs[62])

    # --------------------------------------------------------
    # High-level body pitch feedback
    #
    # Positive pitch / pitch-rate:
    # robot is moving toward the forward-fall direction.
    #
    # From our empirical direction test:
    #
    # positive hip-pitch action   → helpful
    # positive ankle-pitch action → helpful
    #
    # Knee response was ambiguous,
    # so keep knee correction at zero first.
    # --------------------------------------------------------

    correction = (
        1.5 * pitch
        + 0.20 * pitch_rate
    )

    correction = np.clip(
        correction,
        -1.0,
        1.0
    )

    action = np.zeros(
        6,
        dtype=np.float32
    )

    # Day 9 policy mapping:
    #
    # 0 left hip pitch
    # 1 left knee
    # 2 left ankle pitch
    # 3 right hip pitch
    # 4 right knee
    # 5 right ankle pitch

    action[0] = correction
    action[2] = correction

    action[3] = correction
    action[5] = correction

    return action


def run_episode(
    controller="zero"
):

    env = G1Env()

    obs = env.reset()

    total_reward = 0.0

    max_abs_pitch = 0.0
    max_abs_pitch_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    for step in range(
        env.max_episode_steps
    ):

        if controller == "zero":

            action = np.zeros(
                6,
                dtype=np.float32
            )

        else:

            action = (
                get_scripted_action(
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

        pitch = float(
            obs[59]
        )

        pitch_rate = float(
            obs[62]
        )

        height = float(
            env.data.qpos[2]
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(pitch)
        )

        max_abs_pitch_rate = max(
            max_abs_pitch_rate,
            abs(pitch_rate)
        )

        min_height = min(
            min_height,
            height
        )

        if terminated or truncated:

            break

    return {
        "length":
            step + 1,

        "return":
            total_reward,

        "max_pitch":
            max_abs_pitch,

        "max_pitch_rate":
            max_abs_pitch_rate,

        "min_height":
            min_height
    }


def print_result(
    name,
    result
):

    print(
        f"{name:10s} | "
        f"length={result['length']:4d} | "
        f"return={result['return']:8.2f} | "
        f"max_pitch={result['max_pitch']:.3f} | "
        f"max_rate={result['max_pitch_rate']:.3f} | "
        f"min_height={result['min_height']:.3f}"
    )


def main():

    print(
        "=== SCRIPTED BALANCE TEST ==="
    )

    print()

    zero_result = (
        run_episode(
            "zero"
        )
    )

    scripted_result = (
        run_episode(
            "scripted"
        )
    )

    print_result(
        "zero",
        zero_result
    )

    print_result(
        "scripted",
        scripted_result
    )


if __name__ == "__main__":
    main()