import numpy as np

from g1_env import G1Env


# ------------------------------------------------------------
# G1 action indices
# ------------------------------------------------------------

JOINT_GROUPS = {
    "hip_pitch": [0, 3],
    "knee": [1, 4],
    "ankle_pitch": [2, 5],
}

TEST_AMPLITUDE = 0.5

# Run only briefly.
# We want the initial physical response,
# not the final fall.
TEST_STEPS = 10


def run_direction_test(
    joint_name,
    joint_indices,
    sign
):
    env = G1Env()

    # Use a small fixed action scale for controlled diagnosis.
    env.action_scale = 0.10

    obs = env.reset()

    initial_pitch = float(
        obs[59]
    )

    initial_pitch_rate = float(
        obs[62]
    )

    action = np.zeros(
        env.ACTION_DIM,
        dtype=np.float32
    )

    for index in joint_indices:
        action[index] = (
            sign
            * TEST_AMPLITUDE
        )

    max_pitch = abs(
        initial_pitch
    )

    for step in range(
        TEST_STEPS
    ):
        (
            obs,
            reward,
            terminated,
            truncated
        ) = env.step(
            action
        )

        max_pitch = max(
            max_pitch,
            abs(float(obs[59]))
        )

        if terminated or truncated:
            break

    final_pitch = float(
        obs[59]
    )

    final_pitch_rate = float(
        obs[62]
    )

    return {
        "joint":
            joint_name,

        "sign":
            sign,

        "initial_pitch":
            initial_pitch,

        "final_pitch":
            final_pitch,

        "delta_pitch":
            final_pitch
            - initial_pitch,

        "initial_pitch_rate":
            initial_pitch_rate,

        "final_pitch_rate":
            final_pitch_rate,

        "max_pitch":
            max_pitch,
    }

def run_zero_baseline():

    env = G1Env()

    env.action_scale = 0.10

    obs = env.reset()

    action = np.zeros(
        env.ACTION_DIM,
        dtype=np.float32
    )

    for _ in range(TEST_STEPS):

        (
            obs,
            reward,
            terminated,
            truncated
        ) = env.step(
            action
        )

    return {
        "pitch":
            float(obs[59]),

        "pitch_rate":
            float(obs[62])
    }

def main():

    print(
        "=== JOINT DIRECTION DIAGNOSTIC ==="
    )

    print()

    baseline = (
        run_zero_baseline()
    )

    print(
        "ZERO BASELINE    | "
        f"pitch={baseline['pitch']:+.4f} | "
        f"pitch_rate={baseline['pitch_rate']:+.4f}"
    )

    print()

    print(
        "Positive delta_pitch means "
        "more positive pitch."
    )

    print()

    for (
        joint_name,
        joint_indices
    ) in JOINT_GROUPS.items():

        for sign in [
            -1.0,
            +1.0
        ]:

            result = (
                run_direction_test(
                    joint_name,
                    joint_indices,
                    sign
                )
            )

            print(
                f"{result['joint']:12s} | "
                f"action={result['sign']:+.1f} | "
                f"pitch={result['final_pitch']:+.4f} | "
                f"d_pitch={result['delta_pitch']:+.4f} | "
                f"vs_zero_pitch="
                f"{result['final_pitch'] - baseline['pitch']:+.4f} | "
                f"pitch_rate={result['final_pitch_rate']:+.4f} | "
                f"vs_zero_rate="
                f"{result['final_pitch_rate'] - baseline['pitch_rate']:+.4f}"
            )

        print()


if __name__ == "__main__":
    main()