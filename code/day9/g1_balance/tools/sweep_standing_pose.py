import numpy as np

from g1_env import G1Env


# ============================================================
# Run zero-action standing test
# ============================================================

def run_pose(
    hip_pitch,
    knee,
):

    env = G1Env()

    # --------------------------------------------------------
    # Preserve approximate sagittal-chain orientation:
    #
    # hip + knee + ankle = 0
    # --------------------------------------------------------

    ankle_pitch = -(
        hip_pitch
        + knee
    )

    # Left leg
    env.default_q[0] = hip_pitch
    env.default_q[3] = knee
    env.default_q[4] = ankle_pitch

    # Right leg
    env.default_q[6] = hip_pitch
    env.default_q[9] = knee
    env.default_q[10] = ankle_pitch

    # IMPORTANT:
    # reset again after changing default_q
    obs = env.reset()

    zero_action = np.zeros(
        env.ACTION_DIM,
        dtype=np.float32
    )

    total_reward = 0.0

    max_abs_pitch = 0.0
    max_abs_rate = 0.0

    mean_abs_pitch = 0.0
    mean_abs_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    initial_height = float(
        env.data.qpos[2]
    )

    initial_contacts = int(
        env.data.ncon
    )

    for step in range(
        env.max_episode_steps
    ):

        (
            obs,
            reward,
            terminated,
            truncated,
        ) = env.step(
            zero_action
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

        max_abs_rate = max(
            max_abs_rate,
            abs(pitch_rate)
        )

        mean_abs_pitch += abs(
            pitch
        )

        mean_abs_rate += abs(
            pitch_rate
        )

        min_height = min(
            min_height,
            height
        )

        if terminated or truncated:
            break

    length = step + 1

    return {
        "hip":
            hip_pitch,

        "knee":
            knee,

        "ankle":
            ankle_pitch,

        "length":
            length,

        "return":
            total_reward,

        "mean_pitch":
            mean_abs_pitch
            / length,

        "mean_rate":
            mean_abs_rate
            / length,

        "max_pitch":
            max_abs_pitch,

        "max_rate":
            max_abs_rate,

        "initial_height":
            initial_height,

        "min_height":
            min_height,

        "contacts":
            initial_contacts,
    }


# ============================================================
# Main sweep
# ============================================================

def main():

    hip_values = [
        -0.25,
        -0.20,
        -0.15,
        -0.10,
        -0.05,
         0.00,
    ]

    knee_values = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
    ]

    results = []

    print(
        "=== STANDING POSE SWEEP ==="
    )

    print(
        "Zero policy. Fixed joint PD."
    )

    print()

    for hip in hip_values:

        for knee in knee_values:

            result = run_pose(
                hip_pitch=hip,
                knee=knee,
            )

            results.append(
                result
            )

            print(
                f"hip={result['hip']:+.2f} | "
                f"knee={result['knee']:+.2f} | "
                f"ankle={result['ankle']:+.2f} | "
                f"len={result['length']:4d} | "
                f"return={result['return']:8.2f} | "
                f"mean_pitch={result['mean_pitch']:.3f} | "
                f"mean_rate={result['mean_rate']:.3f} | "
                f"min_h={result['min_height']:.3f}"
            )

    # --------------------------------------------------------
    # Ranking:
    #
    # 1. Survival
    # 2. Higher minimum height
    # 3. Smaller mean pitch
    # 4. Smaller mean angular velocity
    # --------------------------------------------------------

    results.sort(
        key=lambda x: (
            x["length"],
            x["min_height"],
            -x["mean_pitch"],
            -x["mean_rate"],
        ),
        reverse=True,
    )

    print()
    print(
        "=== TOP 10 STANDING POSES ==="
    )

    print()

    for rank, result in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"hip={result['hip']:+.2f} | "
            f"knee={result['knee']:+.2f} | "
            f"ankle={result['ankle']:+.2f} | "
            f"len={result['length']:4d} | "
            f"return={result['return']:8.2f} | "
            f"mean_pitch={result['mean_pitch']:.3f} | "
            f"mean_rate={result['mean_rate']:.3f} | "
            f"min_h={result['min_height']:.3f}"
        )


if __name__ == "__main__":
    main()