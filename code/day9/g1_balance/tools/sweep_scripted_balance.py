import numpy as np

from g1_env import G1Env


# ============================================================
# Controller
# ============================================================

def get_scripted_action(
    obs,
    kp_body,
    kd_body,
    hip_scale=1.0,
    ankle_scale=1.0,
):

    pitch = float(obs[59])
    pitch_rate = float(obs[62])

    correction = (
        kp_body * pitch
        + kd_body * pitch_rate
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

    # 6D policy mapping:
    #
    # 0 left hip pitch
    # 1 left knee
    # 2 left ankle pitch
    # 3 right hip pitch
    # 4 right knee
    # 5 right ankle pitch

    hip_action = np.clip(
        hip_scale * correction,
        -1.0,
        1.0
    )

    ankle_action = np.clip(
        ankle_scale * correction,
        -1.0,
        1.0
    )

    action[0] = hip_action
    action[3] = hip_action

    action[2] = ankle_action
    action[5] = ankle_action

    # Knee stays zero for now.

    return action


# ============================================================
# Run one episode
# ============================================================

def run_episode(
    kp_body,
    kd_body,
    hip_scale=1.0,
    ankle_scale=1.0,
):

    env = G1Env()

    obs = env.reset()

    total_reward = 0.0

    max_abs_pitch = 0.0
    max_abs_pitch_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    mean_abs_pitch_sum = 0.0
    mean_abs_pitch_rate_sum = 0.0

    for step in range(
        env.max_episode_steps
    ):

        action = get_scripted_action(
            obs=obs,
            kp_body=kp_body,
            kd_body=kd_body,
            hip_scale=hip_scale,
            ankle_scale=ankle_scale,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
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

        mean_abs_pitch_sum += abs(
            pitch
        )

        mean_abs_pitch_rate_sum += abs(
            pitch_rate
        )

        if terminated or truncated:
            break

    length = step + 1

    return {
        "length":
            length,

        "return":
            total_reward,

        "max_pitch":
            max_abs_pitch,

        "max_rate":
            max_abs_pitch_rate,

        "mean_pitch":
            mean_abs_pitch_sum
            / length,

        "mean_rate":
            mean_abs_pitch_rate_sum
            / length,

        "min_height":
            min_height,
    }


# ============================================================
# Phase 1:
# Sweep body feedback Kp / Kd
# ============================================================

def phase_1():

    kp_values = [
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
    ]

    kd_values = [
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]

    results = []

    print(
        "=== PHASE 1: BODY KP / KD SWEEP ==="
    )

    print()

    for kp in kp_values:

        for kd in kd_values:

            result = run_episode(
                kp_body=kp,
                kd_body=kd,
                hip_scale=1.0,
                ankle_scale=1.0,
            )

            result["kp"] = kp
            result["kd"] = kd

            results.append(
                result
            )

            print(
                f"Kp={kp:4.2f} | "
                f"Kd={kd:4.2f} | "
                f"len={result['length']:4d} | "
                f"return={result['return']:8.2f} | "
                f"mean_pitch={result['mean_pitch']:.3f} | "
                f"mean_rate={result['mean_rate']:.3f} | "
                f"min_h={result['min_height']:.3f}"
            )

    # Primary criterion:
    # longest survival
    #
    # Secondary:
    # avoid excessive sinking
    results.sort(
        key=lambda x: (
            x["length"],
            x["min_height"],
            x["return"],
        ),
        reverse=True,
    )

    best = results[0]

    print()

    print(
        "=== BEST PHASE 1 ==="
    )

    print(
        f"Kp={best['kp']:.2f} | "
        f"Kd={best['kd']:.2f} | "
        f"length={best['length']} | "
        f"return={best['return']:.2f} | "
        f"mean_pitch={best['mean_pitch']:.3f} | "
        f"mean_rate={best['mean_rate']:.3f} | "
        f"min_height={best['min_height']:.3f}"
    )

    return (
        best["kp"],
        best["kd"]
    )


# ============================================================
# Phase 2:
# Separate hip / ankle contribution
# ============================================================

def phase_2(
    best_kp,
    best_kd
):

    scales = [
        0.5,
        1.0,
        1.5,
        2.0,
    ]

    results = []

    print()
    print(
        "=== PHASE 2: HIP / ANKLE SCALE SWEEP ==="
    )

    print()

    for hip_scale in scales:

        for ankle_scale in scales:

            result = run_episode(
                kp_body=best_kp,
                kd_body=best_kd,
                hip_scale=hip_scale,
                ankle_scale=ankle_scale,
            )

            result[
                "hip_scale"
            ] = hip_scale

            result[
                "ankle_scale"
            ] = ankle_scale

            results.append(
                result
            )

            print(
                f"hip={hip_scale:3.1f} | "
                f"ankle={ankle_scale:3.1f} | "
                f"len={result['length']:4d} | "
                f"return={result['return']:8.2f} | "
                f"mean_pitch={result['mean_pitch']:.3f} | "
                f"mean_rate={result['mean_rate']:.3f} | "
                f"min_h={result['min_height']:.3f}"
            )

    results.sort(
        key=lambda x: (
            x["length"],
            x["min_height"],
            x["return"],
        ),
        reverse=True,
    )

    best = results[0]

    print()

    print(
        "=== BEST SCRIPTED CONTROLLER ==="
    )

    print(
        f"Kp={best_kp:.2f} | "
        f"Kd={best_kd:.2f} | "
        f"hip={best['hip_scale']:.1f} | "
        f"ankle={best['ankle_scale']:.1f}"
    )

    print(
        f"length={best['length']} | "
        f"return={best['return']:.2f} | "
        f"max_pitch={best['max_pitch']:.3f} | "
        f"max_rate={best['max_rate']:.3f} | "
        f"mean_pitch={best['mean_pitch']:.3f} | "
        f"mean_rate={best['mean_rate']:.3f} | "
        f"min_height={best['min_height']:.3f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    best_kp, best_kd = (
        phase_1()
    )

    phase_2(
        best_kp,
        best_kd
    )


if __name__ == "__main__":
    main()