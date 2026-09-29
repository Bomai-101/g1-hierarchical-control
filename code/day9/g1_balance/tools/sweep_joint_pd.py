import numpy as np

from g1_env import G1Env


# ============================================================
# Run one zero-policy episode with specified PD gains
# ============================================================

def run_pd(
    hip_kp,
    hip_kd,
    knee_kp,
    knee_kd,
    ankle_kp,
    ankle_kd,
):

    env = G1Env()

    # --------------------------------------------------------
    # Apply symmetric left/right sagittal PD gains
    #
    # G1 indices:
    #
    # 0, 6   hip pitch
    # 3, 9   knee
    # 4, 10  ankle pitch
    # --------------------------------------------------------

    for idx in [0, 6]:
        env.kp[idx] = hip_kp
        env.kd[idx] = hip_kd

    for idx in [3, 9]:
        env.kp[idx] = knee_kp
        env.kd[idx] = knee_kd

    for idx in [4, 10]:
        env.kp[idx] = ankle_kp
        env.kd[idx] = ankle_kd

    # Important:
    # reset again because __init__ already performed
    # a settling phase using the old gains.
    obs = env.reset()

    zero_action = np.zeros(
        env.ACTION_DIM,
        dtype=np.float32
    )

    total_reward = 0.0

    mean_abs_pitch = 0.0
    mean_abs_rate = 0.0

    max_abs_pitch = 0.0
    max_abs_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    terminated = False
    truncated = False

    for step in range(
        env.max_episode_steps
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

        pitch = float(obs[59])
        pitch_rate = float(obs[62])

        height = float(
            env.data.qpos[2]
        )

        mean_abs_pitch += abs(
            pitch
        )

        mean_abs_rate += abs(
            pitch_rate
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(pitch)
        )

        max_abs_rate = max(
            max_abs_rate,
            abs(pitch_rate)
        )

        min_height = min(
            min_height,
            height
        )

        if terminated or truncated:
            break

    length = step + 1

    # --------------------------------------------------------
    # Diagnose why episode ended
    # --------------------------------------------------------

    roll = float(obs[58])
    pitch = float(obs[59])
    height = float(
        env.data.qpos[2]
    )

    if truncated:
        reason = "timeout"

    elif height < 0.45:
        reason = "height"

    elif abs(roll) > 0.8:
        reason = "roll"

    elif abs(pitch) > 0.8:
        reason = "pitch"

    else:
        reason = "unknown"

    return {
        "length": length,
        "return": total_reward,
        "mean_pitch":
            mean_abs_pitch / length,
        "mean_rate":
            mean_abs_rate / length,
        "max_pitch":
            max_abs_pitch,
        "max_rate":
            max_abs_rate,
        "min_height":
            min_height,
        "reason":
            reason,
    }


# ============================================================
# Ranking helper
# ============================================================

def rank_results(
    results,
    title
):

    results.sort(
        key=lambda x: (
            x["length"],
            x["min_height"],
            -x["mean_pitch"],
            -x["mean_rate"],
        ),
        reverse=True
    )

    print()
    print(
        f"=== {title} ==="
    )
    print()

    for rank, r in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"len={r['length']:4d} | "
            f"return={r['return']:8.2f} | "
            f"mean_pitch={r['mean_pitch']:.3f} | "
            f"mean_rate={r['mean_rate']:.3f} | "
            f"min_h={r['min_height']:.3f} | "
            f"reason={r['reason']} | "
            f"Kp={r['tested_kp']:6.1f} | "
            f"Kd={r['tested_kd']:4.1f}"
        )

    return results[0]


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Starting gains
    # ========================================================

    best_hip_kp = 100.0
    best_hip_kd = 2.0

    best_knee_kp = 150.0
    best_knee_kd = 4.0

    best_ankle_kp = 40.0
    best_ankle_kd = 2.0

    # ========================================================
    # PHASE 1 — ANKLE
    # ========================================================

    print(
        "=== PHASE 1: ANKLE PD SWEEP ==="
    )

    ankle_kp_values = [
        20,
        30,
        40,
        50,
        60,
        80,
    ]

    ankle_kd_values = [
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    results = []

    for kp in ankle_kp_values:

        for kd in ankle_kd_values:

            r = run_pd(
                hip_kp=best_hip_kp,
                hip_kd=best_hip_kd,
                knee_kp=best_knee_kp,
                knee_kd=best_knee_kd,
                ankle_kp=kp,
                ankle_kd=kd,
            )

            r["tested_kp"] = kp
            r["tested_kd"] = kd

            results.append(r)

            print(
                f"ankle "
                f"Kp={kp:5.1f} | "
                f"Kd={kd:3.1f} | "
                f"len={r['length']:4d} | "
                f"mean_pitch={r['mean_pitch']:.3f} | "
                f"mean_rate={r['mean_rate']:.3f} | "
                f"min_h={r['min_height']:.3f} | "
                f"reason={r['reason']}"
            )

    best = rank_results(
        results,
        "BEST ANKLE GAINS"
    )

    best_ankle_kp = (
        best["tested_kp"]
    )

    best_ankle_kd = (
        best["tested_kd"]
    )

    # ========================================================
    # PHASE 2 — HIP
    # ========================================================

    print()
    print(
        "=== PHASE 2: HIP PD SWEEP ==="
    )

    hip_kp_values = [
        60,
        80,
        100,
        120,
        150,
    ]

    hip_kd_values = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    results = []

    for kp in hip_kp_values:

        for kd in hip_kd_values:

            r = run_pd(
                hip_kp=kp,
                hip_kd=kd,
                knee_kp=best_knee_kp,
                knee_kd=best_knee_kd,
                ankle_kp=best_ankle_kp,
                ankle_kd=best_ankle_kd,
            )

            r["tested_kp"] = kp
            r["tested_kd"] = kd

            results.append(r)

            print(
                f"hip   "
                f"Kp={kp:5.1f} | "
                f"Kd={kd:3.1f} | "
                f"len={r['length']:4d} | "
                f"mean_pitch={r['mean_pitch']:.3f} | "
                f"mean_rate={r['mean_rate']:.3f} | "
                f"min_h={r['min_height']:.3f} | "
                f"reason={r['reason']}"
            )

    best = rank_results(
        results,
        "BEST HIP GAINS"
    )

    best_hip_kp = (
        best["tested_kp"]
    )

    best_hip_kd = (
        best["tested_kd"]
    )

    # ========================================================
    # PHASE 3 — KNEE
    # ========================================================

    print()
    print(
        "=== PHASE 3: KNEE PD SWEEP ==="
    )

    knee_kp_values = [
        100,
        125,
        150,
        175,
        200,
    ]

    knee_kd_values = [
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]

    results = []

    for kp in knee_kp_values:

        for kd in knee_kd_values:

            r = run_pd(
                hip_kp=best_hip_kp,
                hip_kd=best_hip_kd,
                knee_kp=kp,
                knee_kd=kd,
                ankle_kp=best_ankle_kp,
                ankle_kd=best_ankle_kd,
            )

            r["tested_kp"] = kp
            r["tested_kd"] = kd

            results.append(r)

            print(
                f"knee  "
                f"Kp={kp:5.1f} | "
                f"Kd={kd:3.1f} | "
                f"len={r['length']:4d} | "
                f"mean_pitch={r['mean_pitch']:.3f} | "
                f"mean_rate={r['mean_rate']:.3f} | "
                f"min_h={r['min_height']:.3f} | "
                f"reason={r['reason']}"
            )

    best = rank_results(
        results,
        "BEST KNEE GAINS"
    )

    best_knee_kp = (
        best["tested_kp"]
    )

    best_knee_kd = (
        best["tested_kd"]
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=== FINAL COARSE PD CONFIG ==="
    )

    print(
        f"Hip:   "
        f"Kp={best_hip_kp:.1f}, "
        f"Kd={best_hip_kd:.1f}"
    )

    print(
        f"Knee:  "
        f"Kp={best_knee_kp:.1f}, "
        f"Kd={best_knee_kd:.1f}"
    )

    print(
        f"Ankle: "
        f"Kp={best_ankle_kp:.1f}, "
        f"Kd={best_ankle_kd:.1f}"
    )

    final_result = run_pd(
        hip_kp=best_hip_kp,
        hip_kd=best_hip_kd,
        knee_kp=best_knee_kp,
        knee_kd=best_knee_kd,
        ankle_kp=best_ankle_kp,
        ankle_kd=best_ankle_kd,
    )

    print()

    print(
        f"Final length: "
        f"{final_result['length']}"
    )

    print(
        f"Final return: "
        f"{final_result['return']:.2f}"
    )

    print(
        f"Mean pitch: "
        f"{final_result['mean_pitch']:.3f}"
    )

    print(
        f"Mean rate: "
        f"{final_result['mean_rate']:.3f}"
    )

    print(
        f"Min height: "
        f"{final_result['min_height']:.3f}"
    )

    print(
        f"Termination: "
        f"{final_result['reason']}"
    )


if __name__ == "__main__":
    main()