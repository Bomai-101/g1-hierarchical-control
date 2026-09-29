import numpy as np

from g1_env import G1Env


# ============================================================
# Controller
# ============================================================

def get_action(
    obs,
    hip_kp=0.0,
    hip_kd=0.0,
    knee_kp=0.0,
    knee_kd=0.0,
    ankle_kp=0.0,
    ankle_kd=0.0,
):

    pitch = float(obs[59])
    pitch_rate = float(obs[62])

    # --------------------------------------------------------
    # Short-horizon system identification showed:
    #
    # positive hip action  -> negative pitch effect
    # positive knee action -> negative pitch effect
    # positive ankle action strongly reduces positive
    # pitch rate
    #
    # Therefore positive gains here form physical
    # negative feedback.
    # --------------------------------------------------------

    hip_action = (
        hip_kp * pitch
        + hip_kd * pitch_rate
    )

    knee_action = (
        knee_kp * pitch
        + knee_kd * pitch_rate
    )

    ankle_action = (
        ankle_kp * pitch
        + ankle_kd * pitch_rate
    )

    hip_action = np.clip(
        hip_action,
        -1.0,
        1.0
    )

    knee_action = np.clip(
        knee_action,
        -1.0,
        1.0
    )

    ankle_action = np.clip(
        ankle_action,
        -1.0,
        1.0
    )

    action = np.zeros(
        6,
        dtype=np.float32
    )

    # 6D action mapping:
    #
    # 0 left hip pitch
    # 1 left knee
    # 2 left ankle pitch
    # 3 right hip pitch
    # 4 right knee
    # 5 right ankle pitch

    action[0] = hip_action
    action[3] = hip_action

    action[1] = knee_action
    action[4] = knee_action

    action[2] = ankle_action
    action[5] = ankle_action

    return action


# ============================================================
# Episode
# ============================================================

def run_episode(
    hip_kp=0.0,
    hip_kd=0.0,
    knee_kp=0.0,
    knee_kd=0.0,
    ankle_kp=0.0,
    ankle_kd=0.0,
):

    env = G1Env()

    # IMPORTANT:
    #
    # Do NOT overwrite env.action_scale.
    #
    # We now want to test exactly the same control interface
    # that PPO will later use.
    #
    # Current Day 9 action_scale should remain 0.15.

    obs = env.reset()

    total_reward = 0.0

    mean_pitch = 0.0
    mean_rate = 0.0

    max_pitch = 0.0
    max_rate = 0.0

    min_height = float(
        env.data.qpos[2]
    )

    for step in range(
        env.max_episode_steps
    ):

        action = get_action(
            obs=obs,

            hip_kp=hip_kp,
            hip_kd=hip_kd,

            knee_kp=knee_kp,
            knee_kd=knee_kd,

            ankle_kp=ankle_kp,
            ankle_kd=ankle_kd,
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

        mean_pitch += abs(
            pitch
        )

        mean_rate += abs(
            pitch_rate
        )

        max_pitch = max(
            max_pitch,
            abs(pitch)
        )

        max_rate = max(
            max_rate,
            abs(pitch_rate)
        )

        min_height = min(
            min_height,
            height
        )

        if terminated or truncated:
            break

    length = step + 1

    return {
        "length":
            length,

        "return":
            total_reward,

        "mean_pitch":
            mean_pitch / length,

        "mean_rate":
            mean_rate / length,

        "max_pitch":
            max_pitch,

        "max_rate":
            max_rate,

        "min_height":
            min_height,
    }


# ============================================================
# Ranking
# ============================================================

def rank_results(
    results,
    title,
):

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
        f"=== {title} ==="
    )
    print()

    for i, r in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"len={r['length']:4d} | "
            f"return={r['return']:8.2f} | "
            f"mean_pitch={r['mean_pitch']:.3f} | "
            f"mean_rate={r['mean_rate']:.3f} | "
            f"min_h={r['min_height']:.3f} | "
            f"Kp={r['kp']:.3f} | "
            f"Kd={r['kd']:.3f}"
        )

    return results[0]


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Zero baseline
    # ========================================================

    baseline = run_episode()

    print(
        "=== SCRIPTED BALANCE V2 ==="
    )

    print()

    print(
        "ZERO BASELINE | "
        f"len={baseline['length']} | "
        f"return={baseline['return']:.2f} | "
        f"mean_pitch={baseline['mean_pitch']:.3f} | "
        f"mean_rate={baseline['mean_rate']:.3f} | "
        f"min_h={baseline['min_height']:.3f}"
    )

    # ========================================================
    # PHASE 1
    #
    # Ankle damping only
    #
    # Diagnostic showed ankle has very little short-term
    # pitch authority but strong pitch-rate authority.
    # ========================================================

    print()
    print(
        "=== PHASE 1: ANKLE DAMPING ==="
    )

    ankle_kd_values = [
        0.00,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]

    results = []

    for kd in ankle_kd_values:

        r = run_episode(
            ankle_kp=0.0,
            ankle_kd=kd,
        )

        r["kp"] = 0.0
        r["kd"] = kd

        results.append(
            r
        )

        print(
            f"ankle Kd={kd:4.2f} | "
            f"len={r['length']:4d} | "
            f"return={r['return']:8.2f} | "
            f"mean_pitch={r['mean_pitch']:.3f} | "
            f"mean_rate={r['mean_rate']:.3f} | "
            f"min_h={r['min_height']:.3f}"
        )

    best_ankle = rank_results(
        results,
        "BEST ANKLE DAMPING"
    )

    best_ankle_kd = (
        best_ankle["kd"]
    )

    # ========================================================
    # PHASE 2
    #
    # Hip PD
    #
    # Keep best ankle damping fixed.
    # ========================================================

    print()
    print(
        "=== PHASE 2: HIP FEEDBACK ==="
    )

    hip_kp_values = [
        0.00,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
    ]

    hip_kd_values = [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]

    results = []

    for kp in hip_kp_values:

        for kd in hip_kd_values:

            r = run_episode(
                hip_kp=kp,
                hip_kd=kd,

                ankle_kp=0.0,
                ankle_kd=best_ankle_kd,
            )

            r["kp"] = kp
            r["kd"] = kd

            results.append(
                r
            )

            print(
                f"hip "
                f"Kp={kp:4.2f} | "
                f"Kd={kd:4.2f} | "
                f"len={r['length']:4d} | "
                f"mean_pitch={r['mean_pitch']:.3f} | "
                f"mean_rate={r['mean_rate']:.3f} | "
                f"min_h={r['min_height']:.3f}"
            )

    best_hip = rank_results(
        results,
        "BEST HIP FEEDBACK"
    )

    best_hip_kp = (
        best_hip["kp"]
    )

    best_hip_kd = (
        best_hip["kd"]
    )

    # ========================================================
    # PHASE 3
    #
    # Add knee feedback.
    #
    # Knee showed the same physical direction as hip,
    # but weaker authority.
    # ========================================================

    print()
    print(
        "=== PHASE 3: KNEE FEEDBACK ==="
    )

    knee_kp_values = [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
    ]

    knee_kd_values = [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
    ]

    results = []

    for kp in knee_kp_values:

        for kd in knee_kd_values:

            r = run_episode(
                hip_kp=best_hip_kp,
                hip_kd=best_hip_kd,

                knee_kp=kp,
                knee_kd=kd,

                ankle_kp=0.0,
                ankle_kd=best_ankle_kd,
            )

            r["kp"] = kp
            r["kd"] = kd

            results.append(
                r
            )

            print(
                f"knee "
                f"Kp={kp:4.2f} | "
                f"Kd={kd:4.2f} | "
                f"len={r['length']:4d} | "
                f"mean_pitch={r['mean_pitch']:.3f} | "
                f"mean_rate={r['mean_rate']:.3f} | "
                f"min_h={r['min_height']:.3f}"
            )

    best_knee = rank_results(
        results,
        "BEST KNEE FEEDBACK"
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=== FINAL SCRIPTED V2 ==="
    )

    print(
        f"Hip: "
        f"Kp={best_hip_kp:.3f}, "
        f"Kd={best_hip_kd:.3f}"
    )

    print(
        f"Knee: "
        f"Kp={best_knee['kp']:.3f}, "
        f"Kd={best_knee['kd']:.3f}"
    )

    print(
        f"Ankle: "
        f"Kp=0.000, "
        f"Kd={best_ankle_kd:.3f}"
    )

    final = run_episode(
        hip_kp=best_hip_kp,
        hip_kd=best_hip_kd,

        knee_kp=best_knee["kp"],
        knee_kd=best_knee["kd"],

        ankle_kp=0.0,
        ankle_kd=best_ankle_kd,
    )

    print()

    print(
        f"ZERO:  "
        f"{baseline['length']} steps"
    )

    print(
        f"V2:    "
        f"{final['length']} steps"
    )

    print(
        f"Return: "
        f"{final['return']:.2f}"
    )

    print(
        f"Mean pitch: "
        f"{final['mean_pitch']:.3f}"
    )

    print(
        f"Mean rate: "
        f"{final['mean_rate']:.3f}"
    )

    print(
        f"Min height: "
        f"{final['min_height']:.3f}"
    )


if __name__ == "__main__":
    main()