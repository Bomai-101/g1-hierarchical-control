import numpy as np

from .sweep_standing_pose import run_pose


def main():

    # ========================================================
    # Micro search around the best region found so far.
    #
    # Current best:
    #
    # hip   = -0.075
    # knee  = +0.050
    # ankle = +0.025
    #
    # The optimum is still on the lower knee boundary.
    # Extend knee search to 0 while refining hip resolution.
    # ========================================================

    hip_values = np.arange(
        -0.10,
        -0.049,
        0.005
    )

    knee_values = np.arange(
        0.00,
        0.101,
        0.01
    )

    results = []

    print(
        "=== MICRO STANDING POSE SWEEP ==="
    )

    print(
        "Zero policy. Fixed joint PD."
    )

    print()

    for hip in hip_values:

        for knee in knee_values:

            result = run_pose(
                hip_pitch=float(hip),
                knee=float(knee),
            )

            results.append(
                result
            )

            print(
                f"hip={result['hip']:+.3f} | "
                f"knee={result['knee']:+.3f} | "
                f"ankle={result['ankle']:+.3f} | "
                f"len={result['length']:4d} | "
                f"return={result['return']:8.2f} | "
                f"mean_pitch={result['mean_pitch']:.3f} | "
                f"mean_rate={result['mean_rate']:.3f} | "
                f"min_h={result['min_height']:.3f}"
            )

    # ========================================================
    # Ranking
    # ========================================================

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
        "=== TOP 15 MICRO POSES ==="
    )

    print()

    for rank, result in enumerate(
        results[:15],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"hip={result['hip']:+.3f} | "
            f"knee={result['knee']:+.3f} | "
            f"ankle={result['ankle']:+.3f} | "
            f"len={result['length']:4d} | "
            f"return={result['return']:8.2f} | "
            f"mean_pitch={result['mean_pitch']:.3f} | "
            f"mean_rate={result['mean_rate']:.3f} | "
            f"min_h={result['min_height']:.3f}"
        )


if __name__ == "__main__":
    main()
