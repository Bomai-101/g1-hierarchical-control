import numpy as np

from .sweep_standing_pose import run_pose


def main():

    # ========================================================
    # 1D sweep along the stability ridge discovered from
    # the previous grid searches:
    #
    # knee  ≈ -2 * hip - 0.09
    # ankle = -(hip + knee)
    #
    # Search narrowly around the observed optimum.
    # ========================================================

    hip_values = np.arange(
        #-0.115,
        #-0.0775,
        #0.0025
        -0.200,
        -0.1124,
        0.0025
    )

    results = []

    print(
        "=== 1D STANDING POSE RIDGE SWEEP ==="
    )

    print(
        "Zero policy. Fixed joint PD."
    )

    print()

    for hip in hip_values:

        knee = (
            -2.0 * hip
            - 0.09
        )

        result = run_pose(
            hip_pitch=float(hip),
            knee=float(knee),
        )

        results.append(
            result
        )

        print(
            f"hip={result['hip']:+.4f} | "
            f"knee={result['knee']:+.4f} | "
            f"ankle={result['ankle']:+.4f} | "
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
        "=== RIDGE SWEEP RANKING ==="
    )

    print()

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"hip={result['hip']:+.4f} | "
            f"knee={result['knee']:+.4f} | "
            f"ankle={result['ankle']:+.4f} | "
            f"len={result['length']:4d} | "
            f"return={result['return']:8.2f} | "
            f"mean_pitch={result['mean_pitch']:.3f} | "
            f"mean_rate={result['mean_rate']:.3f} | "
            f"min_h={result['min_height']:.3f}"
        )


if __name__ == "__main__":
    main()
