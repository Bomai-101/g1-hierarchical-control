from .sweep_standing_pose import run_pose


def main():

    # --------------------------------------------------------
    # Finer search around the promising region.
    #
    # Coarse sweep suggested a ridge near:
    #
    # (-0.15, 0.20, -0.05)
    # (-0.20, 0.30, -0.10)
    # (-0.25, 0.40, -0.15)
    #
    # The previous optimum was on the lower knee boundary,
    # so extend the search toward straighter legs.
    # --------------------------------------------------------

    hip_values = [
        -0.25,
        -0.225,
        -0.20,
        -0.175,
        -0.15,
        -0.125,
        -0.10,
        -0.075,
        -0.05,
    ]

    knee_values = [
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.225,
        0.25,
        0.275,
        0.30,
    ]

    results = []

    print(
        "=== FINE STANDING POSE SWEEP ==="
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
                f"hip={result['hip']:+.3f} | "
                f"knee={result['knee']:+.3f} | "
                f"ankle={result['ankle']:+.3f} | "
                f"len={result['length']:4d} | "
                f"return={result['return']:8.2f} | "
                f"mean_pitch={result['mean_pitch']:.3f} | "
                f"mean_rate={result['mean_rate']:.3f} | "
                f"min_h={result['min_height']:.3f}"
            )

    # --------------------------------------------------------
    # Ranking
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
        "=== TOP 15 FINE POSES ==="
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
