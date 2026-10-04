"""Plot complete flat continuation comparisons; failed traces end at failure."""
import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads((args.results_dir / "analysis.json").read_text())
    colors = {"baseline1499": "#2874A6", "candidate1999": "#D35400"}
    names = {"baseline1499": "Baseline 1499", "candidate1999": "Candidate 1999"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for col, backend in enumerate(("isaac", "mujoco")):
        groups = [g for g in analysis["groups"] if g["backend"] == backend]
        for index, group in enumerate(groups):
            ax = axes[0, col]
            ax.bar(index, group["full_30s_cases"], color=colors[group["actor"]], width=.6)
            ax.text(index, group["full_30s_cases"] + .25, f"{group['full_30s_cases']}/12", ha="center")
        axes[0, col].set(ylim=(0, 14), xticks=[0, 1], xticklabels=["Baseline 1499", "Candidate 1999"],
                         ylabel="Cases completing 30 s", title=backend.capitalize())
        axes[0, col].grid(axis="y", alpha=.2)
        for actor in colors:
            if backend == "isaac":
                arrays = np.load(args.run_root / "isaac_posttraining" / f"{actor}_09.npz")
                t, data = arrays["time"], arrays["trace"]
                valid = data[:, :, 0].astype(bool)
                error = np.abs(np.arctan2(np.sin(data[:, :, 20]), np.cos(data[:, :, 20])))
                counts = valid.sum(axis=1)
                keep = counts > 0
                t = t[keep]
                error = np.where(valid, error, 0).sum(axis=1)[keep] / counts[keep]
                case = json.loads((args.run_root / "isaac_posttraining" / f"{actor}_09.json").read_text())
                failed = case["falls"] > 0
            else:
                path = args.run_root / "mujoco_posttraining" / f"{actor}_09.csv"
                rows = list(csv.DictReader(path.open()))
                t = np.array([float(r["time_s"]) for r in rows])
                error = np.array([abs(float(r["heading_error"])) for r in rows])
                failed = json.loads((path.with_suffix(".json")).read_text())["summary"]["fall"] == 1
            axes[1, col].plot(t, error, color=colors[actor], label=names[actor], linewidth=1.5)
            if failed:
                axes[1, col].plot(t[-1], error[-1], "x", color=colors[actor], markersize=9)
        axes[1, col].set(xlim=(0, 30), xlabel="Time (s)", ylabel="Absolute heading error (rad)",
                         title="Heading 0, start +0.5 rad, forward 1 m/s")
        axes[1, col].grid(alpha=.2)
        axes[1, col].legend(fontsize=9)
    fig.suptitle("500 extra PPO updates: matched flat-ground evaluation", fontsize=14)
    fig.supxlabel("Fixed starts, single seed. Cross = terminated trace; missing time is not successful tracking.", fontsize=9)
    directory = args.results_dir / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "continuation_comparison.svg")
    fig.savefig(directory / "continuation_comparison.png", dpi=160)
    (args.results_dir / "plot_runtime.json").write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, matplotlib=matplotlib.__version__), indent=2))


if __name__ == "__main__":
    main()
