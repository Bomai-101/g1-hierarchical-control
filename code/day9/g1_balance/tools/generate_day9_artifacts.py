"""Generate curated Day 9 result data and dependency-free SVG figures."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from html import escape
import json
from pathlib import Path


WIDTH = 1100
HEIGHT = 650
LEFT = 90
RIGHT = 35
TOP = 85
BOTTOM = 105
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = HEIGHT - TOP - BOTTOM

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "gray": "#64748b",
    "grid": "#dbe3ee",
    "ink": "#172033",
    "muted": "#526071",
    "paper": "#ffffff",
}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def svg_start(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["paper"]}"/>',
        (
            "<style>text{font-family:Inter,Segoe UI,Arial,sans-serif}"
            ".title{font-size:25px;font-weight:700;fill:#172033}"
            ".subtitle{font-size:14px;fill:#526071}"
            ".axis{font-size:12px;fill:#526071}"
            ".legend{font-size:13px;fill:#172033}</style>"
        ),
        f'<text class="title" x="{LEFT}" y="36">{escape(title)}</text>',
        f'<text class="subtitle" x="{LEFT}" y="61">{escape(subtitle)}</text>',
    ]


def svg_end(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def axis_map(values: list[float], low: float | None = None, high: float | None = None):
    minimum = min(values) if low is None else low
    maximum = max(values) if high is None else high
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    padding = 0.08 * (maximum - minimum)
    if low is None:
        minimum -= padding
    if high is None:
        maximum += padding
    return minimum, maximum


def draw_axes(
    parts: list[str],
    x_count: int,
    y_min: float,
    y_max: float,
    y_label: str,
    x_label: str = "PPO update",
) -> None:
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = TOP + PLOT_HEIGHT - PLOT_HEIGHT * index / 5
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_WIDTH}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="axis" x="{LEFT - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>'
        )
    parts.append(
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + PLOT_HEIGHT}" stroke="{COLORS["ink"]}"/>'
    )
    parts.append(
        f'<line x1="{LEFT}" y1="{TOP + PLOT_HEIGHT}" x2="{LEFT + PLOT_WIDTH}" y2="{TOP + PLOT_HEIGHT}" stroke="{COLORS["ink"]}"/>'
    )
    parts.append(
        f'<text class="axis" transform="translate(25 {TOP + PLOT_HEIGHT / 2}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'
    )
    if x_count > 1:
        for update in range(1, x_count + 1):
            if update == 1 or update == x_count or update % 5 == 0:
                x = LEFT + PLOT_WIDTH * (update - 1) / (x_count - 1)
                parts.append(
                    f'<text class="axis" x="{x:.1f}" y="{TOP + PLOT_HEIGHT + 24}" text-anchor="middle">{update}</text>'
                )
    if x_label:
        parts.append(
            f'<text class="axis" x="{LEFT + PLOT_WIDTH / 2}" y="{HEIGHT - 25}" text-anchor="middle">{escape(x_label)}</text>'
        )


def polyline_points(values: list[float], y_min: float, y_max: float) -> str:
    points = []
    for index, value in enumerate(values):
        x = LEFT + PLOT_WIDTH * index / max(1, len(values) - 1)
        y = TOP + PLOT_HEIGHT * (y_max - value) / (y_max - y_min)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def nominal_figure(rows: list[dict], output: Path) -> None:
    values = [row["nominal"]["ppo"]["length"] for row in rows]
    baseline = rows[0]["nominal"]["zero"]["length"]
    y_min, y_max = axis_map(values + [baseline])
    parts = svg_start(
        "Nominal deterministic performance by PPO update",
        "Randomized-reset run; dashed line is the 215-step zero-policy baseline.",
    )
    draw_axes(parts, len(rows), y_min, y_max, "Episode length (policy steps)")
    baseline_y = TOP + PLOT_HEIGHT * (y_max - baseline) / (y_max - y_min)
    parts.append(
        f'<line x1="{LEFT}" y1="{baseline_y:.1f}" x2="{LEFT + PLOT_WIDTH}" y2="{baseline_y:.1f}" '
        f'stroke="{COLORS["gray"]}" stroke-width="2" stroke-dasharray="8 7"/>'
    )
    parts.append(
        f'<polyline points="{polyline_points(values, y_min, y_max)}" fill="none" '
        f'stroke="{COLORS["blue"]}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for index, value in enumerate(values):
        x = LEFT + PLOT_WIDTH * index / max(1, len(values) - 1)
        y = TOP + PLOT_HEIGHT * (y_max - value) / (y_max - y_min)
        radius = 6 if rows[index]["update"] == 4 else 3
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{COLORS["blue"]}"/>')
    parts.append(f'<text class="legend" x="{LEFT + 15}" y="{TOP + 22}" fill="{COLORS["blue"]}">PPO nominal length</text>')
    parts.append(f'<text class="legend" x="{LEFT + 190}" y="{TOP + 22}" fill="{COLORS["gray"]}">Zero baseline = {baseline}</text>')
    svg_end(parts, output)


def robustness_figure(rows: list[dict], output: Path) -> None:
    mean_values = [row["nonzero_summary"]["mean_delta_steps"] for row in rows]
    worst_values = [row["nonzero_summary"]["worst_delta_steps"] for row in rows]
    y_min, y_max = axis_map(mean_values + worst_values + [0.0])
    parts = svg_start(
        "Robustness mean and worst-case delta by update",
        "A positive mean can coexist with a strongly negative worst case.",
    )
    draw_axes(parts, len(rows), y_min, y_max, "Delta versus zero policy (steps)")
    zero_y = TOP + PLOT_HEIGHT * y_max / (y_max - y_min)
    parts.append(f'<line x1="{LEFT}" y1="{zero_y:.1f}" x2="{LEFT + PLOT_WIDTH}" y2="{zero_y:.1f}" stroke="{COLORS["ink"]}" stroke-width="1.5"/>')
    parts.append(f'<polyline points="{polyline_points(mean_values, y_min, y_max)}" fill="none" stroke="{COLORS["green"]}" stroke-width="3"/>')
    parts.append(f'<polyline points="{polyline_points(worst_values, y_min, y_max)}" fill="none" stroke="{COLORS["red"]}" stroke-width="3"/>')
    legend_y = TOP + 19
    parts.append(
        f'<line x1="{LEFT + 15}" y1="{legend_y - 5}" x2="{LEFT + 38}" y2="{legend_y - 5}" '
        f'stroke="{COLORS["green"]}" stroke-width="4"/>'
    )
    parts.append(
        f'<text class="legend" x="{LEFT + 45}" y="{legend_y}">Nonzero mean delta</text>'
    )
    parts.append(
        f'<line x1="{LEFT + 200}" y1="{legend_y - 5}" x2="{LEFT + 223}" y2="{legend_y - 5}" '
        f'stroke="{COLORS["red"]}" stroke-width="4"/>'
    )
    parts.append(
        f'<text class="legend" x="{LEFT + 230}" y="{legend_y}">Worst-case delta</text>'
    )
    svg_end(parts, output)


def outcome_figure(rows: list[dict], output: Path) -> None:
    parts = svg_start(
        "Perturbation outcomes by update",
        "Each stacked bar contains the 12 fixed nonzero near-zero perturbation cases.",
    )
    draw_axes(parts, len(rows), 0.0, 12.0, "Number of cases")
    bar_width = PLOT_WIDTH / len(rows) * 0.72
    for index, row in enumerate(rows):
        x_center = LEFT + PLOT_WIDTH * index / max(1, len(rows) - 1)
        y_bottom = TOP + PLOT_HEIGHT
        for key, color in (("wins", COLORS["green"]), ("ties", COLORS["gray"]), ("losses", COLORS["red"])):
            value = row["nonzero_summary"][key]
            height = PLOT_HEIGHT * value / 12.0
            y_bottom -= height
            parts.append(f'<rect x="{x_center - bar_width / 2:.1f}" y="{y_bottom:.1f}" width="{bar_width:.1f}" height="{height:.1f}" fill="{color}"/>')
    legend_y = TOP + PLOT_HEIGHT + 42
    legend_x = LEFT + PLOT_WIDTH - 255
    for offset, (label, color) in enumerate((("Wins", COLORS["green"]), ("Ties", COLORS["gray"]), ("Losses", COLORS["red"]))):
        x = legend_x + offset * 85
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{x + 20}" y="{legend_y}">{label}</text>')
    svg_end(parts, output)


def selected_case_figure(rows: list[dict], output: Path) -> None:
    selected_updates = (4, 19, 24)
    selected = [row for row in rows if row["update"] in selected_updates]
    labels = []
    for case in selected[0]["cases"][1:]:
        if case["kind"] == "pitch":
            labels.append(f"p{case['pitch_delta']:+.3f}")
        else:
            labels.append(f"r{case['pitch_rate_delta']:+.2f}")
    values = [case["delta_steps"] for row in selected for case in row["cases"][1:]]
    y_min, y_max = axis_map(values + [0.0])
    parts = svg_start(
        "Selected checkpoints reveal directional specialization",
        "Update 24 gains strongly for positive pitch-rate but degrades for the opposite direction.",
    )
    draw_axes(
        parts,
        0,
        y_min,
        y_max,
        "Delta versus zero policy (steps)",
        x_label="Initial perturbation",
    )
    zero_y = TOP + PLOT_HEIGHT * y_max / (y_max - y_min)
    parts.append(f'<line x1="{LEFT}" y1="{zero_y:.1f}" x2="{LEFT + PLOT_WIDTH}" y2="{zero_y:.1f}" stroke="{COLORS["ink"]}" stroke-width="1.5"/>')
    group_width = PLOT_WIDTH / len(labels)
    bar_width = group_width * 0.22
    palette = (COLORS["blue"], COLORS["green"], COLORS["orange"])
    for case_index, label in enumerate(labels):
        x_center = LEFT + group_width * (case_index + 0.5)
        for series_index, row in enumerate(selected):
            value = row["cases"][case_index + 1]["delta_steps"]
            y_value = TOP + PLOT_HEIGHT * (y_max - value) / (y_max - y_min)
            y = min(y_value, zero_y)
            height = abs(zero_y - y_value)
            x = x_center + (series_index - 1) * bar_width
            parts.append(f'<rect x="{x - bar_width * 0.42:.1f}" y="{y:.1f}" width="{bar_width * 0.84:.1f}" height="{max(height, 1):.1f}" fill="{palette[series_index]}"/>')
        parts.append(f'<text class="axis" x="{x_center:.1f}" y="{TOP + PLOT_HEIGHT + 24}" text-anchor="middle">{escape(label)}</text>')
    for index, row in enumerate(selected):
        x = LEFT + 15 + index * 125
        parts.append(f'<rect x="{x}" y="{TOP + 8}" width="14" height="14" fill="{palette[index]}"/>')
        parts.append(f'<text class="legend" x="{x + 20}" y="{TOP + 20}">update {row["update"]}</text>')
    svg_end(parts, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-sweep-json", type=Path, required=True)
    parser.add_argument("--original-robustness-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sweep_path = args.checkpoint_sweep_json.resolve(strict=True)
    original_path = args.original_robustness_json.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    original = json.loads(original_path.read_text(encoding="utf-8"))
    rows = sorted(sweep["results"], key=lambda row: row["update"])

    csv_path = output_dir / "checkpoint_robustness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "update", "checkpoint", "nominal_zero_length", "nominal_ppo_length",
            "nominal_delta_steps", "nonzero_mean_delta_steps", "worst_delta_steps",
            "best_delta_steps", "wins", "ties", "losses", "mean_action_rms",
        ])
        for row in rows:
            robust = row["nonzero_summary"]
            nominal = row["nominal"]
            writer.writerow([
                row["update"], row["checkpoint"], nominal["zero"]["length"],
                nominal["ppo"]["length"], nominal["delta_steps"],
                robust["mean_delta_steps"], robust["worst_delta_steps"],
                robust["best_delta_steps"], robust["wins"], robust["ties"],
                robust["losses"], robust["mean_action_rms"],
            ])

    selected = {row["update"]: row for row in rows if row["update"] in (4, 19, 24)}
    summary = {
        "milestone": "Day 9 low-level residual PPO research baseline — frozen",
        "zero_policy_nominal_steps": original["nominal"]["zero"]["length"],
        "original_ppo_nominal_steps": original["nominal"]["ppo"]["length"],
        "original_ppo_nominal_delta_steps": (
            original["nominal"]["ppo"]["length"]
            - original["nominal"]["zero"]["length"]
        ),
        "randomized_run_best_nominal": {
            "update": 4,
            "steps": selected[4]["nominal"]["ppo"]["length"],
            "delta_steps": selected[4]["nominal"]["delta_steps"],
        },
        "randomized_run_best_worst_case": {
            "update": 19,
            "nominal_steps": selected[19]["nominal"]["ppo"]["length"],
            **selected[19]["nonzero_summary"],
        },
        "directional_specialization_example": {
            "update": 24,
            "nominal_steps": selected[24]["nominal"]["ppo"]["length"],
            **selected[24]["nonzero_summary"],
        },
        "evaluation_cases": sweep["protocol"],
        "conclusion": (
            "Residual PPO improved nominal standing and later learned "
            "direction-specific corrections, but no checkpoint demonstrated "
            "consistent symmetric robustness across all nonzero cases."
        ),
        "source_sha256": {
            "checkpoint_sweep_summary.json": file_sha256(sweep_path),
            "original_robustness_evaluation.json": file_sha256(original_path),
        },
    }
    (output_dir / "day9_results_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    nominal_figure(rows, figures_dir / "nominal_length_by_update.svg")
    robustness_figure(rows, figures_dir / "robustness_mean_and_worst_delta.svg")
    outcome_figure(rows, figures_dir / "wins_ties_losses_by_update.svg")
    selected_case_figure(rows, figures_dir / "selected_checkpoint_case_deltas.svg")

    print(f"Saved curated Day 9 artifacts to: {output_dir}")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
