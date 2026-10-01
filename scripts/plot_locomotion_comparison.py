"""Generate the fixed-command comparison SVG using only the standard library."""

import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "locomotion_reproduction"


def main():
    with (RESULTS / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row["metric"]: row for row in csv.DictReader(stream)}
    panels = [
        ("mean_speed_x_mps", "Forward speed", 1.1),
        ("speed_tracking_rmse_mps", "Speed tracking RMSE", 0.25),
        ("yaw_rate_tracking_rmse_rps", "Yaw-rate tracking RMSE", 0.14),
        ("mean_tilt_deg", "Mean tilt", 5.0),
    ]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="670" viewBox="0 0 1060 670">',
        '<rect width="1060" height="670" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#182234">',
        '<text x="45" y="38" font-size="24" font-weight="bold">Fixed-command Isaac locomotion evaluation</text>',
        '<text x="45" y="64" font-size="14">16 environments; 20 s; flat ground; friction 0.8; seed 42; command [1, 0, 0]</text>',
        '<rect x="45" y="83" width="14" height="14" fill="#168578"/>',
        '<text x="67" y="95" font-size="14">Locally trained</text>',
        '<rect x="235" y="83" width="14" height="14" fill="#4263b8"/>',
        '<text x="257" y="95" font-size="14">Supplied reference</text>',
    ]
    for index, (key, title, maximum) in enumerate(panels):
        x = 75 + (index % 2) * 505
        y = 145 + (index // 2) * 235
        row = rows[key]
        svg.append(f'<text x="{x}" y="{y}" font-size="17" font-weight="bold">{escape(title)} ({escape(row["unit"])})</text>')
        bottom = y + 160
        for tick in range(5):
            value = maximum * tick / 4
            ty = bottom - 125 * tick / 4
            svg.append(f'<path d="M{x},{ty}h365" stroke="#e2e7ef"/>')
            svg.append(f'<text x="{x-8}" y="{ty+4}" text-anchor="end" font-size="11">{value:.2f}</text>')
        for offset, name, color, label in [
            (65, "local", "#168578", "Local"),
            (215, "reference", "#4263b8", "Reference"),
        ]:
            value = float(row[name])
            height = value / maximum * 125
            svg.append(f'<rect x="{x+offset}" y="{bottom-height}" width="80" height="{height}" fill="{color}"/>')
            svg.append(f'<text x="{x+offset+40}" y="{bottom-height-7}" text-anchor="middle" font-size="13">{value:.3f}</text>')
            svg.append(f'<text x="{x+offset+40}" y="{bottom+20}" text-anchor="middle" font-size="12">{label}</text>')
        if key == "mean_speed_x_mps":
            ty = bottom - 125 / maximum
            svg.append(f'<path d="M{x},{ty}h365" stroke="#67758a" stroke-dasharray="5 4"/>')
            svg.append(f'<text x="{x+365}" y="{ty-5}" text-anchor="end" font-size="11">Target: 1 m/s</text>')
    svg.extend([
        '<text x="45" y="610" font-size="14">Both runs reported zero fall/reset events. This is one condition, not a robustness benchmark.</text>',
        '<text x="45" y="636" font-size="13">Lower tracking RMSE is better; tilt alone does not establish overall controller quality.</text>',
        '</g></svg>',
    ])
    output = RESULTS / "figures" / "baseline_comparison.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
