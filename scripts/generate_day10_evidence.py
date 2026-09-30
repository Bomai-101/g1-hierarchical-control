"""Build the public Day 10 five-state evidence figure from compact CSV data.

The input is a small, tracked summary of the deterministic local sweep. This
script does not load checkpoints, run physics, or alter controller parameters.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "day10" / "source_candidates.csv"
SUMMARY = ROOT / "results" / "day10" / "five_state_summary.csv"
FIGURE = ROOT / "results" / "day10" / "figures" / "early_intervention_five_states.svg"

BRANCH_STEP = 145
EXPECTED_CASES = 85
EXPECTED_CANDIDATES_PER_STATE = 17
BASELINE = "pulse_then_pd"
SECOND = "fixed_220_knee_0.25_5"
STATES = (
    ("Exact center", 0.0, 0.0),
    ("Pitch -0.0005 rad", -0.0005, 0.0),
    ("Pitch +0.0005 rad", 0.0005, 0.0),
    ("Angular-y -0.002 rad/s", 0.0, -0.002),
    ("Angular-y +0.002 rad/s", 0.0, 0.002),
)


def state_key(pitch: str | float, rate: str | float) -> tuple[float, float]:
    return round(float(pitch), 6), round(float(rate), 6)


def true_value(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Unexpected boolean value: {value!r}")
    return value == "True"


def load_evidence() -> list[dict[str, object]]:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} source cases, found {len(rows)}")
    by_key: dict[tuple[float, float, str], dict[str, str]] = {}
    names_by_state: dict[tuple[float, float], set[str]] = {}
    for row in rows:
        if int(row["branch_step"]) != BRANCH_STEP:
            raise ValueError("Mixed branch steps in public source data")
        state = state_key(row["pitch_offset_rad"], row["rate_offset_rad_s"])
        key = (*state, row["candidate"])
        if key in by_key:
            raise ValueError(f"Duplicate source case: {key}")
        by_key[key] = row
        names_by_state.setdefault(state, set()).add(row["candidate"])
    expected_states = {state_key(pitch, rate) for _, pitch, rate in STATES}
    if set(names_by_state) != expected_states:
        raise ValueError("Source states differ from the fixed five-state comparison")
    if any(len(names) != EXPECTED_CANDIDATES_PER_STATE for names in names_by_state.values()):
        raise ValueError("A state is missing candidate evaluations")

    evidence: list[dict[str, object]] = []
    for label, pitch, rate in STATES:
        state = state_key(pitch, rate)
        baseline = by_key[(*state, BASELINE)]
        second = by_key[(*state, SECOND)]
        baseline_steps = int(baseline["steps"])
        second_steps = int(second["steps"])
        fired_at = second["correction_step"]
        if state == (0.0, 0.0):
            if (baseline_steps, second_steps, fired_at) != (347, 373, "220"):
                raise ValueError("Exact-center reference does not reproduce")
        elif fired_at or baseline_steps != second_steps or second_steps >= 220:
            raise ValueError(f"Neighbor timing or outcome changed: {label}")
        if int(baseline["torque_clip_count"]) or int(second["torque_clip_count"]):
            raise ValueError(f"Unexpected torque clipping: {label}")
        evidence.append({
            "state": label,
            "pitch_offset_rad": pitch,
            "angular_y_offset_rad_s": rate,
            "baseline_steps_after_branch": baseline_steps,
            "second_candidate_steps_after_branch": second_steps,
            "second_pulse_executed": bool(fired_at),
            "baseline_passed_4s": true_value(baseline["passed_4s"]),
            "second_passed_4s": true_value(second["passed_4s"]),
            "baseline_held_10s": true_value(baseline["held_continuously_after_2s"]),
            "second_held_10s": true_value(second["held_continuously_after_2s"]),
        })
    if any(item["baseline_held_10s"] or item["second_held_10s"] for item in evidence):
        raise ValueError("The frozen result no longer supports the published 10 s conclusion")
    return evidence


def write_summary(evidence: list[dict[str, object]]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)


def text(x: float, y: float, value: str, *, size: int = 15,
         color: str = "#1e293b", weight: str = "normal", anchor: str = "start") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def render_svg(evidence: list[dict[str, object]]) -> str:
    width, height = 1120, 730
    plot_x, plot_width, top, row_gap = 285, 640, 200, 88
    scale = plot_width / 400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Five-state early intervention comparison">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(40, 48, "Early intervention: one exact-state gain, no neighboring-state recovery",
             size=24, weight="bold"),
        text(40, 79, "Deterministic G1 MuJoCo branches at PPO step 145; survival is measured after the branch.",
             size=15, color="#475569"),
        '<rect x="42" y="108" width="18" height="12" fill="#64748b"/>',
        text(67, 119, "First pulse, then PD", size=14),
        '<rect x="240" y="108" width="18" height="12" fill="#2563eb"/>',
        text(265, 119, "Scheduled second knee pulse", size=14),
        text(40, 164, "Branch state", size=14, color="#475569", weight="bold"),
        text(plot_x, 164, "Episode survival after branch (policy steps)",
             size=14, color="#475569", weight="bold"),
        text(949, 164, "4 s / 10 s (both)", size=14, color="#475569", weight="bold"),
    ]
    for tick in (0, 100, 200, 300, 400):
        x = plot_x + tick * scale
        parts.append(f'<line x1="{x:.1f}" y1="181" x2="{x:.1f}" y2="623" '
                     f'stroke="{("#94a3b8" if tick == 200 else "#e2e8f0")}" '
                     f'stroke-dasharray="{("5 5" if tick == 200 else "none")}"/>')
        parts.append(text(x, 649, str(tick), size=13, color="#475569", anchor="middle"))
    for index, item in enumerate(evidence):
        y = top + row_gap * index
        label = str(item["state"])
        a = int(item["baseline_steps_after_branch"])
        b = int(item["second_candidate_steps_after_branch"])
        parts.append(text(40, y + 25, label, size=15, weight="bold" if index == 0 else "normal"))
        parts.append(f'<rect x="{plot_x}" y="{y}" width="{a * scale:.1f}" height="17" fill="#64748b"/>')
        parts.append(f'<rect x="{plot_x}" y="{y + 25}" width="{b * scale:.1f}" height="17" fill="#2563eb"/>')
        parts.append(text(plot_x + a * scale + 7, y + 14, str(a), size=13))
        parts.append(text(plot_x + b * scale + 7, y + 39, str(b), size=13))
        status = "pass / fail" if item["baseline_passed_4s"] and item["second_passed_4s"] else "fail / fail"
        parts.append(text(949, y + 26, status, size=14))
        if not item["second_pulse_executed"]:
            parts.append(text(949, y + 47, "second pulse not reached", size=11, color="#64748b"))
    parts.extend([
        text(40, 681, "Dashed line: 200 steps (4 s) is a minimum duration, not the full entry-and-hold success test.",
             size=14, color="#475569"),
        text(40, 704, "No case held continuously for 10 s; all four neighbors fell before the scheduled second pulse at step 220.",
             size=14, color="#475569"),
        "</svg>",
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    evidence = load_evidence()
    write_summary(evidence)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.write_text(render_svg(evidence), encoding="utf-8")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print(f"Source SHA-256: {digest}")
    print(f"Saved: {SUMMARY.relative_to(ROOT)}")
    print(f"Saved: {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
