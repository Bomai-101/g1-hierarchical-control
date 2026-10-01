"""Plot world-frame paths and unwrapped heading from continuous rollouts."""
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/locomotion_reproduction'


def load(name):
    with (RESULTS / 'sim2sim' / name / 'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    heading = []
    previous = None
    total = 0.0
    for row in rows:
        angle = float(row['yaw_rad'])
        if previous is None:
            total = angle
        else:
            total += math.atan2(math.sin(angle-previous), math.cos(angle-previous))
        heading.append(total * 180 / math.pi)
        previous = angle
    return rows, heading


def main():
    tracks = [(load('local'), '#168578', 'Locally trained'),
              (load('reference'), '#4263b8', 'Supplied reference')]
    positions = [(float(row['world_x_m']), float(row['world_y_m']))
                 for (rows, _), _, _ in tracks for row in rows]
    xmin, xmax = min(x for x, _ in positions), max(x for x, _ in positions)
    ymin, ymax = min(y for _, y in positions), max(y for _, y in positions)
    span = max(xmax-xmin, ymax-ymin, 1) * 1.15
    xmin = (xmin+xmax-span)/2
    ymin = (ymin+ymax-span)/2
    angles = [v for (_, heading), _, _ in tracks for v in heading]
    low, high = min(angles+[0])-20, max(angles+[0])+20
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="600">',
           '<rect width="1080" height="600" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#182234">',
           '<text x="40" y="36" font-size="23" font-weight="bold">MuJoCo Sim2Sim: continuous walking with heading drift</text>',
           '<text x="40" y="63" font-size="14">20 s without resets; flat ground; friction 0.8; command [1, 0, 0]; seed 42</text>']
    for offset, (_, color, label) in enumerate(tracks):
        x = 40 + 250*offset
        svg.extend([f'<path d="M{x},89h24" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{x+32}" y="94" font-size="14">{label}</text>'])
    svg.extend(['<text x="80" y="130" font-size="17">World XY path (equal axis scale)</text>',
                '<text x="620" y="130" font-size="17">Unwrapped heading (degrees)</text>'])
    for tick in range(5):
        position = tick/4
        svg.extend([
            f'<path d="M{80+360*position},160v360 M80,{520-360*position}h360" stroke="#e2e7ef"/>',
            f'<text x="{80+360*position}" y="540" text-anchor="middle" font-size="11">{xmin+span*position:.1f}</text>',
            f'<text x="72" y="{524-360*position}" text-anchor="end" font-size="11">{ymin+span*position:.1f}</text>',
            f'<path d="M{620+380*position},160v360 M620,{520-360*position}h380" stroke="#e2e7ef"/>',
            f'<text x="{620+380*position}" y="540" text-anchor="middle" font-size="11">{20*position:.0f}</text>',
            f'<text x="612" y="{524-360*position}" text-anchor="end" font-size="11">{low+(high-low)*position:.0f}</text>',
        ])
    for (rows, heading), color, _ in tracks:
        xy = ' '.join(f'{80+(float(r["world_x_m"])-xmin)/span*360:.2f},{520-(float(r["world_y_m"])-ymin)/span*360:.2f}' for r in rows)
        yaw = ' '.join(f'{620+float(r["time_s"])/20*380:.2f},{520-(a-low)/(high-low)*360:.2f}' for r, a in zip(rows, heading))
        svg.extend([f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="2"/>',
                    f'<polyline points="{yaw}" fill="none" stroke="{color}" stroke-width="2"/>'])
    svg.extend(['<text x="220" y="565" font-size="12">World X (m); vertical axis: world Y (m)</text>',
                '<text x="765" y="565" font-size="12">Time (s)</text>', '</g></svg>'])
    output = RESULTS / 'figures/sim2sim_heading_drift.svg'
    output.write_text('\n'.join(svg)+'\n')
    print(output)


if __name__ == '__main__':
    main()
