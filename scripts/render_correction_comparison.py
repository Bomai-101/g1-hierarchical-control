"""Render paired G1 correction trajectories to a labelled local MP4.

Uses saved qpos/qvel only, so playback cannot affect experiment outcomes.
Run in the project's venv; EGL offscreen rendering and ffmpeg are required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g1_control.config import balance as config
from g1_control.envs.balance_env import G1Env
from record_instability_trajectories import sha256


PHASE1 = config.CHECKPOINT_DIR / "instability_trajectories" / "phase1_20260930T002805Z"
FULL_RUN = config.CHECKPOINT_DIR / "correction_experiments" / "correction_20260930T015410Z"
CENTER_BEST_RUN = config.CHECKPOINT_DIR / "correction_experiments" / "correction_20260930T015345Z"
FRAME_WIDTH, FRAME_HEIGHT, FPS = 640, 480, 50


def arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        qpos = payload["qpos"].copy()
        qvel = payload["qvel"].copy()
    if len(qpos) != len(qvel):
        raise RuntimeError(f"Position/velocity frame mismatch: {path}")
    return qpos, qvel


def prepend_history(history: tuple[np.ndarray, np.ndarray],
                    branch: tuple[np.ndarray, np.ndarray], step: int) -> tuple[np.ndarray, np.ndarray]:
    hq, hv = history
    bq, bv = branch
    if (np.max(np.abs(hq[step] - bq[0])) > 1e-10
            or np.max(np.abs(hv[step] - bv[0])) > 1e-10):
        raise RuntimeError("Branch video does not match the Phase 1 historical prefix")
    return np.concatenate((hq[:step], bq)), np.concatenate((hv[:step], bv))


def render_frame(renderer: mujoco.Renderer, env: G1Env, camera: mujoco.MjvCamera,
                 qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera=camera)
    return renderer.render().copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", type=Path, default=PHASE1)
    parser.add_argument("--full-run", type=Path, default=FULL_RUN)
    parser.add_argument("--center-best-run", type=Path, default=CENTER_BEST_RUN)
    parser.add_argument("--comparison", choices=("historical", "pulse_baseline", "trigger"),
                        default="historical")
    parser.add_argument("--output", type=Path,
                        help="MP4 path; defaults to the full-run local checkpoint directory")
    args = parser.parse_args()
    phase1 = args.phase1.expanduser().resolve(strict=True)
    full_run = args.full_run.expanduser().resolve(strict=True)
    center_run = args.center_best_run.expanduser().resolve(strict=True)
    full = json.loads((full_run / "results.json").read_text(encoding="utf-8"))
    center = json.loads((center_run / "results.json").read_text(encoding="utf-8"))
    source = json.loads((phase1 / "summary.json").read_text(encoding="utf-8"))
    if (full["branch_step"] != center["branch_step"]
            or full["checkpoint_sha256"] != center["checkpoint_sha256"]
            or full["checkpoint_sha256"] != source["model_reference"]["checkpoint_sha256"]):
        raise RuntimeError("Video source provenance mismatch")
    scene = Path.home() / "robotics/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    if sha256(scene) != source["model_reference"]["scene_sha256"]:
        raise RuntimeError("Scene XML differs from recorded Phase 1 trajectory")

    history = arrays(phase1 / "best_ppo.npz")
    baseline = prepend_history(history, arrays(full_run / "baseline_center.npz"),
                               int(full["branch_step"]))
    center_best = prepend_history(history, arrays(center_run / "selected_center.npz"),
                                  int(center["branch_step"]))
    selected = prepend_history(history, arrays(full_run / "selected_center.npz"),
                               int(full["branch_step"]))
    if args.comparison == "historical":
        left, right = history, center_best
        left_label = "Historical PPO 241 steps"
        right_label = "Second pulse 518 steps"
    elif args.comparison == "pulse_baseline":
        left, right = baseline, center_best
        left_label = "Pulse PD 492 steps"
        right_label = "Second pulse 518 steps"
    else:
        left, right = baseline, selected
        left_label = "Pulse PD 492 steps"
        right_label = f"Trigger {len(selected[0])-1} steps"
    output = (args.output.expanduser().resolve() if args.output else
              full_run / f"{args.comparison}_comparison.mp4")
    if output.parent != full_run and args.output is None:
        raise RuntimeError("Unexpected default output location")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = G1Env()
    env.reset()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.78)
    camera.distance = 3.1
    camera.azimuth = 135
    camera.elevation = -12
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        raise FileNotFoundError(font)
    vf = (
        f"drawtext=fontfile={font}:text='{left_label}':x=18:y=18:"
        "fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8,"
        f"drawtext=fontfile={font}:text='{right_label}':x={FRAME_WIDTH+18}:y=18:"
        "fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{2*FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-framerate", str(FPS), "-i", "pipe:0",
        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "23",
        "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    frame_count = max(len(left[0]), len(right[0]))
    with mujoco.Renderer(env.model, width=FRAME_WIDTH, height=FRAME_HEIGHT) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None and process.stderr is not None
        try:
            for frame in range(frame_count):
                li = min(frame, len(left[0]) - 1)
                ri = min(frame, len(right[0]) - 1)
                left_image = render_frame(renderer, env, camera, left[0][li], left[1][li])
                right_image = render_frame(renderer, env, camera, right[0][ri], right[1][ri])
                process.stdin.write(np.concatenate((left_image, right_image), axis=1).tobytes())
        finally:
            process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    print(f"Saved {frame_count} frames at {FPS} fps: {output}")


if __name__ == "__main__":
    main()
