"""Record a continuous reference-evaluator rollout without changing control."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=20.0)
    parser.add_argument('--label', default='local')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('Duration must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(args.reference_root / 'mujoco'))
    import mujoco
    import numpy as np
    from mujoco_eval import run_grid
    from mujoco_eval.play_viewer import configure_viewer_floor

    metadata = args.reference_root / 'mujoco/policies/isaac_metadata.json'
    cfg = SimpleNamespace(
        model=str(args.reference_root / 'mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml'),
        policy=str(args.policy.resolve()), metadata=str(metadata),
        command_x=1.0, command_y=0.0, command_yaw=0.0,
        timestep=0.001, duration=args.duration, seed=42, device='cpu',
        allow_missing_joints=False, initial_base_height=0.74,
        min_base_height=0.35, base_body='pelvis', control_mode='pd',
    )
    rows = []
    renderer = None
    sample_data = None
    encoder = None
    frame_count = 0
    step_count = 0
    original_step = mujoco.mj_step

    def sample(model, data):
        nonlocal renderer, encoder, frame_count, sample_data
        if sample_data is None:
            sample_data = mujoco.MjData(model)
        sample_data.qpos[:] = data.qpos
        sample_data.qvel[:] = data.qvel
        mujoco.mj_forward(model, sample_data)
        quat, velocity, omega, height = run_grid.read_base_state(model, sample_data, 'pelvis')
        w, x, y, z = quat
        yaw = float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        rows.append([float(data.time), float(data.qpos[0]), float(data.qpos[1]),
                     yaw, height, *velocity.tolist(), *omega.tolist()])
        if not args.render:
            return
        if renderer is None:
            configure_viewer_floor(model, 'plane')
            model.vis.global_.offwidth = 960
            model.vis.global_.offheight = 540
            renderer = mujoco.Renderer(model, height=540, width=960)
            encoder = subprocess.Popen([
                'ffmpeg', '-loglevel', 'error', '-n', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '960x540', '-r', '50', '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(args.output_dir / 'rollout.mp4'),
            ], stdin=subprocess.PIPE)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.65]
        camera.distance = 3.0
        camera.azimuth = 140.0
        camera.elevation = -35.0
        renderer.update_scene(sample_data, camera=camera)
        encoder.stdin.write(renderer.render().tobytes())
        frame_count += 1

    def observed_step(model, data, *extra, **kwargs):
        nonlocal step_count
        if step_count == 0:
            sample(model, data)
        original_step(model, data, *extra, **kwargs)
        step_count += 1
        if step_count % 20 == 0:
            sample(model, data)

    try:
        with patch.object(mujoco, 'mj_step', observed_step):
            metrics = run_grid.run_once(cfg, 'plane', 0.8)
    finally:
        if renderer is not None:
            renderer.close()
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait(timeout=30) != 0:
                raise RuntimeError('Video encoding failed')

    with (args.output_dir / 'trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'world_x_m', 'world_y_m', 'yaw_rad',
                         'base_height_m', 'world_vx_mps', 'world_vy_mps',
                         'world_vz_mps', 'world_wx_rps', 'world_wy_rps', 'world_wz_rps'])
        writer.writerows(rows)
    metrics['model'] = Path(metrics['model']).name
    metrics['policy'] = args.label
    metrics['metadata'] = 'isaac_metadata.json'
    metrics['policy_sha256'] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    metrics['metadata_sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    metrics['reference_head'] = subprocess.check_output(
        ['git', '-C', str(args.reference_root), 'rev-parse', 'HEAD'], text=True).strip()
    metrics['recording_frames'] = frame_count
    metrics['termination_definition'] = 'Base height below 0.35 m, checked at policy steps; no automatic reset'
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
