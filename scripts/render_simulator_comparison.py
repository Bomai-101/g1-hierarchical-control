"""Render recorded Isaac rigid-body poses and MuJoCo states without stepping physics."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np


def render(source, engine, actor, output, duration=8, fps=25):
    trace = np.load(source / f"{actor}.npz", allow_pickle=False)
    isaac = engine == "isaac"
    model = (mujoco.MjModel.from_xml_path(str(source / "geometry/isaac_visual.xml")) if isaac
             else mujoco.MjModel.from_binary_path(str(source / f"{actor}.mjb")))
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -22
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    renderer = mujoco.Renderer(model, height=480, width=640)
    last = float(trace["time"][-1])
    heading = "ISAAC | actual body-pose replay" if isaac else "MUJOCO | actual simulation replay"
    filters = [f"drawbox=x=0:y=0:w=iw:h=65:color=black@0.75:t=fill",
               f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{heading}':x=14:y=12:fontsize=20:fontcolor=white",
               "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Command vx=1 m/s, vy=0, yaw rate=0':x=14:y=40:fontsize=16:fontcolor=white"]
    if last < duration - .01:
        filters += [f"drawbox=x=0:y=410:w=iw:h=70:color=black@0.85:t=fill:enable='gte(t,{last})'",
                    f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='FALL - terminated at {last:.2f}s; frame held':x=12:y=430:fontsize=21:fontcolor=red:enable='gte(t,{last})'"]
    process = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "640x480", "-r", str(fps), "-i", "-", "-vf", ",".join(filters),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    body_names = list(trace["body_names"]) if isaac else []
    addresses = [model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))]] for name in body_names]
    pelvis = body_names.index("pelvis") if isaac else 0
    try:
        for frame in range(round(duration * fps)):
            t = min(frame / fps, last)
            index = int(np.argmin(np.abs(trace["time"] - t)))
            if isaac:
                positions = trace["body_positions"][index] - trace["origin"]
                quats = trace["body_quaternions"][index]
                for body, address in enumerate(addresses):
                    data.qpos[address:address + 3] = positions[body]
                    data.qpos[address + 3:address + 7] = quats[body]
                root = positions[pelvis]
            else:
                data.qpos[:] = trace["states"][index, :model.nq]
                data.qvel[:] = trace["states"][index, model.nq:]
                root = data.qpos[:3]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [root[0], root[1], .65]
            renderer.update_scene(data, camera=camera, scene_option=options)
            pixels = renderer.render()
            process.stdin.write(pixels.tobytes())
    finally:
        process.stdin.close()
        renderer.close()
    if process.wait() != 0:
        raise RuntimeError("Video encoder failed")
    return dict(engine=engine, actor=actor, real_duration_s=last, video_duration_s=duration,
                fps=fps, frames=round(duration * fps), terminal_frame_held=last < duration - .01,
                trace_sha256=hashlib.sha256((source / f"{actor}.npz").read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--mujoco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for actor in ("baseline1499", "candidate1999"):
        for engine, source in (("isaac", args.isaac), ("mujoco", args.mujoco)):
            file = args.output / f"{actor}_{engine}.mp4"
            records.append(render(source, engine, actor, file))
            print("RENDERED", file, flush=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.output / f"{actor}_isaac.mp4"),
            "-i", str(args.output / f"{actor}_mujoco.mp4"), "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output / f"{actor}_comparison.mp4")], check=True)
    (args.output / "render_manifest.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
