"""Render actual recorded positive-turn replays; no physics is advanced."""
import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for profile,title,model_path in (("original","ORIGINAL MUJOCO",args.run_root/"visual_mujoco/baseline1499.mjb"),
        ("usd_position","USD-MATCHED MUJOCO",args.run_root/"usd_physical_models/g1_usd_physical_position.mjb")):
        trace=np.load(args.run_root/"usd_physical_replays"/f"{profile}_vx1_yaw+0.2.npz")
        model=mujoco.MjModel.from_binary_path(str(model_path));model.vis.global_.offwidth=640;model.vis.global_.offheight=480
        model.vis.headlight.ambient[:]=.6;model.vis.headlight.diffuse[:]=.6
        data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=480,width=640)
        options=mujoco.MjvOption();options.geomgroup[3:]=0
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=3;camera.azimuth=135;camera.elevation=-22
        filter_text=f"drawbox=x=0:y=0:w=iw:h=65:color=black@0.8:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{title} | actor1499':x=12:y=12:fontsize=20:fontcolor=white,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='vx=1 m/s, vy=0, yaw rate=+0.2 rad/s':x=12:y=40:fontsize=16:fontcolor=white"
        path=args.output/f"{profile}_positive_turn.mp4"
        encoder=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r","25","-i","-",
            "-vf",filter_text,"-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(path)],stdin=subprocess.PIPE)
        try:
            for frame in range(200):
                index=int(np.argmin(np.abs(trace["time"]-frame/25)))
                state=trace["states"][index];data.qpos[:]=state[:model.nq];data.qvel[:]=state[model.nq:];mujoco.mj_forward(model,data)
                camera.lookat[:]=[data.qpos[0],data.qpos[1],.65];renderer.update_scene(data,camera=camera,scene_option=options)
                encoder.stdin.write(renderer.render().tobytes())
        finally:encoder.stdin.close();renderer.close()
        if encoder.wait()!=0:raise RuntimeError("Encoding failed")
        records.append(dict(profile=profile,source_duration_s=float(trace['time'][-1]),preview_s=8,frames=200,physics_steps_rendered=0))
        print("RENDERED",profile,flush=True)
    output=args.output/"positive_turn_comparison.mp4"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(args.output/"original_positive_turn.mp4"),"-i",str(args.output/"usd_position_positive_turn.mp4"),
        "-filter_complex","[0:v][1:v]hstack[v]","-map","[v]","-c:v","libx264","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(output),"-filter_complex",
        "[0:v]fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff:max_colors=128[p];[b][p]paletteuse=dither=bayer",str(args.output/"positive_turn_preview.gif")],check=True)
    (args.output/"render_manifest.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__":main()
