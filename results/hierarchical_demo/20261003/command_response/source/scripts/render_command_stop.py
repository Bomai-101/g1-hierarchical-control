"""Render independent left-turn command responses; saved states, no physics."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np

def stamp(seconds):
 cs=round(seconds*100);return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','model','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);fps=25;frames=400;outputs=[];manifest=[]
 for scenario,title in [('left_hold','HOLD COMMAND'),('left_zero','IMMEDIATE ZERO'),('left_ramp','0.50s RAMP TO ZERO')]:
  folder=a.run/scenario;x=np.load(folder/'trace.npz');subtitles=a.output/(scenario+'.ass');header='''[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 480
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,17,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,7,12,12,12,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
''';lines=[]
  for frame in range(frames):
   now=min(frame/fps,float(x['time'][-1]));idx=int(np.searchsorted(x['time'],now+1e-9,side='right')-1);command=x['commands'][min(idx,len(x['commands'])-1)];speed=float(np.linalg.norm(x['signals'][idx,13:15]));heading=float(np.rad2deg(x['signals'][idx,20]-x['signals'][261,20]));ended=frame/fps>=float(x['time'][-1]);phase='before response' if now<5.22 else 'command response, not safe stop';status='SIMULATION ENDED - last sample' if ended else 'Independent experiment; no watchdog integration';text=f"{title} | fixed1499\\Nsim={now:.2f}s | {phase}\\Ncmd vx={command[0]:.2f}, yaw={command[2]:.2f}\\Nactual speed={speed:.3f}m/s | delta yaw={heading:.1f}deg\\N{status}";text=r'{\an7\pos(12,12)}'+text;lines.append(f'Dialogue: 0,{stamp(frame/fps)},{stamp((frame+1)/fps)},Default,,0,0,0,,{text}')
  subtitles.write_text(header+'\n'.join(lines)+'\n');m=mujoco.MjModel.from_binary_path(str(a.model));m.vis.global_.offwidth=640;m.vis.global_.offheight=480;m.vis.headlight.ambient[:]=.6;m.vis.headlight.diffuse[:]=.6;data=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=480,width=640);options=mujoco.MjvOption();options.geomgroup[3:]=0;camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.7;camera.azimuth=135;camera.elevation=-20;out=a.output/(scenario+'.mp4');filter_arg="drawbox=x=0:y=0:w=iw:h=125:color=black@0.8:t=fill,ass=filename='"+str(subtitles.resolve())+"'";encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r',str(fps),'-i','-','-vf',filter_arg,'-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
  try:
   for frame in range(frames):
    now=min(frame/fps,float(x['time'][-1]));idx=int(np.searchsorted(x['time'],now+1e-9,side='right')-1);state=x['states'][idx];data.qpos[:]=state[:m.nq];data.qvel[:]=state[m.nq:];mujoco.mj_kinematics(m,data);mujoco.mj_comPos(m,data);camera.lookat[:]=[data.qpos[0],data.qpos[1],.65];renderer.update_scene(data,camera=camera,scene_option=options)
    # Visual-only world grid, added to the rendering scene, never the model.
    for grid in range(-16,17):
     for begin,end in (([grid*.25,-4,.002],[grid*.25,4,.002]),([-4,grid*.25,.002],[4,grid*.25,.002])):
      geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.5,.55,.6,1.],dtype=np.float32));mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.,np.array(begin),np.array(end));renderer.scene.ngeom+=1
    encoder.stdin.write(renderer.render().tobytes())
  finally:encoder.stdin.close();renderer.close()
  if encoder.wait()!=0:raise RuntimeError('Encoding failed')
  outputs.append(out);manifest.append(dict(scenario=scenario,source_trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest(),simulation_end_s=float(x['time'][-1]),frames=frames,fps=fps,physics_steps=0,visual_grid='Rendering-scene line primitives only, no physical model changes',after_end='Hold last recorded state with SIMULATION ENDED label, not a physical stop.'));print('RENDERED',scenario,flush=True)
 combined=a.output/'command_response.mp4';subprocess.run(['ffmpeg','-y','-loglevel','error',*[arg for out in outputs for arg in ('-i',str(out))],'-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(combined)],check=True)
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(combined),'-frames:v','1',str(a.output/'preview.png')],check=True);(a.output/'render_manifest.json').write_text(json.dumps(dict(cases=manifest,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
