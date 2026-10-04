"""Render Isaac recorded poses on matching MuJoCo geometry, with zero physics steps."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco,numpy as np
from render_hierarchical_demo import stamp
from skill_sequence_protocol import NAMES
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
z=dict(np.load(a.case/'trace.npz'));r=json.loads((a.case/'complete.json').read_text())['results'][0];j=0;terminal=np.flatnonzero(np.isin(z['stage'][:,j],[4,5]));last=int(terminal[0]) if len(terminal) else len(z['time'])-1;duration=float(z['time'][last]);fps=25;frames=int(np.ceil(duration*fps))+1
meta=json.loads(Path('/home/omai/robotics/reference_projects/g1_walk_isaaclab_mujoco/mujoco/policies/isaac_metadata.json').read_text());m=mujoco.MjModel.from_binary_path(str(a.model));ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in meta['joint_names']];qa=m.jnt_qposadr[ids];data=mujoco.MjData(m);m.vis.global_.offwidth=640;m.vis.global_.offheight=480;m.vis.headlight.ambient[:]=.6;m.vis.headlight.diffuse[:]=.6;renderer=mujoco.Renderer(m,480,640);camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=135;camera.elevation=-20;options=mujoco.MjvOption();options.geomgroup[3:]=0
header='''[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 480
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,7,12,12,12,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
''';lines=[]
for frame in range(frames):
    now=min(frame/fps,duration);idx=min(int(np.searchsorted(z['time'],now,side='right')-1),last);s=z['signals'][idx,j];text=f'Isaac saved-state replay | {r["protocol"]} | yaw={r["yaw"]:+.1f}\\NStage: {NAMES[z["stage"][idx,j]]} | t={now:.2f}s\\NBody speed={np.linalg.norm(s[13:15]):.3f} m/s\\NMuJoCo geometry render only; no physics validation';lines.append(f'Dialogue: 0,{stamp(frame/fps)},{stamp((frame+1)/fps)},Default,,0,0,0,,'+r'{\an7\pos(12,12)}'+text)
sub=a.output/'sequence.ass';sub.write_text(header+'\n'.join(lines)+'\n');out=a.output/'walk_hold_walk.mp4';encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r',str(fps),'-i','-','-vf',"drawbox=x=0:y=0:w=iw:h=110:color=black@0.8:t=fill,ass=filename='"+str(sub.resolve())+"'",'-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
origin=z['states'][0,j,:2].copy()
try:
    for frame in range(frames):
        now=min(frame/fps,duration);idx=min(int(np.searchsorted(z['time'],now,side='right')-1),last);state=z['states'][idx,j];data.qpos[:]=m.qpos0;data.qpos[:3]=state[:3];data.qpos[:2]-=origin;data.qpos[3:7]=state[3:7];data.qpos[qa]=state[13:50];mujoco.mj_kinematics(m,data);mujoco.mj_comPos(m,data);camera.lookat[:]=[data.qpos[0],data.qpos[1],.65];renderer.update_scene(data,camera=camera,scene_option=options);
        for grid in range(-16,17):
            for begin,finish in (([grid*.25,-4,.002],[grid*.25,4,.002]),([-4,grid*.25,.002],[4,grid*.25,.002])):
                geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.5,.55,.6,1.],dtype=np.float32));mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.,np.array(begin),np.array(finish));renderer.scene.ngeom+=1
        encoder.stdin.write(renderer.render().tobytes())
finally:encoder.stdin.close();renderer.close()
if encoder.wait()!=0:raise RuntimeError('Encoding failed')
subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out),'-ss',str(r['resume_start_s'] or 0),'-frames:v','1',str(a.output/'preview.png')],check=True)
(a.output/'render_manifest.json').write_text(json.dumps(dict(source_trace=str(a.case/'trace.npz'),trace_sha256=hashlib.sha256((a.case/'trace.npz').read_bytes()).hexdigest(),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),env=j,selection='Preselected direct / yaw0 / env0; not best-case selection',frames=frames,fps=fps,duration_s=duration,physics_steps=0,scope='Isaac recorded physical trajectory, displayed using MuJoCo geometry; not MuJoCo physical replay.'),indent=2));print('RENDERED',out)
