"""Measured-goal task sequence, task-message watchdog, and fixed1499 skill."""
import argparse,csv,hashlib,json,sys,time
from dataclasses import asdict
from pathlib import Path
import mujoco
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from g1_control.hierarchy import TaskRequest,RobotSample,Stage,TaskSequencer
from g1_control.hierarchy.skill_interface import VelocityCommand

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_file(p,rows):
 if not rows:raise RuntimeError('Empty evidence '+str(p))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('reference-root','model','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--suite',choices=('initial','handoff'),default='initial')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'mujoco'))
 from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
 from mujoco_eval.run_grid import read_base_state
 meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text());policy_path=a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz';actor=TorchActorPolicy(policy_path);defaults=np.array(meta['default_joint_pos']);names=meta['joint_names']
 stages=(Stage('walk_1','forward_distance',1.,(.5,0.,0.)),Stage('turn','relative_heading',.4,(.5,0.,.2)),Stage('walk_2','forward_distance',1.,(.5,0.,0.)))
 drop_start=5. if a.suite=='handoff' else 3.
 scenarios={'normal':None,'temporary_dropout':(drop_start,drop_start+1.),'persistent_dropout':(drop_start,float('inf'))}
 plan=dict(suite=a.suite,task_id='forward_turn_forward',stages=[asdict(s) for s in stages],physics_dt=.001,policy_dt=.02,heartbeat_dt=.1,message_timeout_s=.3,abort_after_stale_s=2.,recovery_messages=2,max_duration_s=20,scenarios={'normal':None,'temporary_dropout':[drop_start,drop_start+1.],'persistent_dropout':[drop_start,None]},clock='All freshness, goal and abort logic uses simulation time. Wall-clock timings are separately measured.',watchdog_response='Block new task stages, retain last velocity skill command. Abort ends simulation experiment; no physical stop/fallback.',goal_measurements='forward distance projected onto stage-entry heading; relative unwrapped Euler yaw. No vision model or learned high-level planner.',comparisons='Record task events enabled/disabled for each scenario; entire control/state digest must match. Watchdog logic active in both.')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2));files=[Path(__file__).resolve(),a.model,meta_path,policy_path,a.output/'plan.json',REPO/'src/g1_control/hierarchy/task_sequence.py',REPO/'src/g1_control/hierarchy/skill_interface.py',a.reference_root/'mujoco/mujoco_eval/policy.py',a.reference_root/'mujoco/mujoco_eval/run_grid.py'];hashes={str(f.resolve()):sha(f) for f in files};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2));results=[]
 for scenario,dropout in scenarios.items():
  for logging in (True,False):
   label=scenario+('_events' if logging else '_no_events');out=a.output/label;out.mkdir()
   m=mujoco.MjModel.from_binary_path(str(a.model))
   if abs(m.opt.timestep-.001)>1e-12:raise RuntimeError('Wrong physics step')
   d=mujoco.MjData(m);ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names])
   if min(ids)<0 or min(ai)<0:raise RuntimeError('Joint/actuator contract')
   d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,.74];d.qpos[3:7]=[1,0,0,0];d.qpos[qa]=defaults;mujoco.mj_forward(m,d)
   engine=TaskSequencer(plan['task_id'],stages,record_events=logging);last=np.zeros(37);peak=np.zeros(37);digest=hashlib.sha256();states=[];signals=[];obs_list=[];actions=[];targets=[];decisions=[];packets=[];timings=[];previous_yaw=0.;unwrapped=0.;steps=0;termination='experiment_time_limit'
   for tick in range(1001):
    frame_start=time.perf_counter_ns();now=tick*.02;mujoco.mj_forward(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');yaw=float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2)));unwrapped+=float((yaw-previous_yaw+np.pi)%(2*np.pi)-np.pi);previous_yaw=yaw;blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang)
    states.append(np.r_[d.qpos,d.qvel].copy());signals.append(np.r_[d.qpos[:3],quat,lin,ang,blin,bang,height,unwrapped]);request=None
    if tick%5==0:
     delivered=not(dropout is not None and dropout[0]<=now<dropout[1]);packet=TaskRequest(plan['task_id'],tick//5,now);packets.append(dict(time_s=now,sequence=packet.sequence,task_id=packet.task_id,issued_s=packet.issued_s,delivered=delivered));request=packet if delivered else None
    task_start=time.perf_counter_ns();decision=engine.update(RobotSample(now,float(d.qpos[0]),float(d.qpos[1]),unwrapped),request);task_ms=(time.perf_counter_ns()-task_start)/1e6;decisions.append(asdict(decision))
    if height<.35:termination='height_termination';break
    if decision.task_state in ('completed','aborted'):termination=decision.reason;break
    if tick==1000:break
    if decision.command is None:raise RuntimeError('No skill command available')
    command=np.array(VelocityCommand(*decision.command).values);actor_start=time.perf_counter_ns();obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last);last=actor.act(obs).astype(float);target=defaults+meta['action_scale']*last;actor_ms=(time.perf_counter_ns()-actor_start)/1e6;obs_list.append(obs);actions.append(last.copy());targets.append(target.copy());d.ctrl[ai]=target
    physical_start=time.perf_counter_ns()
    for substep in range(20):
     mujoco.mj_step(m,d);steps+=1;peak=np.maximum(peak,np.abs(d.qfrc_actuator[va]))
     for values in (d.qpos,d.qvel,d.ctrl):digest.update(values.tobytes())
     if not np.all(np.isfinite(d.qpos)) or any(w.number for w in d.warning):raise RuntimeError('Nonfinite or physics warning')
    physical_ms=(time.perf_counter_ns()-physical_start)/1e6;frame_ms=(time.perf_counter_ns()-frame_start)/1e6;timings.append(dict(time_s=now,task_logic_ms=task_ms,observation_actor_ms=actor_ms,physics_and_digest_ms=physical_ms,frame_ms=frame_ms,over_20ms=frame_ms>20))
   if np.any(peak>m.jnt_actfrcrange[ids,1]+1e-7):raise RuntimeError('Force cap exceeded')
   np.savez_compressed(out/'trace.npz',time=np.arange(len(states))*.02,states=states,signals=signals,observations=obs_list,actions=actions,targets=targets,force_max=peak)
   (out/'decisions.json').write_text(json.dumps(decisions,indent=2));(out/'events.json').write_text(json.dumps(engine.events,indent=2));csv_file(out/'packets.csv',packets);csv_file(out/'timing.csv',timings)
   timing_summary={k:{str(q):float(np.percentile([r[k] for r in timings],q)) for q in (50,95,99)} for k in ('task_logic_ms','observation_actor_ms','physics_and_digest_ms','frame_ms')}
   result=dict(label=label,scenario=scenario,event_logging=logging,termination=termination,task_state=engine.state,elapsed_simulation_s=steps*.001,physical_steps=steps,policy_calls=len(actions),states=len(states),final_stage=decision.stage_name,final_progress=decision.progress,watchdog_final_state=decision.watchdog_state,physics_state_control_sha256=digest.hexdigest(),timing_ms_percentiles=timing_summary,frames_over_20ms=sum(r['over_20ms'] for r in timings),note='Timing includes this headless recorder and digest workload, no hard-real-time or ROS/network measurement. Ending the experiment is not a physical stop.')
   (out/'complete.json').write_text(json.dumps(result,indent=2));results.append(result);print('HIERARCHY_DEMO',label,termination,result['elapsed_simulation_s'],flush=True)
 for path,expected in hashes.items():
  if sha(Path(path))!=expected:raise RuntimeError('Input changed '+path)
 (a.output/'complete.json').write_text(json.dumps(dict(results=results,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,plan=plan),indent=2))

if __name__=='__main__':main()
