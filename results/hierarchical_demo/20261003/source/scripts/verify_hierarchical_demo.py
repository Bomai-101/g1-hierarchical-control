"""Rebuild measured goals, watchdog decisions, actor inputs and timing evidence."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from dataclasses import asdict
import mujoco
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from g1_control.hierarchy import TaskRequest,RobotSample,Stage,TaskSequencer

def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_out(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser()
 for name in ('run','reference-root','model','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.reference_root/'mujoco'))
 from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
 from mujoco_eval.run_grid import read_base_state
 complete=json.loads((a.run/'complete.json').read_text());plan=json.loads((a.run/'plan.json').read_text());require(len(complete['results'])==6,'Scenario count');hashes=json.loads((a.run/'inputs.json').read_text())
 for path,digest in hashes.items():require(sha(Path(path))==digest,'Changed input '+path)
 require(str(a.model.resolve()) in hashes,'Unproven model');meta=json.loads((a.reference_root/'mujoco/policies/isaac_metadata.json').read_text());names=meta['joint_names'];defaults=np.array(meta['default_joint_pos']);actor=TorchActorPolicy(a.reference_root/'mujoco/policies/reproduction_20261001/policy_actor.npz');stages=[Stage(**{**r,'command':tuple(r['command'])}) for r in plan['stages']];rows=[];actors=0;states=0;replays={}
 for result in complete['results']:
  folder=a.run/result['label'];x=np.load(folder/'trace.npz');decisions=json.loads((folder/'decisions.json').read_text());packets=list(csv.DictReader((folder/'packets.csv').open()));events=json.loads((folder/'events.json').read_text());timing=list(csv.DictReader((folder/'timing.csv').open()));count=result['policy_calls'];require(x['states'].shape==(count+1,87) and x['actions'].shape==(count,37),'Counts');require(result['physical_steps']==20*count,'Step count');require(len(timing)==count and len(decisions)==count+1,'Records');require(np.max(abs(x['time']-np.arange(count+1)*.02))<1e-12,'Timestamps')
  dropout=plan['scenarios'][result['scenario']];packet_map={}
  for i,row in enumerate(packets):
   t=float(row['time_s']);require(abs(t-i*.1)<1e-10 and int(row['sequence'])==i and row['task_id']==plan['task_id'],'Source heartbeat');should_deliver=dropout is None or not (dropout[0]<=t<(dropout[1] if dropout[1] is not None else float('inf')));require((row['delivered']=='True')==should_deliver,'Fault injection');require(float(row['issued_s'])==t,'Issue time')
   if should_deliver:packet_map[round(t/.02)]=TaskRequest(row['task_id'],int(row['sequence']),float(row['issued_s']))
  engine=TaskSequencer(plan['task_id'],stages,message_timeout_s=plan['message_timeout_s'],abort_after_stale_s=plan['abort_after_stale_s'],recovery_messages=plan['recovery_messages'],record_events=result['event_logging']);m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in names]);last=np.zeros(37);old_yaw=0.;unwrapped=0.
  for tick,state in enumerate(x['states']):
   require(np.all(np.isfinite(state)),'Finite state');d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d);quat,lin,ang,height=read_base_state(m,d,'pelvis');yaw=float(np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2)));unwrapped+=(yaw-old_yaw+np.pi)%(2*np.pi)-np.pi;old_yaw=yaw;blin=quat_rotate_inverse(quat,lin);bang=quat_rotate_inverse(quat,ang);signal=np.r_[d.qpos[:3],quat,lin,ang,blin,bang,height,unwrapped];require(np.max(abs(signal-x['signals'][tick]))<1e-11,'State-to-signal');require(height>=.35,'Height');now=tick*.02;decision=engine.update(RobotSample(now,float(d.qpos[0]),float(d.qpos[1]),float(unwrapped)),packet_map.get(tick));require(json.loads(json.dumps(asdict(decision)))==decisions[tick],'Task/watchdog replay');require(not decision.switch_authorized and not decision.fallback_authorized,'Recovery authority');states+=1
   if tick<count:
    require(decision.task_state=='running' and decision.command is not None,'Invalid skill admission');command=np.array(decision.command);obs=build_policy_observation(quat,lin,ang,command,d.qpos[qa],d.qvel[va],defaults,last);require(np.array_equal(obs,x['observations'][tick]),'Observation');action=actor.act(obs);require(np.max(abs(action-x['actions'][tick]))<1e-6,'Actor');last=x['actions'][tick];target=defaults+meta['action_scale']*last;require(np.array_equal(target,x['targets'][tick]),'Target scaling');actors+=1
    if tick%32==0:
     d.ctrl[ai]=target;mujoco.mj_forward(m,d);require(np.all(abs(d.qfrc_actuator[va])<=m.jnt_actfrcrange[ids,1]+1e-7),'Force cap')
  require(json.loads(json.dumps(engine.events))==events,'Event replay');require(np.all(x['force_max']<=m.jnt_actfrcrange[ids,1]+1e-7),'Peak cap');require(engine.state==result['task_state'] and decision.reason==result['termination'],'Termination');require(abs(x['time'][-1]-result['elapsed_simulation_s'])<1e-12,'Elapsed')
  for key,quantiles in result['timing_ms_percentiles'].items():
   values=np.array([float(r[key]) for r in timing]);require(np.all(values>=0),'Negative timing')
   for q,value in quantiles.items():require(abs(float(np.percentile(values,float(q)))-value)<1e-12,'Timing quantile')
  require(sum(float(r['frame_ms'])>20 for r in timing)==result['frames_over_20ms'],'Deadline count')
  stale=[e for e in events if e['event']=='watchdog_stale'];recovered=[e for e in events if e['event']=='watchdog_recovered'];row=dict(scenario=result['scenario'],event_logging=result['event_logging'],task_state=engine.state,termination=decision.reason,duration_s=float(x['time'][-1]),stage_final=decision.stage_name,stale_at_s=stale[0]['time_s'] if stale else '',recovered_at_s=recovered[0]['time_s'] if recovered else '',task_logic_p99_ms=result['timing_ms_percentiles']['task_logic_ms']['99'],actor_p99_ms=result['timing_ms_percentiles']['observation_actor_ms']['99'],frame_p99_ms=result['timing_ms_percentiles']['frame_ms']['99'],frames_over_20ms=result['frames_over_20ms']);rows.append(row);replays[result['label']]=(x,decisions,result)
 for scenario in plan['scenarios']:
  left,ld,lr=replays[scenario+'_events'];right,rd,rr=replays[scenario+'_no_events'];require(all(np.array_equal(left[k],right[k]) for k in left.files),'Logging changes trajectory');require(ld==rd,'Logging changes decisions');require(lr['physics_state_control_sha256']==rr['physics_state_control_sha256'],'Logging physics digest')
  if scenario=='persistent_dropout':require(lr['termination']=='communication_timeout','Missing timeout')
  else:require(lr['termination']=='all_goals_reached','Mission incomplete')
 csv_out(a.output/'summary.csv',rows);proof=dict(status='PASS',states_rebuilt=states,actor_outputs_recomputed=actors,task_watchdog_replay=True,event_logging_pairs_exact=3,physics_digest_pairs_exact=3,scope='Measured-goal and message-freshness logic replay, all sampled states/observations/actions/targets, events and recorded timing metrics. Digest equality proves logging noninterference, not measured recovery/safe stopping.');(a.output/'verification.json').write_text(json.dumps(proof,indent=2));raw_hashes={str(p.resolve()):sha(p) for p in a.run.rglob('*') if p.is_file()};(a.output/'raw_hashes.json').write_text(json.dumps(raw_hashes,indent=2));print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
