"""Prospective first-episode, frozen-policy walk/hold/walk experiment."""
import argparse, hashlib, json, sys
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser()
    for name in ('reference-root','walk-checkpoint','skill-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=12);p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303]);p.add_argument('--smoke',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'isaac_sim'));app=AppLauncher(a).app;raw=None
    try:
        import torch,numpy as np,gymnasium as gym
        import g1_walk_sim51
        from skills_sprint_task import make_cfg
        from skill_sequence_protocol import Sequence as NormalSequence,WALK,BRAKE,HOLD,RESUME,DONE,ABORT
        from skill_watchdog_protocol import WatchdogSequence as Sequence
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from g1_control.hierarchy.skill_metrics import metrics
        def actor(path):
            state=torch.load(path,map_location=a.device,weights_only=False)['model_state_dict'];keys=sorted([k for k in state if k.startswith('actor.') and k.endswith('.weight')],key=lambda k:int(k.split('.')[1]));layers=[(state[k],state[k.replace('.weight','.bias')]) for k in keys]
            if layers[0][0].shape[1]!=123 or layers[-1][0].shape[0]!=37:raise RuntimeError('Actor schema')
            def act(obs):
                x=obs
                for i,(w,b) in enumerate(layers):
                    x=torch.nn.functional.linear(x,w,b)
                    if i<len(layers)-1:x=torch.nn.functional.elu(x)
                return x
            return act
        walk,hold=actor(a.walk_checkpoint),actor(a.skill_checkpoint)
        cfg=make_cfg('hold',a.num_envs,a.seeds[0],a.device);cfg.episode_length_s=45.;cfg.observations.policy.enable_corruption=False
        raw=gym.make('G1-Walk-Flat-Sim51-v0',cfg=cfg).unwrapped;robot=raw.scene['robot'];sensor=raw.scene['contact_forces'];cmd=raw.command_manager.get_term('base_velocity')
        meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json'
        if robot.joint_names!=json.loads(meta_path.read_text())['joint_names']:raise RuntimeError('Joint order')
        sources=[Path(__file__).resolve(),Path(__file__).with_name('skill_sequence_protocol.py'),Path(__file__).with_name('skill_watchdog_protocol.py'),Path(__file__).with_name('skills_sprint_task.py'),a.walk_checkpoint,a.skill_checkpoint,meta_path,a.reference_root/'isaac_sim/assets/g1_minimal.usd']
        sources.extend((a.reference_root/'isaac_sim/g1_walk_sim51').rglob('*.py'));sources.extend([a.walk_checkpoint.parent/'policy_actor.npz',a.skill_checkpoint.parent/'export/policy_actor.npz']);hashes={str(f.resolve()):sha(f) for f in sources};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2))
        plan=dict(watchdog=dict(heartbeat_hz=10,ttl_s=.3,recovery_messages=2,perception_fault=False,scope='Task request freshness, not perception pipeline'),protocols=['normal','temporary','persistent'],yaw_commands=[-.2,0.,.2],vx=.5,num_envs=a.num_envs,seeds=a.seeds,decision_s=[(3.22,6.22,9.22)[j//4]+(j%4)*.02 for j in range(a.num_envs)],reset_variation='Inherited flat-task random world root x/y and yaw; fixed initial joint poses/zero velocities, no observation noise/force or rough terrain',policy_dt=.02,physics_dt=.005,gate=dict(speed_mean_max=.05,euler_yaw_rms_max=.05,window_s=.5,tilt_max_deg=20,both_feet=True,confirmation_s=1,retention_s=2,min_hold_s=2,max_hold_s=12),resume_s=10,resume_command=[.5,0,0],exploratory_sequence_acceptance=dict(resume_tail_vx_rmse_max=.2,resume_tail_euler_yaw_rms_max=.15,resume_tilt_max_deg=20),first_episode_only=True,noise=False,smoke=a.smoke,scope='Watchdog experiment using previously tested direct low-speed sequence. Persistent loss does not authorize return-to-walk. Sequencing diagnostic, not original standalone skill acceptance or safety certification. Abort terminates scoring, not a physical recovery. Within a seed, parallel copies share reset seed; evaluate variation without assuming independent Bernoulli trials.')
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2));all_results=[]
        cases=[(seed,protocol,yaw) for seed in a.seeds for protocol in plan['protocols'] for yaw in plan['yaw_commands']]
        if a.num_envs!=12:raise RuntimeError('Fixed prospective protocol requires12 environments')
        if a.smoke:cases=cases[:1]
        for ci,(seed,protocol,yaw) in enumerate(cases):
            folder=a.output/f'case_{ci:02d}';folder.mkdir();raw.reset(seed=seed)
            seq=[NormalSequence(plan['decision_s'][j],'direct') if protocol=='normal' else Sequence(plan['decision_s'][j],protocol) for j in range(a.num_envs)];last=torch.zeros(a.num_envs,37,device=a.device);entry=torch.zeros_like(last);prev=np.full(a.num_envs,WALK);alive=np.ones(a.num_envs,bool);failure=np.full(a.num_envs,np.nan)
            arrays={k:[] for k in ('signals','states','feet_contact','feet_height','valid','stage','commands','weights','base_observations','proposals','actions')}
            ticks=round((max(plan['decision_s'])+.32+22.)/.02)
            if a.smoke:ticks=330
            for tick in range(ticks+1):
                now=tick*.02
                h=robot.data.root_pos_w[:,2]-raw.scene.env_origins[:,2]
                signals=torch.cat((robot.data.root_pos_w,robot.data.root_quat_w,robot.data.root_lin_vel_w,robot.data.root_ang_vel_w,robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,h[:,None],robot.data.heading_w[:,None]),dim=1).cpu().numpy()
                feet=(sensor.data.net_forces_w[:,cmd.sensor_ids].norm(dim=-1)>1.).cpu().numpy()
                low=(h<.35).cpu().numpy()&alive;failure[low]=now;alive[low]=False
                stage=np.array([q.update(now,signals[j],feet[j],bool(alive[j])) for j,q in enumerate(seq)])
                commands=np.array([q.command(now,yaw) for q in seq]);weights=np.array([q.weight(now) if q.stage in (HOLD,RESUME) else 1. for q in seq])
                for key,val in dict(signals=signals,states=torch.cat((robot.data.root_state_w,robot.data.joint_pos,robot.data.joint_vel),dim=1).cpu().numpy(),feet_contact=feet,feet_height=(robot.data.body_pos_w[:,cmd.foot_ids,2]-raw.scene.env_origins[:,2,None]).cpu().numpy(),valid=alive.copy(),stage=stage,commands=commands,weights=weights).items():arrays[key].append(val)
                if tick==ticks or np.all(np.isin(stage,[DONE,ABORT])):break
                # Prevent the task's training command scheduler from replacing experimental commands.
                cmd.warm_s[:]=1000.;cmd.incoming[:]=torch.as_tensor(commands,device=a.device,dtype=torch.float32);cmd._update_command()
                obs=torch.cat((robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,robot.data.projected_gravity_b,cmd.command,robot.data.joint_pos-robot.data.default_joint_pos,robot.data.joint_vel-robot.data.default_joint_vel,last),dim=1)
                new=np.isin(stage,[HOLD,RESUME])&(stage!=prev);entry[new]=last[new]
                with torch.inference_mode():wp=walk(obs);hp=hold(obs)
                holding=torch.as_tensor(np.isin(stage,[HOLD,ABORT]),device=a.device)
                proposal=torch.where(holding[:,None],hp,wp);weight=torch.as_tensor(weights,device=a.device,dtype=torch.float32)
                blending=torch.as_tensor(np.isin(stage,[HOLD,RESUME]),device=a.device)
                last=torch.where(blending[:,None],entry*(1-weight[:,None])+proposal*weight[:,None],proposal)
                if not torch.isfinite(last).all():raise RuntimeError('Nonfinite action')
                for key,val in dict(base_observations=obs.cpu().numpy(),proposals=proposal.cpu().numpy(),actions=last.cpu().numpy()).items():arrays[key].append(val)
                prev=stage.copy();_,_,terminated,truncated,_=raw.step(last);died=(terminated|truncated).cpu().numpy()&alive;failure[died]=(tick+1)*.02;alive[died]=False
            array={k:np.asarray(v) for k,v in arrays.items()};array['time']=np.arange(len(array['signals']))*.02;array['signals'][:,:,20]=np.unwrap(array['signals'][:,:,20],axis=0);array['failure_s']=failure
            np.savez_compressed(folder/'trace.npz',**array);results=[]
            for j,q in enumerate(seq):
                stages=array['stage'][:,j];end=np.flatnonzero(np.isin(stages,[DONE,ABORT]));end=int(end[0])+1 if len(end) else len(stages)
                time=array['time'][:end];s=array['signals'][:end,j]
                score=dict(env=j,seed=seed,planned_trigger_s=plan['decision_s'][j],protocol=protocol,yaw=yaw,decision_s=q.decision if np.isfinite(q.decision) else None,hold_start_s=q.hold_start if np.isfinite(q.hold_start) else None,resume_start_s=q.resume_start,final_stage=int(stages[end-1]),events=q.events,watchdog_events=getattr(q,'watchdog_events',[]),drop_start_s=getattr(q,'drop_start',None),drop_duration_s=1. if protocol=='temporary' else None,failure_s=float(failure[j]) if np.isfinite(failure[j]) else None,sequence_completed=bool(stages[end-1]==DONE),sequence_passed=False)
                if np.isfinite(q.hold_start) and len(time)>round(q.hold_start/.02)+1:
                    hi=round(q.hold_start/.02);he=round(q.resume_start/.02)+1 if q.resume_start is not None else end
                    score['hold_metrics']=metrics(dict(time=time[:he],signals=s[:he]),q.hold_start)
                    di=round(q.decision/.02);score['decision_to_hold_exit_path_m']=float(np.linalg.norm(np.diff(s[di:he,:2],axis=0),axis=1).sum());score['decision_to_hold_exit_net_m']=float(np.linalg.norm(s[he-1,:2]-s[di,:2]))
                if q.resume_start is not None and end>round(q.resume_start/.02)+1:
                    ri=round(q.resume_start/.02);rs=metrics(dict(time=time,signals=s),q.resume_start);tail=time>=time[-1]-2.-1e-8;rmse=float(np.sqrt(np.mean((s[tail,13]-.5)**2)));score['resume_metrics']=rs;score['resume_tail_vx_rmse_mps']=rmse
                    score['sequence_passed']=bool(score['sequence_completed'] and rmse<=.2 and rs['final_2s_euler_yaw_rate_rms_rps']<=.15 and max(rs['max_abs_roll_deg'],rs['max_abs_pitch_deg'])<=20)
                results.append(score)
            result=dict(case=ci,seed=seed,protocol=protocol,yaw=yaw,results=results);(folder/'complete.json').write_text(json.dumps(result,indent=2));all_results.extend(results);print('RETEST',seed,protocol,yaw,'resumed',sum(r['resume_start_s'] is not None for r in results),'completed',sum(r['sequence_completed'] for r in results),'passed',sum(r['sequence_passed'] for r in results),flush=True)
        for path,digest in hashes.items():
            if sha(Path(path))!=digest:raise RuntimeError('Input changed')
        (a.output/'complete.json').write_text(json.dumps(dict(engine='isaac',python=sys.version,torch=torch.__version__,smoke=a.smoke,results=all_results),indent=2))
    finally:
        if raw is not None:raw.close()
        app.close()
if __name__=='__main__':main()
