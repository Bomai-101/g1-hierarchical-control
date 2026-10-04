"""Native deterministic teacher-to-candidate evaluation, first episode only."""
import argparse,hashlib,json,sys
from pathlib import Path
from skill_handoff_protocol import cases

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser()
    for name in ('reference-root','walk-checkpoint','skill-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--skill',choices=('hold','march'),required=True);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--smoke',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'isaac_sim'));app=AppLauncher(a).app;raw=None
    try:
        import torch,numpy as np,gymnasium as gym
        import g1_walk_sim51
        from skills_sprint_task import make_cfg
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from g1_control.hierarchy.skill_metrics import metrics
        meta_path=a.reference_root/'mujoco/policies/isaac_metadata.json';meta=json.loads(meta_path.read_text())
        def actor(path,dim):
            state=torch.load(path,map_location=a.device,weights_only=False)['model_state_dict'];keys=sorted([k for k in state if k.startswith('actor.') and k.endswith('.weight')],key=lambda k:int(k.split('.')[1]));layers=[(state[k],state[k.replace('.weight','.bias')]) for k in keys]
            if layers[0][0].shape[1]!=dim or layers[-1][0].shape[0]!=37:raise RuntimeError('Actor schema')
            def act(obs):
                x=obs
                for i,(w,b) in enumerate(layers):
                    x=torch.nn.functional.linear(x,w,b)
                    if i<len(layers)-1:x=torch.nn.functional.elu(x)
                return x
            return act
        walk=actor(a.walk_checkpoint,123);candidate=actor(a.skill_checkpoint,123 if a.skill=='hold' else 125)
        cfg=make_cfg(a.skill,a.num_envs,42,a.device);cfg.episode_length_s=45.;cfg.observations.policy.enable_corruption=False
        raw=gym.make('G1-Walk-Flat-Sim51-v0',cfg=cfg).unwrapped;robot=raw.scene['robot'];sensor=raw.scene['contact_forces'];cmd=raw.command_manager.get_term('base_velocity')
        if robot.joint_names!=meta['joint_names']:raise RuntimeError('Joint order differs from frozen metadata')
        experiment_cases=cases(native=True)
        if a.smoke:experiment_cases=experiment_cases[:1]
        post=1. if a.smoke else 10.;switches=5.22+torch.arange(a.num_envs,device=a.device)*.02;offsets=torch.arange(a.num_envs,device=a.device)*2*np.pi/a.num_envs;plan=dict(skill=a.skill,cases=experiment_cases,diagnostic=True,num_envs=a.num_envs,switch_s=switches.cpu().tolist(),phase_offsets=offsets.cpu().tolist(),post_duration_s=post,blend_s=.5,phase_period_s=.9,physics_dt=.005,policy_dt=.02,seed=42,first_episode_only=True,noise=False,scope='Fixed1499 before switch, candidate after with continuous37D handoff. Parallel copies differ by switch/phase, not independent seeds. March has explicit125D internal input. Contact sensor forces>1N; not equivalent to MuJoCo geometric contact.')
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2));sources=[Path(__file__).resolve(),Path(__file__).with_name('skills_sprint_task.py'),Path(__file__).with_name('skill_handoff_protocol.py'),a.output/'plan.json',Path(__file__).resolve().parents[1]/'src/g1_control/hierarchy/skill_metrics.py',a.walk_checkpoint,a.skill_checkpoint,meta_path,a.reference_root/'isaac_sim/assets/g1_minimal.usd'];hashes={str(f.resolve()):sha(f) for f in sources};(a.output/'inputs.json').write_text(json.dumps(hashes,indent=2));results=[]
        for ci,c in enumerate(experiment_cases):
            decisions=torch.zeros_like(switches) if c['protocol']=='static' else 5.22+torch.arange(a.num_envs,device=a.device)*.02
            switches=decisions+c['ramp_s']
            folder=a.output/f'case_{ci:02d}';folder.mkdir();raw.reset(seed=42);last=torch.zeros(a.num_envs,37,device=a.device);entry=torch.zeros_like(last);entered=torch.zeros(a.num_envs,dtype=torch.bool,device=a.device);alive=np.ones(a.num_envs,dtype=bool);failure_s=np.full(a.num_envs,np.nan);rows=[];states=[];contacts=[];heights=[];valid=[];observations=[];inputs=[];actions=[];proposals=[];ticks=round((float(switches.max())+post)/.02)
            for tick in range(ticks+1):
                now=tick*.02;cmd.warm_s[:]=switches;cmd.incoming[:,0]=c['vx'];cmd.incoming[:,1]=0.;cmd.incoming[:,2]=0. if now<2.4-1e-9 else c['yaw']
                if c['ramp_s']>0:cmd.incoming[:]*=(1.-(now-decisions).clamp(min=0)/c['ramp_s']).clamp(0,1)[:,None]
                cmd._update_command()
                q=robot.data.root_quat_w;lin=robot.data.root_lin_vel_w;ang=robot.data.root_ang_vel_w;h=robot.data.root_pos_w[:,2]-raw.scene.env_origins[:,2];heading=robot.data.heading_w
                signals=torch.cat((robot.data.root_pos_w,q,lin,ang,robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,h[:,None],heading[:,None]),dim=1)
                observed_low=(h<.35).cpu().numpy();newlow=observed_low&alive;failure_s[newlow]=now;alive[newlow]=False
                rows.append(signals.cpu().numpy());states.append(torch.cat((robot.data.root_state_w,robot.data.joint_pos,robot.data.joint_vel),dim=1).cpu().numpy());contacts.append((sensor.data.net_forces_w[:,cmd.sensor_ids].norm(dim=-1)>1.).cpu().numpy());heights.append((robot.data.body_pos_w[:,cmd.foot_ids,2]-raw.scene.env_origins[:,2,None]).cpu().numpy());valid.append(alive.copy())
                if tick==ticks:break
                obs=torch.cat((robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,robot.data.projected_gravity_b,cmd.command,robot.data.joint_pos-robot.data.default_joint_pos,robot.data.joint_vel-robot.data.default_joint_vel,last),dim=1)
                after=torch.full((a.num_envs,),now,device=a.device)>=switches-1e-6;new=after&~entered;entry[new]=last[new];entered|=after;elapsed=(torch.full_like(switches,now)-switches).clamp(min=0);phase=elapsed*2*np.pi/.9+offsets
                internal=torch.cat((obs,torch.stack((torch.sin(phase),torch.cos(phase)),dim=1)*after[:,None]),dim=1) if a.skill=='march' else obs
                with torch.inference_mode():old_proposal=walk(obs);new_proposal=candidate(internal)
                u=(elapsed/.5).clamp(0,1);weight=u*u*(3-2*u);blended=entry*(1-weight[:,None])+new_proposal*weight[:,None];proposal=torch.where(after[:,None],new_proposal,old_proposal);last=new_proposal if c['protocol']=='static' else torch.where(after[:,None],blended,old_proposal)
                if not torch.isfinite(last).all():raise RuntimeError('Nonfinite candidate')
                observations.append(obs.cpu().numpy());inputs.append(internal.cpu().numpy());actions.append(last.cpu().numpy());proposals.append(proposal.cpu().numpy());_,_,terminated,truncated,_=raw.step(last)
                died=(terminated|truncated).cpu().numpy()&alive;failure_s[died]=(tick+1)*.02;alive[died]=False
            array=dict(time=np.arange(len(rows))*.02,states=np.array(states),signals=np.array(rows),feet_contact=np.array(contacts),feet_height=np.array(heights),valid=np.array(valid),base_observations=np.array(observations),skill_inputs=np.array(inputs),actions=np.array(actions),proposals=np.array(proposals),failure_s=failure_s)
            # Unwrap each independent world's Euler heading before scoring.
            array['signals'][:,:,20]=np.unwrap(array['signals'][:,:,20],axis=0);np.savez_compressed(folder/'trace.npz',**array);env_results=[]
            for j in range(a.num_envs):
                switch=round(float(switches[j])/.02)*.02;i=round(switch/.02);end=i+round(post/.02)+1;first_invalid=np.flatnonzero(~array['valid'][:,j]);end=min(end,int(first_invalid[0]) if len(first_invalid) else end)
                if end<=i+1:env_results.append(dict(env=j,passed=False,failure_s=float(failure_s[j]),reason='Failed before/during initial handoff'));continue
                score=metrics(dict(time=array['time'][:end],signals=array['signals'][:end,j]),switch);contact=array['feet_contact'][i:end,j];edges=np.diff(contact.astype(int),axis=0);losses=(edges<0).sum(axis=0);landings=(edges>0).sum(axis=0);lift=np.max(array['feet_height'][i:end,j],axis=0)-.055;full=not a.smoke and end==i+501;tilt=max(score['max_abs_roll_deg'],score['max_abs_pitch_deg']);passed=full and tilt<=20 and score['max_heading_excursion_rad']<=.2
                if a.skill=='hold':passed=passed and score['eligible_through_end_after_confirmation'] and score['net_displacement_m']<=.5
                else:passed=passed and bool(np.all(losses>=5)) and bool(np.all(landings>=5)) and bool(np.all(lift>=.025)) and score['net_displacement_m']<=.2
                env_results.append(dict(env=j,switch_s=switch,passed=bool(passed),full_post_horizon=full,decision_s=round(float(decisions[j])/.02)*.02,decision_metrics=metrics(dict(time=array['time'][:end],signals=array['signals'][:end,j]),round(float(decisions[j])/.02)*.02),failure_s=float(failure_s[j]) if np.isfinite(failure_s[j]) else None,foot_contact_losses=losses.tolist(),foot_landings=landings.tolist(),max_lift_above_ankle_floor_m=lift.tolist(),**score))
            result=dict(case=ci,**c,passed_envs=sum(r['passed'] for r in env_results),environments=a.num_envs,results=env_results);(folder/'complete.json').write_text(json.dumps(result,indent=2));results.append(result);print('ISAAC_SKILL_EVAL',ci,result['passed_envs'],'/',a.num_envs,flush=True)
        for path,digest in hashes.items():
            if sha(Path(path))!=digest:raise RuntimeError('Input changed '+path)
        (a.output/'complete.json').write_text(json.dumps(dict(skill=a.skill,engine='isaac',torch=torch.__version__,python=sys.version,results=results,candidate_only=True,smoke=a.smoke,passed_envs=sum(r['passed_envs'] for r in results),total_env_cases=len(experiment_cases)*a.num_envs),indent=2))
    finally:
        if raw is not None:raw.close()
        app.close()

if __name__=='__main__':main()
