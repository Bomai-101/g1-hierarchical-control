"""Bounded warm-start skill training; isolated outputs and explicit interfaces."""
import argparse,hashlib,json,sys,time
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser()
    for name in ('reference-root','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--skill',choices=('hold','march'),required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iters',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    if a.iters<1:raise ValueError('Positive training iterations required')
    a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_root/'isaac_sim'))
    sources=[Path(__file__).resolve(),Path(__file__).with_name('skills_sprint_task.py'),a.checkpoint,a.reference_root/'isaac_sim/assets/g1_minimal.usd',*[a.reference_root/'isaac_sim/g1_walk_sim51'/n for n in ('g1_env_cfg.py','g1_asset_cfg.py','mdp.py','ppo_cfg.py')]]
    hashes={str(f.resolve()):sha(f) for f in sources};(a.output/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
    app=AppLauncher(a).app;env=None
    try:
        import torch,gymnasium as gym
        import g1_walk_sim51
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from rsl_rl.runners import OnPolicyRunner
        from g1_walk_sim51.ppo_cfg import G1FlatPPORunnerCfg
        from skills_sprint_task import make_cfg
        cfg=make_cfg(a.skill,a.num_envs,a.seed,a.device);cfg.log_dir=str(a.output)
        agent=G1FlatPPORunnerCfg();agent.device=a.device;agent.seed=a.seed;agent.save_interval=100;agent.max_iterations=a.iters;agent.algorithm.learning_rate=3e-4;agent.algorithm.desired_kl=.01
        dump_yaml(str(a.output/'params/env.yaml'),cfg);dump_yaml(str(a.output/'params/agent.yaml'),agent)
        env=RslRlVecEnvWrapper(gym.make('G1-Walk-Flat-Sim51-v0',cfg=cfg));runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(a.output),device=a.device)
        expected=123 if a.skill=='hold' else 125
        state=torch.load(a.checkpoint,map_location=a.device,weights_only=False)['model_state_dict'];state={k:v.clone() for k,v in state.items()}
        adaptation=[]
        for key in ('actor.0.weight','critic.0.weight'):
            if state[key].shape[1]!=123:raise RuntimeError('Source checkpoint schema')
            if expected==125:
                state[key]=torch.cat([state[key],torch.zeros(state[key].shape[0],2,device=a.device)],dim=1);adaptation.append(key+' append two zero columns')
        state['std']=torch.full_like(state['std'],.3)
        state['critic.6.weight'].zero_();state['critic.6.bias'].zero_()
        runner.alg.policy.load_state_dict(state);runner.current_learning_iteration=0
        obs=env.get_observations()
        if obs['policy'].shape[1]!=expected:raise RuntimeError('Actual observation schema '+str(obs['policy'].shape))
        protocol=dict(skill=a.skill,initial_checkpoint=str(a.checkpoint.resolve()),initial_sha256=sha(a.checkpoint),obs_dim=expected,action_dim=37,base_obs_dim=123,extra_phase=('sin/cos appended; zero during incoming walk; period0.90s' if a.skill=='march' else None),actor_warm_started=True,first_layers_adaptation=adaptation,optimizer_loaded=False,critic_final_layer_reset=True,initial_noise_std=.3,iterations=a.iters,num_envs=a.num_envs,seed=a.seed,episode_s=12.,incoming_walk_s=[1.,3.],immediate_stationary_probability=.3,policy_dt=.02,physics_dt=.005,original_position_actuators_preserved=True,candidate_only=True,early_observation=dict(min_height=float(env.unwrapped.scene['robot'].data.root_pos_w[:,2].min()),max_height=float(env.unwrapped.scene['robot'].data.root_pos_w[:,2].max())),python=sys.version,torch=torch.__version__)
        (a.output/'training_protocol.json').write_text(json.dumps(protocol,indent=2));print('SKILL_TRAIN_START '+json.dumps(protocol),flush=True)
        start=time.perf_counter();runner.learn(num_learning_iterations=a.iters,init_at_random_ep_len=False)
        final=a.output/f'model_{runner.current_learning_iteration}.pt'
        if not final.exists():raise RuntimeError('Missing final checkpoint')
        for path,digest in hashes.items():
            if sha(Path(path))!=digest:raise RuntimeError('Input changed '+path)
        result=dict(protocol,final_checkpoint=str(final.resolve()),final_sha256=sha(final),elapsed_wall_s=time.perf_counter()-start,inputs_unchanged=True)
        (a.output/'complete.json').write_text(json.dumps(result,indent=2));print('SKILL_TRAIN_COMPLETE '+json.dumps(result),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

if __name__=='__main__':main()
