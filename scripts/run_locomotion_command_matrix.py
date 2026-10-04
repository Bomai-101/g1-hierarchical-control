"""Paired physical rollouts: command schedules, two speeds, seeded rough terrain."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from verify_locomotion_shadow import verify_pair

POLICIES=('baseline','reproduction_20261001')
PLANS=(('steps',0.),('slow_sine',0.),('steady',0.),('steady',-.1),('steady',.1))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--summary-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() or args.summary_dir.exists():parser.error('Outputs must be new')
    args.output_dir.mkdir(parents=True);args.summary_dir.mkdir(parents=True)
    config=dict(duration_s=30.,forward_commands_mps=[.5,1.],policies=POLICIES,plans=PLANS,
        terrain_seeds=[['plane',42],['rough',42],['rough',43],['rough',44]],friction=.8,
        paired_modes=['off','on'],switch_authorized=False,labels='unlabeled physical rollouts')
    (args.summary_dir/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    recorder=Path(__file__).with_name('record_locomotion_command_plan.py')
    comparisons=[]
    for terrain,seed in config['terrain_seeds']:
        for policy in POLICIES:
            for speed in config['forward_commands_mps']:
                for mode,yaw in PLANS:
                    name=f'{policy}_{terrain}_{seed}_vx{speed}_{mode}_{yaw}'
                    p=args.output_dir/name;p.mkdir()
                    print(f'[{len(comparisons)+1}/80] {name}',flush=True)
                    for shadow in ('off','on'):
                        command=[sys.executable,str(recorder),'--reference-root',str(args.reference_root),
                            '--policy',str(args.reference_root/f'mujoco/policies/{policy}/policy_actor.npz'),
                            '--output-dir',str(p/shadow),'--label',policy,'--mode',mode,'--forward',str(speed),
                            '--yaw',str(yaw),'--terrain',terrain,'--seed',str(seed),'--duration','30']
                        if shadow=='on':command.append('--shadow')
                        with (p/f'{shadow}.log').open('w') as f:
                            subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                    verification=verify_pair(p/'off',p/'on')
                    a=json.loads((p/'off/provenance.json').read_text());b=json.loads((p/'on/provenance.json').read_text())
                    verification['pre_run_provenance_equal']=a==b
                    with np.load(p/'off/control_trace.npz') as x,np.load(p/'on/control_trace.npz') as y:
                        verification['observations_exactly_equal']=np.array_equal(x['observations'],y['observations'])
                        verification['policy_commands_exactly_equal']=np.array_equal(x['policy_commands'],y['policy_commands'])
                        with (p/'on/trajectory.csv').open() as f:rows=list(csv.DictReader(f))
                        logged=np.array([[float(r['command_forward_mps']),0.,float(r['command_yaw_rps'])]
                                         for r in rows if r['policy_update']=='True'])
                        verification['logged_commands_match_policy_inputs']=np.array_equal(logged,y['policy_commands'])
                    with np.load(p/'off/initial_state.npz') as x,np.load(p/'on/initial_state.npz') as y:
                        verification['initial_integration_state_equal']=np.array_equal(x['integration_state'],y['integration_state'])
                    extra=('pre_run_provenance_equal','observations_exactly_equal','policy_commands_exactly_equal',
                           'logged_commands_match_policy_inputs','initial_integration_state_equal')
                    if verification['status']!='PASS' or not all(verification[k] for k in extra):
                        (p/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
                        raise RuntimeError(f'Non-interference failed: {name}')
                    (p/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
                    m=json.loads((p/'on/metrics.json').read_text())
                    comparisons.append(dict(case=name,policy=policy,terrain=terrain,seed=seed,
                        forward_mps=speed,command_mode=mode,steady_yaw_rps=yaw,elapsed_s=m['elapsed_s'],fell=m['fell'],
                        mean_body_speed_x_mps=m['mean_body_speed_x_mps'],speed_rmse_mps=m['speed_tracking_rmse_mps'],
                        yaw_rmse_rps=m['yaw_rate_tracking_rmse_rps'],mean_tilt_deg=m['mean_tilt_deg'],
                        max_tilt_deg=m['max_tilt_deg'],pair_status=verification['status'],
                        hfield_sha256=b['compiled_model_arrays_sha256']['hfield_data'],
                        provenance_sha256=sha(p/'on/provenance.json'),verification_sha256=sha(p/'verification.json')))
                    # Incremental results remain useful if execution is interrupted.
                    with (args.summary_dir/'comparison.csv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(comparisons[0]),lineterminator='\n')
                        w.writeheader();w.writerows(comparisons)
                    print(f"  PASS; elapsed {m['elapsed_s']:.2f}s; fell={m['fell']}",flush=True)
    summary=dict(status='PASS',paired_case_count=len(comparisons),rollout_count=2*len(comparisons),
        completed_30s_cases=sum(r['elapsed_s']==30. for r in comparisons),
        fell_cases=sum(r['fell'] for r in comparisons),
        recorder_sha256=sha(recorder),runner_sha256=sha(Path(__file__).resolve()),
        scope='Physical MuJoCo rollouts, unchanged actors/PD; no reset, no recovery; '
              'PASS denotes monitoring non-interference, not good tracking or safety.')
    (args.summary_dir/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
