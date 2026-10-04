"""Record reference control with explicit command plans and passive shadow.

Only the experiment command is changed. Original observation assembly,
actor, action scaling, PD and termination remain in the reference run_once.
The diagnostic monitor never provides command or controller inputs.
"""
import argparse,csv,hashlib,json,os,platform,subprocess,sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from g1_control.monitoring.command_plan import CommandPlan
from g1_control.monitoring.locomotion_shadow import LocomotionShadow,ShadowSample


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--mode',choices=('steady','slow_sine','steps'),default='steady')
    parser.add_argument('--forward',type=float,default=1.)
    parser.add_argument('--yaw',type=float,default=0.)
    parser.add_argument('--terrain',choices=('plane','rough'),default='plane')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--duration',type=float,default=30.)
    parser.add_argument('--shadow',action='store_true')
    args=parser.parse_args()
    plan=CommandPlan(args.mode,args.forward,args.yaw)
    if not (0 < args.duration < float('inf')):
        parser.error('Duration must be finite and positive')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MUJOCO_GL','egl')
    sys.path.insert(0,str(args.reference_root/'mujoco'))
    import numpy as np
    import mujoco
    from mujoco_eval import run_grid
    model_path=args.reference_root/'mujoco/assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml'
    metadata=args.reference_root/'mujoco/policies/isaac_metadata.json'
    root=ET.parse(model_path).getroot()
    meshdir=root.find('compiler').get('meshdir','')
    inputs=[model_path,metadata,args.policy.resolve()]+[
        model_path.parent/meshdir/m.get('file') for m in root.findall('asset/mesh') if m.get('file')]
    reference_sources=[args.reference_root/'mujoco/mujoco_eval'/n for n in ('run_grid.py','policy.py','terrains.py')]
    source_hashes={str(p.relative_to(args.reference_root)):sha(p) for p in inputs+reference_sources}
    provenance=dict(capture_phase='before first policy action and physics step',
        reference_inputs_sha256=source_hashes,
        reference_head=subprocess.check_output(['git','-C',str(args.reference_root),'rev-parse','HEAD'],text=True).strip(),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,mujoco=mujoco.__version__,
                     OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
                     OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')),
        recorder_sha256=sha(Path(__file__).resolve()),
        command_plan_module_sha256=sha(REPO/'src/g1_control/monitoring/command_plan.py'),
        monitor_module_sha256=sha(REPO/'src/g1_control/monitoring/locomotion_shadow.py'),
        command_plan=asdict(plan),terrain=args.terrain,friction=.8,seed=args.seed,
        duration_s=args.duration,physics_timestep_s=.001,policy_timestep_s=.02,
        control_mode='pd',initial_base_height_m=.74,minimum_base_height_m=.35)
    cfg=SimpleNamespace(model=str(model_path),policy=str(args.policy.resolve()),metadata=str(metadata),
        command_x=args.forward,command_y=0.,command_yaw=args.yaw,timestep=.001,duration=args.duration,
        seed=args.seed,device='cpu',allow_missing_joints=False,initial_base_height=.74,
        min_base_height=.35,base_body='pelvis',control_mode='pd')
    metadata_values=json.loads(metadata.read_text())
    decimation=max(1,int(round(metadata_values['sim_dt']*metadata_values['decimation']/cfg.timestep)))
    provenance['policy_timestep_s']=decimation*cfg.timestep
    monitor=LocomotionShadow() if args.shadow else None
    rows,signals,states,observations,actions,commands=[],[],[],[],[],[]
    digest=hashlib.sha256()
    step_count=0
    initial_captured=False
    actual_model=actual_data=sample_data=None
    original_forward=mujoco.mj_forward
    original_step=mujoco.mj_step
    original_obs=run_grid.build_policy_observation
    original_act=run_grid.TorchActorPolicy.act
    original_scene=run_grid.write_scene_xml

    def scene(*values,**kwargs):
        result=original_scene(*values,**kwargs)
        snapshots={}
        for p in sorted(Path(result).parent.glob('*.xml')):
            # Only snapshot copies normalize temporary include paths.
            text=p.read_text().replace(str(p.parent)+'/', '')
            (args.output_dir/p.name).write_text(text)
            snapshots[p.name]=hashlib.sha256(text.encode()).hexdigest()
        provenance['generated_scene_sha256']=snapshots
        return result

    def forward(model,data,*values,**kwargs):
        nonlocal initial_captured,actual_model,actual_data
        original_forward(model,data,*values,**kwargs)
        if not initial_captured:
            actual_model,actual_data=model,data
            spec=mujoco.mjtState.mjSTATE_INTEGRATION
            integration=np.empty(mujoco.mj_stateSize(model,spec))
            mujoco.mj_getState(model,data,integration,spec)
            np.savez_compressed(args.output_dir/'initial_state.npz',integration_state=integration,
                                qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy())
            provenance['initial_state_sha256']=hashlib.sha256(integration.tobytes()).hexdigest()
            provenance['initial_state_enum']='mjSTATE_INTEGRATION'
            provenance['initial_simulation_time_s']=float(data.time)
            provenance['compiled_model_arrays_sha256']={name:hashlib.sha256(getattr(model,name).tobytes()).hexdigest()
                for name in ('qpos0','body_mass','body_inertia','dof_armature','geom_friction',
                             'hfield_data','actuator_ctrlrange')}
            provenance['model_dimensions']=dict(nq=model.nq,nv=model.nv,nu=model.nu)
            (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            initial_captured=True

    def sample(policy_update):
        nonlocal sample_data
        model,data=actual_model,actual_data
        if sample_data is None:
            sample_data=mujoco.MjData(model)
        sample_data.qpos[:]=data.qpos
        sample_data.qvel[:]=data.qvel
        # Use the original function: the separate sample cannot recapture state.
        original_forward(model,sample_data)
        quat,velocity,omega,height=run_grid.read_base_state(model,sample_data,'pelvis')
        body_v=run_grid.quat_rotate_inverse(quat,velocity)
        body_w=run_grid.quat_rotate_inverse(quat,omega)
        gravity=run_grid.quat_rotate_inverse(quat,np.array([0.,0.,-1.]))
        tilt=float(np.degrees(np.arccos(np.clip(-gravity[2],-1.,1.))))
        w,x,y,z=quat
        command,epoch=plan.at(step_count*cfg.timestep)
        states.append(np.concatenate((data.qpos,data.qvel)).copy())
        rows.append(dict(time_s=float(data.time),world_x_m=float(data.qpos[0]),world_y_m=float(data.qpos[1]),
            yaw_rad=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))),base_height_m=height,
            body_forward_speed_mps=float(body_v[0]),body_yaw_rate_rps=float(body_w[2]),tilt_deg=tilt,
            angular_speed_rps=float(np.linalg.norm(omega)),command_forward_mps=command[0],
            command_yaw_rps=command[2],command_epoch=epoch,policy_update=policy_update))
        if monitor is not None:
            signals.append(monitor.observe(ShadowSample(float(data.time),tilt,float(np.linalg.norm(omega)),
                float(body_v[0]),float(body_w[2]),command[0],command[2])))

    def observation(quat,lin_vel,ang_vel,command,*values,**kwargs):
        # The command plan is an experiment input, independent of shadow state.
        current,_=plan.at(step_count*cfg.timestep)
        command[:]=current
        cfg.command_x,cfg.command_y,cfg.command_yaw=current
        obs=original_obs(quat,lin_vel,ang_vel,command,*values,**kwargs)
        sample(True)
        commands.append(np.array(current))
        observations.append(obs.copy())
        return obs

    def act(policy,obs):
        action=original_act(policy,obs)
        actions.append(action.copy())
        return action

    def step(model,data,*values,**kwargs):
        nonlocal step_count
        original_step(model,data,*values,**kwargs)
        for array in (data.qpos,data.qvel,data.ctrl):
            digest.update(array.tobytes())
        step_count+=1

    with patch.object(run_grid,'write_scene_xml',scene),patch.object(mujoco,'mj_forward',forward),\
         patch.object(run_grid,'build_policy_observation',observation),\
         patch.object(run_grid.TorchActorPolicy,'act',act),patch.object(mujoco,'mj_step',step):
        metrics=run_grid.run_once(cfg,args.terrain,.8)
    if not initial_captured:
        raise RuntimeError('No initial snapshot captured')
    if not rows or actual_data.time > rows[-1]['time_s']+1e-10:
        sample(False)
    for name,before in source_hashes.items():
        if sha(args.reference_root/name)!=before:
            raise RuntimeError(f'Reference input changed during rollout: {name}')
    trace=dict(sampled_states=np.asarray(states),observations=np.asarray(observations),
               actions=np.asarray(actions),policy_commands=np.asarray(commands))
    if not np.array_equal(trace['observations'][:,9:12],trace['policy_commands'].astype(np.float32)):
        raise RuntimeError('Policy observation commands differ from the experiment schedule')
    np.savez_compressed(args.output_dir/'control_trace.npz',**trace)
    write_csv(args.output_dir/'trajectory.csv',rows)
    if monitor is not None:
        write_csv(args.output_dir/'shadow_signals.csv',[{**r,'candidate_reasons':json.dumps(r['candidate_reasons'])} for r in signals])
        (args.output_dir/'shadow_events.json').write_text(json.dumps(dict(mode='shadow_only',
            thresholds=asdict(monitor.thresholds),events=monitor.events,
            interpretation='Exploratory diagnostic candidates; no switching or recovery authority.'),indent=2)+'\n')
    metrics.update(model=model_path.name,policy=args.label,metadata=metadata.name,
        policy_sha256=sha(args.policy),metadata_sha256=sha(metadata),reference_head=provenance['reference_head'],
        shadow_enabled=args.shadow,physics_steps=step_count,physics_state_control_sha256=digest.hexdigest(),
        command_plan=asdict(plan),metric_command_definition='Scheduled commands at each policy update; final command fields are last applied values.',
        rollout_config={k:provenance[k] for k in ('seed','physics_timestep_s','policy_timestep_s',
                                                 'control_mode','initial_base_height_m','minimum_base_height_m')},
        recording_frames=0,termination_definition='Reference policy-step base-height check <0.35 m; no reset')
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(dict(elapsed_s=metrics['elapsed_s'],fell=metrics['fell'],actions=len(actions),
                          samples=len(states),terrain=args.terrain),indent=2))


if __name__=='__main__':
    main()
