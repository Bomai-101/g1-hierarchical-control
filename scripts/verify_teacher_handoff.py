"""Reconstruct diagnostic actor outputs and compare matched predecision traces."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from g1_control.hierarchy.skill_metrics import metrics
from skill_handoff_protocol import command_at


def check(condition,message):
    if not condition:raise RuntimeError(message)


def actor(path,obs):
    with np.load(path) as p:
        x=obs.astype(np.float32)
        for i in range(int(p['num_layers'])):
            x=x@p[f'weight_{i}'].T+p[f'bias_{i}']
            if i<int(p['num_layers'])-1:x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
        return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',choices=('isaac','mujoco'));p.add_argument('--skill',choices=('hold','march'));a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve();a.output.mkdir(exist_ok=False,parents=True);old=a.root.parent
    run=json.loads((a.root/'queue_plan.json').read_text());status=json.loads((a.root/'queue_status.json').read_text());check(status['state']=='completed' or a.engine is not None,'Run incomplete')
    for path,digest in run['source_hashes'].items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Input changed '+path)
    run['jobs']=[dict(j,name=j['name'].split('_')[0]+'_'+j['name'].split('_')[1],output=str(Path(j['artifact']).parent)) for j in run['jobs'] if j['name'].startswith(('isaac_','mujoco_'))]
    rows=[];checks=dict(prefix_pairs=0,old_direct_trace_pairs=0,reconstructed_action_rows=0,metrics_rows=0,max_actor_error=0.);archive=[]
    for job in run['jobs']:
        if a.engine and not job['name'].startswith(a.engine+'_'):continue
        if a.skill and not job['name'].endswith('_'+a.skill):continue
        check(any(j['name']==job['name'] for j in status['completed']),'Job incomplete '+job['name'])
        folder=Path(job['output']);complete=json.loads((folder/'complete.json').read_text());plan=json.loads((folder/'plan.json').read_text());engine,skill=job['name'].split('_');native=engine=='isaac';cases=[dict(c,protocol='direct',decision_s=c.get('switch_s',5.22),ramp_s=0.) for c in plan['cases']];loaded=[]
        for path,h in json.loads((folder/'inputs.json').read_text()).items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'Evaluation input changed '+path)
        metadata=json.loads((Path('/home/omai/robotics/reference_projects/g1_walk_isaaclab_mujoco/mujoco/policies/isaac_metadata.json')).read_text())
        defaults=np.array(metadata['default_joint_pos'])
        if not native:
            import mujoco
            model_path=next(Path(path) for path in json.loads((folder/'inputs.json').read_text()) if path.endswith('.mjb'));model=mujoco.MjModel.from_binary_path(str(model_path));ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in metadata['joint_names']]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
        for ci,c in enumerate(cases):
            casefolder=folder/f'case_{ci:02d}';result=json.loads((casefolder/'complete.json').read_text());z=dict(np.load(casefolder/'trace.npz'));loaded.append(z)
            t=z['time'][:-1];obs=z['base_observations'];actual=z['actions'];n=len(actual);envs=actual.shape[1] if native else 1
            for j in range(envs):
                n=len(actual)
                if native:
                    invalid=np.flatnonzero(~z['valid'][:,j])
                    if len(invalid):n=min(n,int(invalid[0]))
                t=z['time'][:n]
                base=obs[:n,j] if native else obs[:n];inp=z['skill_inputs'][:n,j] if native else z['skill_inputs'][:n];act=actual[:n,j] if native else actual[:n];prop=z['proposals'][:n,j] if native else z['proposals'][:n]
                decision=0. if c['protocol']=='static' else c['decision_s']+(j*.02 if native else 0.);switch=decision+c['ramp_s'];idx=round(switch/.02);after=t>=switch-1e-6
                branch=dict(c,decision_s=decision,switch_s=switch)
                signals=z['signals'][:n,j] if native else z['signals'][:n];state=z['states'][:n,j] if native else z['states'][:n];quat=signals[:,3:7];qw,qx,qy,qz=quat.T
                gravity=np.stack((2*(qw*qy-qx*qz),-2*(qw*qx+qy*qz),-1+2*(qx*qx+qy*qy)),axis=1)
                np.testing.assert_allclose(base[:,:6],signals[:,13:19],atol=3e-6,rtol=1e-6);np.testing.assert_allclose(base[:,6:9],gravity,atol=3e-6,rtol=0)
                q=state[:,13:50] if native else state[:,qa];qv=state[:,50:87] if native else state[:,model.nq+va]
                np.testing.assert_allclose(base[:,12:49],q-defaults,atol=3e-6,rtol=1e-6);np.testing.assert_allclose(base[:,49:86],qv,atol=3e-6,rtol=1e-6)
                expected_commands=np.array([command_at(now,branch) for now in t]);np.testing.assert_allclose(base[:,9:12],expected_commands,atol=2e-6,rtol=0)
                np.testing.assert_array_equal(inp[:,:123],base)
                if skill=='march':
                    offset=j*2*np.pi/envs if native else c['phase_offset'];phase=np.maximum(t-switch,0)*2*np.pi/.9+offset;desired=np.stack((np.sin(phase),np.cos(phase)),axis=1)*after[:,None];np.testing.assert_allclose(inp[:,123:],desired,atol=3e-5,rtol=0)
                expected=np.empty_like(prop)
                for mask,path in ((~after,old/'pretraining_snapshot/assets/policy_actor.npz'),(after,a.root/f'train_{skill}/export/policy_actor.npz')):
                    if mask.any():expected[mask]=actor(path,base[mask] if not mask is after else inp[mask])
                err=float(np.max(abs(expected-prop)));checks['max_actor_error']=max(checks['max_actor_error'],err);check(err<1e-4,f'Actor mismatch {job["name"]}/{ci}/{j}: {err}')
                blended=prop.copy()
                if c['protocol']!='static' and idx<n:
                    u=np.clip((t[idx:]-switch)/.5,0,1);w=u*u*(3-2*u);blended[idx:]=act[idx-1]*(1-w[:,None])+prop[idx:]*w[:,None]
                np.testing.assert_allclose(act,blended,atol=2e-5,rtol=1e-6)
                if n>1:np.testing.assert_allclose(base[1:,-37:],act[:-1],atol=2e-6,rtol=0)
                checks['reconstructed_action_rows']+=n
                r=result['results'][j] if native else result
                if 'post_duration_s' not in r:
                    rows.append(dict(engine=engine,skill=skill,case=ci,env=j,protocol=c['protocol'],vx=c['vx'],yaw=c['yaw'],full=False,passed=False,reason=r.get('reason','')));continue
                end=round(switch/.02)+501
                if native:
                    bad=np.flatnonzero(~z['valid'][:,j]);end=min(end,int(bad[0]) if len(bad) else end);sig=z['signals'][:end,j]
                else:end=len(z['time']);sig=z['signals']
                trace=dict(time=z['time'][:end],signals=sig);score=metrics(trace,switch)
                for k,v in score.items():
                    if v is None or isinstance(v,bool):check(r[k]==v,'Metric mismatch '+k)
                    else:np.testing.assert_allclose(r[k],v,atol=1e-9,rtol=1e-9,equal_nan=True)
                decision_score=metrics(trace,decision)
                for k,v in (decision_score.items() if 'decision_metrics' in r else []):
                    if v is None or isinstance(v,bool):check(r['decision_metrics'][k]==v,'Decision metric '+k)
                    else:np.testing.assert_allclose(r['decision_metrics'][k],v,atol=1e-9,rtol=1e-9,equal_nan=True)
                full=r['full_post_horizon'] if native else r['termination']=='observation_complete';checks['metrics_rows']+=1
                i=round(switch/.02);late=np.flatnonzero(z['time'][:end]>=z['time'][end-1]-2.-1e-9);late_angle=float(sig[-1,20]-sig[late[0],20]);entry_speed=float(np.linalg.norm(sig[i,13:15]));feet=z['feet_contact'][i:end,j] if native else z['feet_contact'][i:end];edges=np.diff(feet.astype(int),axis=0);losses=(edges<0).sum(axis=0);landings=(edges>0).sum(axis=0);lift=(z['feet_height'][i:end,j] if native else z['feet_height'][i:end]).max(axis=0)-.055
                np.testing.assert_array_equal(losses,r['foot_contact_losses']);np.testing.assert_array_equal(landings,r['foot_landings']);np.testing.assert_allclose(lift,r['max_lift_above_ankle_floor_m'],atol=1e-9)
                cycles=bool(np.all(losses>=5)&np.all(landings>=5)&np.all(lift>=.025));tilt=max(score['max_abs_roll_deg'],score['max_abs_pitch_deg']);passed=full and tilt<=20 and score['max_heading_excursion_rad']<=.2 and (score['eligible_through_end_after_confirmation'] and score['net_displacement_m']<=.5 if skill=='hold' else cycles and score['net_displacement_m']<=.2);check(bool(passed)==r['passed'],'Pass predicate mismatch')
                cut=min(i+100,end-1);initial=sig[i:cut+1];remaining=sig[cut:];initial_path=float(np.linalg.norm(np.diff(initial[:,:2],axis=0),axis=1).sum());remaining_path=float(np.linalg.norm(np.diff(remaining[:,:2],axis=0),axis=1).sum())
                check(abs(initial_path+remaining_path-score['path_after_switch_m'])<1e-6,'Path partition')
                rows.append(dict(initial_window_s=float(z['time'][cut]-switch),initial_path_m=initial_path,initial_net_m=float(np.linalg.norm(initial[-1,:2]-initial[0,:2])),initial_heading_rad=float(initial[-1,20]-initial[0,20]),remaining_duration_s=float(z['time'][end-1]-z['time'][cut]),remaining_path_m=remaining_path,remaining_net_m=float(np.linalg.norm(remaining[-1,:2]-remaining[0,:2])),remaining_heading_rad=float(remaining[-1,20]-remaining[0,20]),engine=engine,skill=skill,case=ci,env=j,protocol=c['protocol'],vx=c['vx'],yaw=c['yaw'],full=bool(full),passed=r['passed'],entry_speed_mps=entry_speed,tail_speed_rms=score['final_2s_horizontal_speed_rms_mps'],tail_yaw_rms=score['final_2s_euler_yaw_rate_rms_rps'],tail_signed_heading_rad=late_angle,settled=score['eligible_through_end_after_confirmation'],net_m=score['net_displacement_m'],heading_excursion_rad=score['max_heading_excursion_rad'],cycles_and_lift=cycles,decision_path_m=decision_score['path_after_switch_m'],decision_net_m=decision_score['net_displacement_m'],decision_heading_excursion_rad=decision_score['max_heading_excursion_rad'],whole_decision_pass=bool(full and max(decision_score['max_abs_roll_deg'],decision_score['max_abs_pitch_deg'])<=20 and decision_score['max_heading_excursion_rad']<=.2 and (decision_score['eligible_through_end_after_confirmation'] and decision_score['net_displacement_m']<=.5 if skill=='hold' else cycles and decision_score['net_displacement_m']<=.2)),post_duration_s=score['post_duration_s']))
        for ci,c in enumerate(cases):
            if c['protocol']=='static':continue
            direct=next(k for k,v in enumerate(cases) if v['protocol']=='direct' and v['vx']==c['vx'] and v['yaw']==c['yaw'] and v['decision_s']==c['decision_s'])
            if ci!=direct:
                # All environments share the first decision prefix; then test each later decision separately.
                for j in range(16 if native else 1):
                    stop=round((c['decision_s']+(j*.02 if native else 0))/.02)
                    for key in ('states','signals','base_observations','actions','proposals'):
                        x=loaded[direct][key][:stop+(key in ('states','signals'))];y=loaded[ci][key][:stop+(key in ('states','signals'))]
                        if native:x=x[:,j];y=y[:,j]
                        np.testing.assert_array_equal(x,y,err_msg=f'Prefix mismatch {job["name"]}/{ci}/{j}/{key}')
                    checks['prefix_pairs']+=1
            else:
                # Check the repeated direct branch against the unmodified first-round evaluation.
                prior=json.loads((old/f'train_{skill}/{engine}_eval/plan.json').read_text())['cases'];pc=next(k for k,v in enumerate(prior) if v['vx']==c['vx'] and v['yaw']==c['yaw'] and (native or abs(v['switch_s']-c['switch_s'])<1e-8));before=dict(np.load(old/f'train_{skill}/{engine}_eval/case_{pc:02d}/trace.npz'))
                for key in ('states','signals','base_observations','skill_inputs','actions','proposals'):
                    # Native max horizon identical for direct cases.
                    stop=round(c['decision_s']/.02)
                    if native:
                        for j in range(loaded[ci][key].shape[1]):
                            endpoint=stop+j+(key in ('states','signals'));np.testing.assert_array_equal(loaded[ci][key][:endpoint,j],before[key][:endpoint,j],err_msg=f'1499prefix {job["name"]}/{ci}/{j}/{key}')
                    else:
                        endpoint=stop+(key in ('states','signals'));np.testing.assert_array_equal(loaded[ci][key][:endpoint],before[key][:endpoint],err_msg=f'1499prefix {job["name"]}/{ci}/{key}')
                checks['old_direct_trace_pairs']+=1
        archive.extend([folder/'plan.json',folder/'inputs.json',folder/'complete.json']+[folder/f'case_{ci:02d}/complete.json' for ci in range(len(cases))])
    groups=[]
    for engine in ('isaac','mujoco'):
      for skill in ('hold','march'):
       for protocol in ('static','direct','ramp_0.5','ramp_1.5'):
        for vx in ([0.] if protocol=='static' else [.5,1.]):
         rr=[r for r in rows if r['engine']==engine and r['skill']==skill and r['protocol']==protocol and r['vx']==vx];survivors=[r for r in rr if r['full']]
         if not rr:continue
         groups.append(dict(engine=engine,skill=skill,protocol=protocol,vx=vx,n=len(rr),full=sum(r['full'] for r in rr),passed=sum(r['passed'] for r in rr),whole_decision_pass=sum(r.get('whole_decision_pass',False) for r in rr),tail_speed_pass=sum(r.get('tail_speed_rms',np.inf)<=.05 for r in survivors),settled=sum(r.get('settled',False) for r in survivors),cycles_and_lift=sum(r.get('cycles_and_lift',False) for r in survivors),entry_speed_median=float(np.median([r['entry_speed_mps'] for r in rr if 'entry_speed_mps' in r])) if any('entry_speed_mps' in r for r in rr) else None,tail_heading_range_rad=[min(r['tail_signed_heading_rad'] for r in survivors),max(r['tail_signed_heading_rad'] for r in survivors)] if survivors else None))
    with (a.output/'cases.csv').open('w') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(dict(groups=groups,verification=checks,limits='Seed42 evaluation, seed43 teacherbank/44 training. Candidate weights differ from first round;1499 prefix matched. Fixed2s window is diagnostic, not a validated skill-entry boundary. No deployment.'),indent=2))
    import shutil
    for file in archive:
        target=a.output/'evidence'/file.relative_to(a.root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    trace_hashes={str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (f for job in run['jobs'] if (not a.engine or job['name'].startswith(a.engine+'_')) and (not a.skill or job['name'].endswith('_'+a.skill)) for f in Path(job['output']).glob('case_*/trace.npz'))}
    (a.output/'trace_hashes.json').write_text(json.dumps(trace_hashes,indent=2))
    for name in ('queue_plan.json','queue_status.json'):shutil.copyfile(a.root/name,a.output/name)
    print(json.dumps(dict(groups=groups,verification=checks),indent=2))

if __name__=='__main__':main()
