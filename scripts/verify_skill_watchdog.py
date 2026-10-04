"""Offline actor/action, observation, gate and metric reconstruction."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from verify_skill_heading import actor,check
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.hierarchy.skill_metrics import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.root/'plan.json').read_text());check(not plan['smoke'],'Not formal evaluation')
    for path,digest in json.loads((a.root/'inputs.json').read_text()).items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Changed input '+path)
    parent=a.root.parents[1];walk=parent/'pretraining_snapshot/assets/policy_actor.npz';hold=parent/'teacher_handoff_round/train_hold/export/policy_actor.npz'
    meta=json.loads(Path('/home/omai/robotics/reference_projects/g1_walk_isaaclab_mujoco/mujoco/policies/isaac_metadata.json').read_text());defaults=np.array(meta['default_joint_pos'])
    rows=[];traces=[];count=0;maxerr=0.;hashes={}
    for folder in sorted(a.root.glob('case_*')):
        result=json.loads((folder/'complete.json').read_text());z=dict(np.load(folder/'trace.npz'));traces.append((result,z));hashes[str(folder/'trace.npz')]=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest()
        for j,r in enumerate(result['results']):
            last_message=None;stale=False;fresh=0;expected_events=[]
            for now in z['time']:
                missing=now>=r['drop_start_s']-1e-8 and (r['protocol']=='persistent' or now<r['drop_start_s']+1.-1e-8)
                if round(now/.02)%5==0 and not missing:
                    last_message=float(now)
                    if stale:
                        fresh+=1
                        if fresh>=2:stale=False;expected_events.append(dict(time_s=float(now),event='fresh_two_messages'))
                elif stale and last_message is not None and now-last_message>.3+1e-8:fresh=0
                if (float('inf') if last_message is None else now-last_message)>.3+1e-8 and not stale:
                    stale=True;fresh=0;expected_events.append(dict(time_s=float(now),event='stale'))
            actual_events=[e for e in r['watchdog_events'] if e['event']!='resume_blocked_stale']
            check(actual_events==expected_events,'Watchdog event mismatch')
            check(abs(r['decision_s']-next(e['time_s'] for e in expected_events if e['event']=='stale'))<1e-8,'Stale trigger mismatch')
            if r['protocol']=='persistent':check(r['resume_start_s'] is None,'Stale unauthorized resume')
            if r['resume_start_s'] is not None:check(any(e['event']=='fresh_two_messages' and e['time_s']<=r['resume_start_s'] for e in expected_events),'Resume before fresh requests')
            stages=z['stage'][:,j];terminal=np.flatnonzero(np.isin(stages,[4,5]));end=int(terminal[0])+1 if len(terminal) else len(stages);n=min(end-1,len(z['actions']));bad=np.flatnonzero(~z['valid'][:,j]);n=min(n,int(bad[0])) if len(bad) else n
            base=z['base_observations'][:n,j];s=z['signals'][:n,j];state=z['states'][:n,j];acts=z['actions'][:n,j];props=z['proposals'][:n,j];t=z['time'][:n];stage=stages[:n]
            qw,qx,qy,qz=s[:,3:7].T;gravity=np.stack((2*(qw*qy-qx*qz),-2*(qw*qx+qy*qz),-1+2*(qx*qx+qy*qy)),axis=1)
            np.testing.assert_allclose(base[:,:6],s[:,13:19],atol=3e-6,rtol=1e-6);np.testing.assert_allclose(base[:,6:9],gravity,atol=3e-6,rtol=0);np.testing.assert_allclose(base[:,12:49],state[:,13:50]-defaults,atol=3e-6,rtol=1e-6);np.testing.assert_allclose(base[:,49:86],state[:,50:87],atol=3e-6,rtol=1e-6)
            np.testing.assert_allclose(base[1:,-37:],acts[:-1],atol=2e-6,rtol=0);np.testing.assert_array_equal(base[0,-37:],np.zeros(37))
            desired=np.zeros((n,3));m=stage==0;desired[m,0]=.5;desired[m,2]=np.where(t[m]<2.4-1e-8,0,result['yaw']);m=stage==1;desired[m]=np.array([.5,0,result['yaw']])*np.maximum(0,1-(t[m]-r['decision_s'])/1.5)[:,None];desired[stage==3,0]=.5
            np.testing.assert_allclose(base[:,9:12],desired,atol=2e-6,rtol=0)
            expected=np.zeros_like(props)
            for mask,path in ((stage==2,hold),(stage!=2,walk)):
                if mask.any():expected[mask]=actor(path,base[mask])
            err=float(np.max(abs(expected-props)));maxerr=max(maxerr,err);check(err<1e-4,'Actor mismatch')
            blended=props.copy();expected_weights=np.ones(n)
            for phase,start in ((2,r['hold_start_s']),(3,r['resume_start_s'])):
                mask=stage==phase
                if not mask.any():continue
                i=int(np.flatnonzero(mask)[0]);u=np.clip((t[mask]-start)/.5,0,1);w=np.ones(sum(mask)) if r['protocol'] in ('temporary','persistent') else u*u*(3-2*u);expected_weights[mask]=w;blended[mask]=acts[i-1]*(1-w[:,None])+props[mask]*w[:,None]
            np.testing.assert_allclose(z['weights'][:n,j],expected_weights,atol=1e-9);np.testing.assert_allclose(acts,blended,atol=2e-5,rtol=1e-6);count+=n
            # Independent causal gate reconstruction from recorded physical signals.
            all_s=z['signals'][:end,j];time=z['time'][:end];hi=round(r['hold_start_s']/.02);rates=np.r_[np.nan,np.diff(all_s[:,20])/.02];stable=None;first_resume=None;first_ready=None
            for k in range(hi,end):
                start=max(hi,k-25);v=all_s[start:k+1];w,x,y,zz=all_s[k,3:7];tilt=max(abs(np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y))),abs(np.arcsin(np.clip(2*(w*y-zz*x),-1,1))))
                ready=len(v)==26 and np.linalg.norm(v[:,13:15],axis=1).mean()<=.05 and np.sqrt(np.mean(rates[start:k+1]**2))<=.05 and tilt<=np.deg2rad(20) and z['feet_contact'][k,j].all()
                if ready:
                    if stable is None:stable=time[k]
                else:stable=None
                if time[k]-r['hold_start_s']>=2.-1e-8 and stable is not None and time[k]-stable>=3.-1e-8 and z['valid'][k,j]:first_ready=float(time[k]);first_resume=first_ready if r['protocol']=='temporary' else None;break
                if time[k]-r['hold_start_s']>=12.-1e-8 or not z['valid'][k,j]:break
            check(first_resume==r['resume_start_s'],'Gate mismatch')
            for name,switch,stop in [('hold_metrics',r['hold_start_s'],round(r['resume_start_s']/.02)+1 if first_resume is not None else end),('resume_metrics',first_resume,end)]:
                if name not in r:continue
                score=metrics(dict(time=time[:stop],signals=all_s[:stop]),switch)
                for key,v in score.items():
                    if v is None or isinstance(v,bool):check(r[name][key]==v,'Metric '+key)
                    else:np.testing.assert_allclose(r[name][key],v,atol=1e-9,rtol=1e-9,equal_nan=True)
            complete=stages[end-1]==4;check(complete==r['sequence_completed'],'Completion mismatch');passed=False
            if 'resume_metrics' in r:
                tail=time>=time[-1]-2.-1e-8;rmse=float(np.sqrt(np.mean((all_s[tail,13]-.5)**2)));np.testing.assert_allclose(rmse,r['resume_tail_vx_rmse_mps'],atol=1e-9);rs=r['resume_metrics'];passed=complete and rmse<=.2 and rs['final_2s_euler_yaw_rate_rms_rps']<=.15 and max(rs['max_abs_roll_deg'],rs['max_abs_pitch_deg'])<=20
            check(bool(passed)==r['sequence_passed'],'Pass mismatch')
            row={k:r[k] for k in ('env','protocol','yaw','decision_s','hold_start_s','resume_start_s','sequence_completed','sequence_passed','failure_s')}
            row['failure_s']=r['failure_s'] if r['failure_s'] is not None and r['failure_s']<=time[-1]+1e-8 else None
            row['first_ready_s']=first_ready
            row['persistent_hold_passed']=bool(r['protocol']=='persistent' and r['failure_s'] is None and 'hold_metrics' in r and r['hold_metrics']['post_duration_s']>=12.-1e-8 and r['hold_metrics']['eligible_through_end_after_confirmation'] and max(r['hold_metrics']['max_abs_roll_deg'],r['hold_metrics']['max_abs_pitch_deg'])<=20)
            row['hold_latency_s']=first_resume-r['hold_start_s'] if first_resume is not None else None
            for key in ('decision_to_hold_exit_path_m','decision_to_hold_exit_net_m','resume_tail_vx_rmse_mps'):row[key]=r.get(key)
            row['resume_tail_yaw_rms_rps']=r.get('resume_metrics',{}).get('final_2s_euler_yaw_rate_rms_rps');rows.append(row)
    pairs=0
    for result,z in traces:
        if result['protocol']=='temporary':continue
        ref=next(v for rr,v in traces if rr['protocol'] in ('temporary','persistent') and rr['yaw']==result['yaw'])
        for j in range(plan['num_envs']):
            stop=round(result['results'][j]['decision_s']/.02)
            for key in ('states','signals','base_observations','actions','proposals'):
                extra=int(key in ('states','signals'));np.testing.assert_array_equal(z[key][:stop+extra,j],ref[key][:stop+extra,j])
            pairs+=1
    groups=[]
    for protocol in plan['protocols']:
        rr=[r for r in rows if r['protocol']==protocol];resumed=[r for r in rr if r['resume_start_s'] is not None];completed=[r for r in rr if r['sequence_completed']]
        groups.append(dict(protocol=protocol,total=len(rr),resumed=len(resumed),completed=len(completed),passed=sum(r['sequence_passed'] for r in rr),persistent_hold_passed=sum(r['persistent_hold_passed'] for r in rr),physical_failures=sum(r['failure_s'] is not None for r in rr),hold_latency_median_s=float(np.median([r['hold_latency_s'] for r in resumed])) if resumed else None,resume_vx_rmse_median_mps=float(np.median([r['resume_tail_vx_rmse_mps'] for r in completed])) if completed else None))
    with (a.output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps(dict(groups=groups,verification=dict(cases=len(rows),action_rows=count,max_actor_error=maxerr,paired_prefixes=pairs),limits='Low-speed flat Isaac seed42. Prospective sequence acceptance differs from original standalone hold acceptance. Task heartbeat fault injection, not actual perception dropout; no hardware or broad fallback validation.'),indent=2))
    (a.output/'trace_hashes.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(groups,indent=2));print('VERIFIED',len(rows),count,maxerr,pairs)
if __name__=='__main__':main()
