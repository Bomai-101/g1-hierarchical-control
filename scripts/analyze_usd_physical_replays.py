"""Independently reconstruct physical replay signals and compare matched cases."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True);parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.reference_root/"mujoco"))
    from mujoco_eval.policy import quat_rotate_inverse,TorchActorPolicy,build_policy_observation
    from mujoco_eval.run_grid import resolve_joint_id
    folder=args.run_root/"usd_physical_replays";complete=json.loads((folder/"complete.json").read_text())
    inputs=json.loads((folder/"inputs.json").read_text())
    for file,expected in inputs.items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected:raise RuntimeError("Replay input changed")
    metadata=json.loads((args.reference_root/"mujoco/policies/isaac_metadata.json").read_text());defaults=np.array(metadata["default_joint_pos"])
    policy_path=next(Path(p) for p in inputs if Path(p).name=="policy_actor.npz");actor=TorchActorPolicy(policy_path)
    rows=[];evidence=[]
    for row in complete["results"]:
        profile=row["profile"];speed=row["speed"];yaw=row["yaw_command"];stem=f"{profile}_vx{speed:g}_yaw{yaw:+.1f}"
        trace=np.load(folder/f"{stem}.npz");signal=trace["signals"];time=trace["time"]
        model_path=args.run_root/("visual_mujoco/baseline1499.mjb" if profile=="original" else "usd_physical_models/g1_usd_physical_position.mjb")
        model=mujoco.MjModel.from_binary_path(str(model_path));data=mujoco.MjData(model);pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
        joint_ids=np.array([resolve_joint_id(model,name)[0] for name in metadata["joint_names"]]);qa=model.jnt_qposadr[joint_ids];va=model.jnt_dofadr[joint_ids]
        command=np.array([speed,0,yaw]);rebuilt=[];actor_error=0.
        selected=set(np.linspace(0,len(trace["actions"])-1,32,dtype=int))
        for index,state in enumerate(trace["states"]):
            data.qpos[:]=state[:model.nq];data.qvel[:]=state[model.nq:];mujoco.mj_forward(model,data)
            quat=data.xquat[pelvis].copy();spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,spatial,0)
            ang,lin=spatial[:3],spatial[3:];bodylin=quat_rotate_inverse(quat,lin);bodyang=quat_rotate_inverse(quat,ang)
            gravity=quat_rotate_inverse(quat,[0,0,-1]);tilt=np.arccos(np.clip(-gravity[2],-1,1))
            rebuilt.append(np.r_[data.qpos[:3],quat,lin,ang,bodylin,bodyang,data.xpos[pelvis,2],tilt,command])
            if index in selected:
                previous=trace["actions"][index-1] if index else np.zeros(37)
                obs=build_policy_observation(quat,lin,ang,command,data.qpos[qa],data.qvel[va],defaults,previous)
                actor_error=max(actor_error,float(np.max(np.abs(actor.act(obs)-trace["actions"][index]))))
        error=float(np.max(np.abs(np.array(rebuilt)-signal)))
        if error>1e-10 or actor_error>1e-6:raise RuntimeError(f"Independent signals/actor differ: {error} {actor_error}")
        if not np.allclose(np.diff(time),.02,atol=1e-10) or abs(time[-1]-row["elapsed_s"])>1e-8:raise RuntimeError("Timeline differs")
        score=np.array(rebuilt)[time>=2]
        metrics=dict(body_forward_rmse=float(np.sqrt(np.mean((score[:,13]-speed)**2))),body_yaw_rmse=float(np.sqrt(np.mean((score[:,18]-yaw)**2))),
            world_yaw_rmse=float(np.sqrt(np.mean((score[:,12]-yaw)**2))),mean_body_vx=float(score[:,13].mean()),mean_world_wz=float(score[:,12].mean()),
            max_tilt_deg=float(np.rad2deg(signal[:,20].max())))
        for name,value in metrics.items():
            if abs(row[name]-value)>1e-10:raise RuntimeError(f"Metric differs: {name}")
        if trace['signals'].shape!=(1501,24) or len(trace['actions'])!=1500 or row['physical_steps']!=30000 or row['fell'] or row['numerical_failure']:
            raise RuntimeError("This completed experiment expected full 30-second valid episodes")
        if np.any(trace['force_max']>model.jnt_actfrcrange[joint_ids,1]+1e-7):raise RuntimeError("Applied force exceeds configured cap")
        rows.append(row);evidence.append(dict(case=stem,signal_max_abs_error=error,actor_max_abs_error=actor_error,states=len(time),actor_inputs_checked=32,force_limit_pass=True,
            trace_sha256=hashlib.sha256((folder/f"{stem}.npz").read_bytes()).hexdigest()))
    paired=[]
    for speed in (.5,1.):
        for yaw in (0.,-.2,.2):
            old=next(r for r in rows if r['profile']=='original' and r['speed']==speed and r['yaw_command']==yaw)
            new=next(r for r in rows if r['profile']=='usd_position' and r['speed']==speed and r['yaw_command']==yaw)
            paired.append(dict(speed=speed,yaw_command=yaw,original_forward_rmse=old['body_forward_rmse'],usd_forward_rmse=new['body_forward_rmse'],
                original_world_yaw_rmse=old['world_yaw_rmse'],usd_world_yaw_rmse=new['world_yaw_rmse'],
                original_mean_world_wz=old['mean_world_wz'],usd_mean_world_wz=new['mean_world_wz'],usd_elapsed_s=new['elapsed_s']))
    for file,data in [('summary.csv',rows),('comparison.csv',paired)]:
        with (args.output/file).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (args.output/'replay_verification.json').write_text(json.dumps(dict(cases=evidence,all12valid30s=True,no_policy_changes=True,
        policy_interface='Same fixed velocity commands; warm-up2s; current-state observations on both profiles',
        limitation='Original versus derived comparison changes multiple model/integrator/actuator/contact properties; not isolated causal attribution or validated robustness.'),indent=2))
    print('PASS: independently reconstructed12x1501 states,384 actor inputs, scores, force limits and source hashes')


if __name__=='__main__':main()
