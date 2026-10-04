"""Matched-command physical replay: original versus USD-derived position-servo model."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root",type=Path,required=True);parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_root/"mujoco"))
    from mujoco_eval.policy import TorchActorPolicy,build_policy_observation,quat_rotate_inverse
    from mujoco_eval.run_grid import resolve_joint_id,pd_gains,read_base_state
    metadata_path=args.reference_root/"mujoco/policies/isaac_metadata.json";meta=json.loads(metadata_path.read_text())
    files=[metadata_path,args.policy,Path(__file__).resolve(),args.reference_root/"mujoco/mujoco_eval/policy.py",args.reference_root/"mujoco/mujoco_eval/run_grid.py",
        args.run_root/"visual_mujoco/baseline1499.mjb",args.run_root/"usd_physical_models/g1_usd_physical_position.mjb"]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/"inputs.json").write_text(json.dumps(hashes,indent=2))
    actor=TorchActorPolicy(args.policy);rows=[]
    for profile,file in (("original",files[-2]),("usd_position",files[-1])):
        for speed in (.5,1.):
            for yaw in (0.,-.2,.2):
                model=mujoco.MjModel.from_binary_path(str(file));model.opt.timestep=.001
                data=mujoco.MjData(model)
                ids=np.array([resolve_joint_id(model,name)[0] for name in meta["joint_names"]]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
                ai=np.array([next(i for i in range(model.nu) if model.actuator_trnid[i,0]==jid) for jid in ids])
                defaults=np.array(meta["default_joint_pos"]);kp,kd=pd_gains(meta["joint_names"])
                data.qpos[:]=model.qpos0;data.qpos[:3]=[0,0,.74];data.qpos[3:7]=[1,0,0,0];data.qpos[qa]=defaults
                last=np.zeros(37);target=defaults.copy();command=np.array([speed,0,yaw]);fell=False;numerical=False
                times=[];signals=[];states=[];actions=[];force_max=np.zeros(37);physical_steps=0
                # A fresh mj_forward at each policy boundary keeps all observations at the same timestamp.
                # This is shared by both profiles and differs from older reference cached-field sampling.
                for step in range(30000+1):
                    if step%20==0:
                        mujoco.mj_forward(model,data)
                        quat,lin,ang,height=read_base_state(model,data,"pelvis")
                        bodylin=quat_rotate_inverse(quat,lin);bodyang=quat_rotate_inverse(quat,ang)
                        gravity=quat_rotate_inverse(quat,np.array([0.,0.,-1.]));tilt=float(np.arccos(np.clip(-gravity[2],-1,1)))
                        times.append(float(data.time));states.append(np.r_[data.qpos,data.qvel].copy())
                        signals.append(np.r_[data.qpos[:3],quat,lin,ang,bodylin,bodyang,height,tilt,command])
                        if height<.35:fell=True;break
                        if step==30000:break
                        obs=build_policy_observation(quat,lin,ang,command,data.qpos[qa],data.qvel[va],defaults,last)
                        if obs.shape!=(123,) or not np.array_equal(obs[9:12],command.astype(np.float32)):raise RuntimeError("Policy interface differs")
                        last=actor.act(obs).astype(float);actions.append(last.copy());target=defaults+meta["action_scale"]*last
                    if profile=="usd_position":data.ctrl[ai]=target
                    else:
                        ctrl=kp*(target-data.qpos[qa])-kd*data.qvel[va]
                        limited=model.actuator_ctrllimited[ai]
                        ctrl[limited]=np.clip(ctrl[limited],model.actuator_ctrlrange[ai[limited],0],model.actuator_ctrlrange[ai[limited],1])
                        data.ctrl[ai]=ctrl
                    mujoco.mj_step(model,data);physical_steps+=1;force_max=np.maximum(force_max,np.abs(data.qfrc_actuator[va]))
                    if not np.all(np.isfinite(data.qpos)) or any(data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
                        numerical=True;break
                signal=np.array(signals);time=np.array(times);mask=time>=2
                score=signal[mask] if mask.any() else signal
                # Signals: pos3,quat4,worldlin3,worldang3,bodylin3,bodyang3,height,tilt,command3.
                row=dict(profile=profile,speed=speed,yaw_command=yaw,elapsed_s=physical_steps*.001,fell=int(fell),numerical_failure=int(numerical),
                    scored_samples=int(len(score)),scored_from_s=2. if mask.any() else 0.,body_forward_rmse=float(np.sqrt(np.mean((score[:,13]-speed)**2))),
                    body_yaw_rmse=float(np.sqrt(np.mean((score[:,18]-yaw)**2))),world_yaw_rmse=float(np.sqrt(np.mean((score[:,12]-yaw)**2))),
                    mean_body_vx=float(score[:,13].mean()),mean_world_wz=float(score[:,12].mean()),max_tilt_deg=float(np.rad2deg(signal[:,20].max())),
                    distance_x_m=float(data.qpos[0]),lateral_drift_m=float(data.qpos[1]),physical_steps=physical_steps,actions=len(actions))
                stem=f"{profile}_vx{speed:g}_yaw{yaw:+.1f}"
                np.savez_compressed(args.output/f"{stem}.npz",time=time,states=states,signals=signals,actions=actions,force_max=force_max)
                (args.output/f"{stem}.json").write_text(json.dumps(row,indent=2));rows.append(row)
                print("PHYSICAL_REPLAY",profile,speed,yaw,row["elapsed_s"],row["fell"],row["body_forward_rmse"],row["world_yaw_rmse"],flush=True)
    for path,expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError("Source input changed")
    (args.output/"complete.json").write_text(json.dumps(dict(results=rows,python=sys.version,numpy=np.__version__,mujoco=mujoco.__version__,
        protocol=dict(seed=42,duration=30.,physics_dt=.001,policy_dt=.02,warmup_s=2.,start_yaw=0.,initial_height=.74,
            termination="pelvis height < 0.35m or numerical warning; no reset",observation_sampling="fresh mj_forward at policy boundaries for both profiles",actor="baseline1499 NPZ")),indent=2))


if __name__=="__main__":main()
