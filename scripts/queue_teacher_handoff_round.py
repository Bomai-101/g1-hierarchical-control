"""Finite two-skill teacher-state round; stop on process/artifact/input failure."""
from pathlib import Path
import json,subprocess,hashlib,time,os
repo=Path(__file__).resolve().parents[1];old=repo/'code/day9/g1_balance/checkpoints/skills_sprint/20261003';root=old/'teacher_handoff_round';ref=Path('/home/omai/robotics/reference_projects/g1_walk_isaaclab_mujoco');bank=root/'teacher_bank/bank.npz';jobs=[]
for s in ('hold','march'):
 jobs.append(dict(name='smoke_'+s,command=['bash','scripts/run_flat_isaac.sh','scripts/train_teacher_handoff.py','--skill',s,'--reference-root',str(ref),'--checkpoint',str(root/f'pretraining_snapshot/{s}_model_999.pt'),'--bank',str(bank),'--output',str(root/f'smoke_{s}'),'--num-envs','16','--iters','2','--seed','44','--headless'],artifact=str(root/f'smoke_{s}/complete.json')))
for s in ('hold','march'):
 train=root/f'train_{s}';ckpt=train/'model_499.pt';export=train/'export'
 common=['--skill',s,'--reference-root',str(ref)]
 jobs.append(dict(name='train_'+s,command=['bash','scripts/run_flat_isaac.sh','scripts/train_teacher_handoff.py',*common,'--checkpoint',str(root/f'pretraining_snapshot/{s}_model_999.pt'),'--bank',str(bank),'--output',str(train),'--num-envs','4096','--iters','500','--seed','44','--headless'],artifact=str(train/'complete.json')))
 jobs.append(dict(name='export_'+s,command=['/home/omai/miniforge3/envs/g1-walk-isaac/bin/python','scripts/export_skill_sprint.py','--skill',s,'--checkpoint',str(ckpt),'--output',str(export)],artifact=str(export/'skill_metadata.json')))
 jobs.append(dict(name='isaac_'+s,command=['bash','scripts/run_flat_isaac.sh','scripts/evaluate_skill_sprint_isaac.py',*common,'--walk-checkpoint',str(old/'pretraining_snapshot/assets/model_1499.pt'),'--skill-checkpoint',str(ckpt),'--output',str(train/'isaac_eval'),'--num-envs','16','--headless'],artifact=str(train/'isaac_eval/complete.json')))
 jobs.append(dict(name='mujoco_'+s,command=[str(repo/'.venv/bin/python'),'scripts/evaluate_skill_sprint_mujoco.py',*common,'--walk-policy',str(old/'pretraining_snapshot/assets/policy_actor.npz'),'--skill-policy',str(export/'policy_actor.npz'),'--model',str(repo/'code/day9/g1_balance/checkpoints/flat_baseline/20261002/usd_physical_models/g1_usd_physical_position.mjb'),'--output',str(train/'mujoco_eval')],artifact=str(train/'mujoco_eval/complete.json')))
files=[repo/'scripts'/n for n in ('train_teacher_handoff.py','teacher_handoff_task.py','skills_sprint_task.py','export_skill_sprint.py','evaluate_skill_sprint_isaac.py','evaluate_skill_sprint_mujoco.py','queue_teacher_handoff_round.py')]+[repo/'src/g1_control/hierarchy/skill_adapter.py',repo/'src/g1_control/hierarchy/skill_metrics.py',bank,old/'pretraining_snapshot/assets/model_1499.pt',old/'pretraining_snapshot/assets/policy_actor.npz']+[root/f'pretraining_snapshot/{s}_model_999.pt' for s in ('hold','march')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in files};(root/'queue_plan.json').write_text(json.dumps(dict(jobs=jobs,source_hashes=hashes,scope='Exactly2smokes+2x500updates+export/native/Mu; unchanged first-round protocol; no additional training/deployment/git automatically.'),indent=2));status=dict(state='running',pid=os.getpid(),completed=[]);logs=root/'queue_logs';logs.mkdir(exist_ok=True)
def save():(root/'queue_status.json').write_text(json.dumps(status,indent=2))
try:
 for j in jobs:
  for p,h in hashes.items():
   if sha(Path(p))!=h:raise RuntimeError('Input changed '+p)
  status['active']=j['name'];save();print('START',j['name'],flush=True)
  with (logs/(j['name']+'.log')).open('w') as f:subprocess.run(j['command'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
  artifact=Path(j['artifact'])
  if not artifact.exists():raise RuntimeError('Missing '+str(artifact))
  status['completed'].append(dict(name=j['name'],artifact=str(artifact),sha256=sha(artifact)));save();print('DONE',j['name'],flush=True)
 status.update(state='completed',active=None);save()
except Exception as e:status.update(state='failed',error=str(e));save();raise
