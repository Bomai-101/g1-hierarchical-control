import subprocess,json,hashlib,time,sys
from pathlib import Path
repo=Path('/home/omai/robotics/projects/g1-hierarchical-control');old=repo/'code/day9/g1_balance/checkpoints/skills_sprint/20261003';root=old/'handoff_diagnostic';root.mkdir(exist_ok=False)
ref='/home/omai/robotics/reference_projects/g1_walk_isaaclab_mujoco';model=repo/'code/day9/g1_balance/checkpoints/flat_baseline/20261002/usd_physical_models/g1_usd_physical_position.mjb'
jobs=[]
for engine in ('mujoco','isaac'):
 for skill in ('hold','march'):
  out=root/(engine+'_'+skill)
  cmd=[str(repo/'.venv/bin/python'),'scripts/evaluate_skill_handoff_mujoco.py','--walk-policy',str(old/'pretraining_snapshot/assets/policy_actor.npz'),'--skill-policy',str(old/f'train_{skill}/export/policy_actor.npz'),'--model',str(model)] if engine=='mujoco' else ['bash','scripts/run_flat_isaac.sh','scripts/evaluate_skill_handoff_isaac.py','--walk-checkpoint',str(old/'pretraining_snapshot/assets/model_1499.pt'),'--skill-checkpoint',str(old/f'train_{skill}/model_999.pt'),'--num-envs','16','--headless']
  cmd+=['--skill',skill,'--reference-root',ref,'--output',str(out)]
  jobs.append(dict(name=engine+'_'+skill,command=cmd,output=str(out)))
sources=list((repo/'scripts').glob('*skill_handoff*.py'))+[repo/'src/g1_control/hierarchy/skill_metrics.py',repo/'src/g1_control/hierarchy/skill_adapter.py',repo/'scripts/skills_sprint_task.py',model,old/'pretraining_snapshot/assets/model_1499.pt',old/'pretraining_snapshot/assets/policy_actor.npz']
for skill in ('hold','march'):sources.extend([old/f'train_{skill}/model_999.pt',old/f'train_{skill}/export/policy_actor.npz'])
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
(root/'run_plan.json').write_text(json.dumps(dict(jobs=jobs,input_hashes=hashes,scope='Fixed weights; no training; no promotion; static is cold-start zero velocity, not a validated steady-state standing posture; ramps use1499 until delayed switch;10s after each actual switch; decision metrics include braking'),indent=2))
status=dict(state='running',completed=[])
def save(): (root/'status.json').write_text(json.dumps(status,indent=2))
for j in jobs:
 status['active']=j['name'];save();print('START',j['name'],flush=True)
 with (root/(j['name']+'.log')).open('w') as f:
  try:subprocess.run(j['command'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
  except Exception as e:status.update(state='failed',error=str(e));save();raise
 artifact=Path(j['output'])/'complete.json'
 if not artifact.exists():raise RuntimeError('Missing complete')
 status['completed'].append(dict(name=j['name'],sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()));save();print('DONE',j['name'],flush=True)
for p,h in hashes.items():
 if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Input changed '+p)
status.update(state='completed',active=None);save();print('ALL_DONE',flush=True)
