import json,csv,hashlib,shutil
from pathlib import Path
repo=Path('/home/omai/robotics/projects/g1-hierarchical-control');out=repo/'results/skills_sprint/20261003/handoff_diagnostic';raw=repo/'code/day9/g1_balance/checkpoints/skills_sprint/20261003/handoff_diagnostic'
groups=[];verifications={};rows=[]
for engine in ('mujoco','isaac'):
 d=out/engine
 if engine=='isaac':
  parts=[d/'hold',d/'march'];combined_groups=[];combined_checks={};combined_rows=[];combined_traces={}
  for part in parts:
   sub=json.loads((part/'summary.json').read_text());combined_groups.extend(sub['groups'])
   for k,v in sub['verification'].items():combined_checks[k]=max(combined_checks.get(k,0),v) if k=='max_actor_error' else combined_checks.get(k,0)+v
   with (part/'cases.csv').open() as f:combined_rows.extend(csv.DictReader(f))
   combined_traces.update(json.loads((part/'trace_hashes.json').read_text()));shutil.copytree(part/'evidence',d/'evidence',dirs_exist_ok=True)
  (d/'summary.json').write_text(json.dumps(dict(groups=combined_groups,verification=combined_checks),indent=2))
  (d/'trace_hashes.json').write_text(json.dumps(combined_traces,indent=2))
  with (d/'cases.csv').open('w') as f:
   fields=list(dict.fromkeys(k for r in combined_rows for k in r));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(combined_rows)
 s=json.loads((d/'summary.json').read_text());groups.extend(s['groups']);verifications[engine]=s['verification']
 with (d/'cases.csv').open() as f:
  for r in csv.DictReader(f):
   if 'decision_net_m' not in r:
    complete=json.loads((d/f'evidence/{engine}_{r["skill"]}/case_{int(r["case"]):02d}/complete.json').read_text());record=complete['results'][int(r['env'])] if engine=='isaac' else complete
    if 'decision_metrics' in record:
     m=record['decision_metrics'];r.update(decision_net_m=m['net_displacement_m'],decision_heading_excursion_rad=m['max_heading_excursion_rad'],whole_decision_pass=bool(r['full']=='True' and max(m['max_abs_roll_deg'],m['max_abs_pitch_deg'])<=20 and m['max_heading_excursion_rad']<=.2 and (m['eligible_through_end_after_confirmation'] and m['net_displacement_m']<=.5 if r['skill']=='hold' else r['cycles_and_lift']=='True' and m['net_displacement_m']<=.2)))
   rows.append(r)
for g in groups:
 g['whole_decision_pass']=sum(str(r.get('whole_decision_pass'))=='True' for r in rows if r['engine']==g['engine'] and r['skill']==g['skill'] and r['protocol']==g['protocol'] and float(r['vx'])==g['vx'])
(out/'summary.json').write_text(json.dumps(dict(groups=groups,verification=verifications,total_cases=len(rows),limits='Same seed42; parallel environments and phases are not independent seeds. Cold-start is not steady standing. Ramp changes speed, stance and history together. No training or skill promotion.'),indent=2))
with (out/'cases.csv').open('w') as f:
 fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
for n in ('run_plan.json','status.json','runtime_isaac.json','runtime_mujoco.json'):shutil.copyfile(raw/n,out/n)
source=out/'source';source.mkdir(exist_ok=True)
for p in list((repo/'scripts').glob('*skill_handoff*.py'))+[repo/'scripts/summarize_skill_motion.py',repo/'tests/test_skill_handoff_protocol.py',repo/'notes/skills_sprint/20261003_handoff_diagnostic.md']:
 target=source/p.relative_to(repo);target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,target)
print(json.dumps(groups,indent=2))
