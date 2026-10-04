"""Compare heading angle wrapping with the actual IsaacLab source function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--isaac-math',type=Path,required=True);p.add_argument('--evaluator',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def extract(path,name,context):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),context)
    return context[name]
reference=extract(a.isaac_math,'wrap_to_pi',dict(torch=torch));candidate=extract(a.evaluator,'wrap_pi',dict(np=np))
rng=np.random.default_rng(42);angles=np.r_[-3*np.pi,-np.pi,0,np.pi,3*np.pi,np.deg2rad([-358,358]),rng.uniform(-100,100,10000)]
expected=reference(torch.from_numpy(angles)).numpy();actual=candidate(angles);error=float(np.max(np.abs(expected-actual)))
if error>1e-12:raise RuntimeError('Angle wrap disagrees with reference')
if not np.allclose(actual[:7],[-np.pi,-np.pi,0,np.pi,np.pi,np.deg2rad(2),np.deg2rad(-2)],atol=1e-12):raise RuntimeError('Boundary convention differs')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(samples=len(angles),max_abs_error=error,boundary_angles_rad=angles[:7].tolist(),wrapped_angles_rad=actual[:7].tolist(),torch=torch.__version__,numpy=np.__version__,source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.isaac_math,a.evaluator,Path(__file__))}),indent=2));print('PASS: actual IsaacLab wrap_to_pi matches10007 samples,including±pi and±179degree crossing')
