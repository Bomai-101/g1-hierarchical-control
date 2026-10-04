"""Export candidate actor and check torch/NumPy agreement, no reference writes."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def forward(obs,weights,biases):
    x=np.asarray(obs,dtype=np.float32)
    for i,(w,b) in enumerate(zip(weights,biases)):
        x=x@w.T+b
        if i<len(weights)-1:x=np.maximum(x,0)+np.expm1(np.minimum(x,0))
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--skill',choices=('hold','march'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model_state_dict'];keys=sorted([k for k in state if k.startswith('actor.') and k.endswith('.weight')],key=lambda k:int(k.split('.')[1]));weights=[state[k].numpy() for k in keys];biases=[state[k.replace('.weight','.bias')].numpy() for k in keys];dim=123 if a.skill=='hold' else 125
    if weights[0].shape[1]!=dim or weights[-1].shape[0]!=37:raise RuntimeError('Skill schema')
    payload=dict(activation=np.array('elu'),num_layers=np.array(len(weights)))
    for i,(w,b) in enumerate(zip(weights,biases)):payload['weight_'+str(i)]=w;payload['bias_'+str(i)]=b
    path=a.output/'policy_actor.npz';np.savez(path,**payload)
    obs=np.random.default_rng(42).normal(size=(256,dim)).astype(np.float32);x=torch.from_numpy(obs)
    for i,(w,b) in enumerate(zip(weights,biases)):
        x=torch.nn.functional.linear(x,torch.from_numpy(w),torch.from_numpy(b))
        if i<len(weights)-1:x=torch.nn.functional.elu(x)
    error=float(np.max(abs(x.numpy()-forward(obs,weights,biases))))
    if error>5e-5:raise RuntimeError('Export disagreement '+str(error))
    meta=dict(skill=a.skill,obs_dim=dim,base_obs_dim=123,action_dim=37,phase_dim=2 if a.skill=='march' else 0,phase_period_s=.9,phase_semantics='sin/cos append on marching activation; explicit adapter, not a velocity command',action_mapping='original default + metadata.action_scale * action',checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),torch_numpy_max_error=error,export_verified=True,candidate_only=True)
    (a.output/'skill_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))

if __name__=='__main__':main()
