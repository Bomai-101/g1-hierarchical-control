"""Fixed 2x2 finger-limit/contact-time factorial; no policy intervention."""
import mujoco
import numpy as np
VARIANTS=('baseline','finger_only','contact_only','combined')
FINGERS=('left_four_joint','right_four_joint','left_six_joint','right_six_joint')
def apply_variant(model,variant):
 if variant not in VARIANTS:raise ValueError(variant)
 before={n:getattr(model,n).copy() for n in dir(model) if isinstance(getattr(model,n),np.ndarray)}
 if variant in ('finger_only','combined'):
  ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in FINGERS]
  if min(ids)<0:raise RuntimeError('Finger mapping')
  model.jnt_solref[ids,0]=.002
 if variant in ('contact_only','combined'):model.geom_solref[(model.geom_contype!=0)|(model.geom_conaffinity!=0),0]=.005
 changed=[n for n,v in before.items() if not np.array_equal(v,getattr(model,n))]
 expected=sorted((['jnt_solref'] if variant in ('finger_only','combined') else [])+(['geom_solref'] if variant in ('contact_only','combined') else []))
 if changed!=expected:raise RuntimeError('Unexpected model changes '+str(changed))
 return changed
