"""Single-parameter contact probes; no joint-limit or actor changes."""
import numpy as np
VARIANTS={'baseline':None,'friction06':('geom_friction',0,.6),'friction10':('geom_friction',0,1.),'contact_time005':('geom_solref',0,.005),'contact_time010':('geom_solref',0,.01),'contact_time040':('geom_solref',0,.04)}
def apply_variant(model,variant):
 before={n:getattr(model,n).copy() for n in dir(model) if isinstance(getattr(model,n),np.ndarray)}
 spec=VARIANTS[variant];mask=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
 if spec:getattr(model,spec[0])[mask,spec[1]]=spec[2]
 changed=[n for n,v in before.items() if not np.array_equal(v,getattr(model,n))]
 if changed!=([spec[0]] if spec else []):raise RuntimeError('Unexpected model changes '+str(changed))
 return changed
