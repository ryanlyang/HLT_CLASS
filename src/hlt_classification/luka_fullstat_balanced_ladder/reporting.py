"""Same selected checkpoint across natural, in-range and per-bin validation."""
import math
import numpy as np
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from . import contracts as c


def rejection(labels,probabilities,efficiency):
    if efficiency not in (.3,.5):raise ValueError('Unregistered rejection efficiency')
    p=np.asarray(probabilities,np.float64);qcd=labels==0;out={}
    for cls in range(1,11):
        signal=labels==cls;den=p[:,cls]+p[:,0];score=np.divide(p[:,cls],den,out=np.zeros(len(p)),where=den>0)
        n=int(qcd.sum());ns=int(signal.sum())
        if not n or not ns:
            out[c.CLASS_NAMES[cls]]=dict(status='missing_class',background_count=n,signal_count=ns);continue
        cut=float(np.sort(score[signal])[::-1][math.ceil(efficiency*ns)-1]);passing=int(np.count_nonzero(score[qcd]>=cut))
        out[c.CLASS_NAMES[cls]]=dict(status='censored' if passing==0 else 'measured',rejection=n/passing if passing else None,
            threshold=cut,background_count=n,background_passing=passing,signal_count=ns,
            signal_passing=int(np.count_nonzero(score[signal]>=cut)),one_event_resolution=n)
    return out


def metrics(labels,p,pt,parents):
    if len(labels)!=len(pt) or not np.isfinite(pt).all():raise ValueError('Validation pT differs')
    primary=evaluate_probabilities(labels,p)
    inside=(pt>=200)&(pt<2000)
    masks={'natural':np.ones(len(pt),bool),'in_range':inside,'outside_range':~inside}
    masks.update({f'{low}_{low+100}':(pt>=low)&(pt<low+100) for low in range(200,2000,100)})
    out={}
    for name,mask in masks.items():
        y=labels[mask];prob=p[mask]
        complete=set(np.unique(y))==set(range(11))
        summary=primary if name=='natural' else evaluate_probabilities(y,prob) if complete else dict(rows=len(y),status='missing_class')
        out[name]=dict(summary=summary,R50=rejection(y,prob,.5),R30=rejection(y,prob,.3))
    return c.artifact('METRICS',parents=parents,populations=out,checkpoint_reselection=False)


def recovery(value,low,high):
    if any(v is None for v in (value,low,high)) or high<=low:return None
    return 100*(value-low)/(high-low)
