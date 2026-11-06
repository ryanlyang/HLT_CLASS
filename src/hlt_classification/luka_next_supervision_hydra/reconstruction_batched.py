"""Chunk-batched execution of the unchanged exact per-jet reconstruction objective."""
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

ASSIGNMENT_WORKERS=6


@lru_cache(maxsize=1)
def solver_pool():
    return ThreadPoolExecutor(max_workers=ASSIGNMENT_WORKERS,thread_name_prefix='reconstruction_assignment')


def solve(cost):
    return linear_sum_assignment(cost)


def host(value,dtype):
    if isinstance(value,torch.Tensor): value=value.detach().cpu().numpy()
    return np.array(value,dtype=dtype,copy=True)


def pack(targets,maximum,device):
    """Validate and canonicalize on CPU before one chunk's target transfers."""
    prepared=[]
    for t in targets:
        if set(t)!={'values','valid','mean','scale','frame','aggregate'}: raise ValueError('Invalid reconstruction target fields')
        x=host(t['values'],np.float32); valid=host(t['valid'],bool)
        mean=host(t['mean'],np.float32); scale=host(t['scale'],np.float32)
        frame=host(t['frame'],np.float64); aggregate=host(t['aggregate'],np.float64)
        if (x.ndim!=2 or x.shape[1]!=22 or valid.shape!=(len(x),4) or mean.shape!=(22,) or scale.shape!=(22,)
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<.001).any()):
            raise ValueError('Invalid reconstruction target shape/normalization')
        if len(x)>maximum: raise ValueError('Target count exceeds fixed TRAIN support; never crop')
        x[:,14:18]=np.where(valid,x[:,14:18],0.)
        if (not np.isfinite(x).all() or not np.isin(x[:,5:14],[0.,1.]).all()
                or not (x[:,5:11].sum(-1)==1).all() or not (x[:,11:14].sum(-1)==1).all()
                or not np.array_equal(x[:,18:22].astype(bool),valid) or not np.isin(x[:,18:22],[0.,1.]).all()):
            raise ValueError('Invalid real measured fields')
        if (frame.shape!=(3,) or not np.isfinite(frame).all() or frame[2]<=0
                or aggregate.shape!=(5,) or not np.isfinite(aggregate).all()):
            raise ValueError('Invalid observable HLT frame or full-set aggregate')
        order=np.lexsort(tuple(x[:,j] for j in range(21,-1,-1)))
        prepared.append(dict(values=x[order],valid=valid[order],mean=mean,scale=scale,frame=frame,aggregate=aggregate))
    counts=np.array([len(t['values']) for t in prepared],np.int64); width=max(1,int(counts.max()))
    values=np.zeros((len(targets),width,22),np.float32); flags=np.zeros((len(targets),width,4),bool)
    for i,t in enumerate(prepared):
        values[i,:counts[i]]=t['values']; flags[i,:counts[i]]=t['valid']
    payload=dict(values=values,valid=flags,counts=counts,real=np.arange(width)[None]<counts[:,None],
        **{k:np.stack([t[k] for t in prepared]) for k in ('mean','scale','frame','aggregate')})
    return {k:torch.as_tensor(v,device=device) for k,v in payload.items()},counts


def fields(score,t,*,pairwise):
    angles=torch.stack((score['angle'].sin(),score['angle'].cos()),-1)
    kin=torch.cat((score['kinematic'],(angles-t['mean'][:,None,3:5])/t['scale'][:,None,3:5]),-1)
    # Only the loss has padded targets; the decoder still predicts all K slots.
    expand=lambda a:a.unsqueeze(2) if pairwise else a
    x=t['values'].unsqueeze(1) if pairwise else t['values']
    flags=t['valid'].unsqueeze(1) if pairwise else t['valid']
    a,b=torch.broadcast_tensors(expand(kin),x[...,:5])
    lk=F.smooth_l1_loss(a,b,reduction='none',beta=1.).mean(-1)
    cat=score['category'].log_softmax(-1); charge=score['charge'].log_softmax(-1)
    lc=-.5*((expand(cat)*x[...,5:11]).sum(-1)/math.log(6)+(expand(charge)*x[...,11:14]).sum(-1)/math.log(3))
    logits,flags=torch.broadcast_tensors(expand(score['validity']),flags)
    lv=F.binary_cross_entropy_with_logits(logits,flags.float(),reduction='none').mean(-1)
    pred=expand(score['track']).expand_as(logits); actual=x[...,14:18].expand_as(pred)
    a=torch.where(flags,pred,0.); b=torch.where(flags,actual,0.)
    regression=F.smooth_l1_loss(a,b,reduction='none',beta=1.).sum(-1)/flags.float().sum(-1).clamp(min=1)
    lt=.5*lv+.5*regression
    return (lk+lc+lt)/3,dict(kinematic=lk,composition=lc,track=lt)


def assignments(scores,t,counts):
    """One cost transfer, then the same SciPy solver on each full rectangular cost."""
    b,k=scores['presence'].shape; width=t['values'].shape[1]
    with torch.no_grad():
        field,_=fields(scores,t,pairwise=True)
        n=t['counts'].float(); positive=F.softplus(-scores['presence']); negative=F.softplus(scores['presence'])
        cost=field/n.clamp(min=1)[:,None,None]+(positive-.1*negative)[:,:,None]/(n+.1*(k-n))[:,None,None]
        # No GPU round trips per jet, and no padded column enters the solver.
        cost=cost.double().cpu().numpy()
        if not np.isfinite(cost).all(): raise ValueError('Nonfinite assignment cost')
    slots=np.zeros((b,width),np.int64); chosen=np.zeros((b,k),bool)
    active=[i for i,n in enumerate(counts) if n]
    matrices=[cost[i,:,:counts[i]] for i in active]
    solutions=solver_pool().map(solve,matrices) if len(active)>1 and max(counts)>=32 else map(solve,matrices)
    for i,(a,c) in zip(active,solutions):
        slots[i,c]=a; chosen[i,a]=True
    return torch.as_tensor(slots,device=scores['presence'].device),torch.as_tensor(chosen,device=scores['presence'].device)


def consistency(scores,t):
    ref=t['frame']; q=ref[:,2]; xyz=scores['kinematic'].double()*t['scale'][:,None,:3].double()+t['mean'][:,None,:3].double()
    pt=xyz[...,0].exp()*q[:,None]; energy=xyz[...,1].exp()*q[:,None]
    eta=xyz[...,2]+ref[:,None,0]; phi=scores['angle'].double()+ref[:,None,1]
    p4=torch.stack((pt*phi.cos(),pt*phi.sin(),pt*eta.sinh(),energy),-1)
    if not torch.isfinite(p4).all(): raise ValueError('Nonfinite inverse predicted four-vector')
    weights=scores['presence'].double().sigmoid(); summed=(weights[...,None]*p4).sum(1); truth=t['aggregate']
    components=F.smooth_l1_loss(torch.asinh(summed/q[:,None]),torch.asinh(truth[:,:4]/q[:,None]),reduction='none').mean(-1)
    mass=lambda v:(v[:,3].square()-v[:,:3].square().sum(-1))/q.square()
    mass_term=F.smooth_l1_loss(torch.asinh(mass(summed)),torch.asinh(mass(truth[:,:4])),reduction='none')
    charged=(weights*p4[...,3]*(1-scores['charge'].double().softmax(-1)[...,1])).sum(1)
    fraction=charged/summed[:,3].clamp(min=q*1e-12)
    fraction_term=F.smooth_l1_loss(fraction,truth[:,4]/truth[:,3].clamp(min=q*1e-12),reduction='none')
    return (components+mass_term+fraction_term)/3


def rows(scores,targets,include_consistency=False,*,timings=None):
    b,k=scores['presence'].shape
    if b==0 or k==0 or len(targets)!=b: raise ValueError('Reconstruction batch join differs')
    shapes=dict(kinematic=(b,k,3),angle=(b,k),category=(b,k,6),charge=(b,k,3),track=(b,k,4),
        validity=(b,k,4),presence=(b,k),count=(b,k+1))
    if set(scores)!=set(shapes) or any(v.shape!=shapes[n] for n,v in scores.items()): raise ValueError('Fixed output/count support differs')
    if not torch.stack([torch.isfinite(v).all() for v in scores.values()]).all(): raise ValueError('Nonfinite reconstruction prediction')
    t,counts=pack(targets,k,scores['presence'].device)
    began=time.perf_counter(); slots,chosen=assignments(scores,t,counts)
    if timings is not None: timings['assignment_seconds']=timings.get('assignment_seconds',0.)+time.perf_counter()-began
    selected={name:torch.gather(v,1,slots if v.ndim==2 else slots[...,None].expand(-1,-1,v.shape[-1]))
        for name,v in scores.items() if name!='count'}
    matched,parts=fields(selected,t,pairwise=False)
    n=t['counts']; average=lambda x:torch.where(t['real'],x,0.).sum(-1)/n.clamp(min=1)
    field=average(matched); parts={name:average(v) for name,v in parts.items()}
    presence=torch.where(chosen,F.softplus(-scores['presence']),.1*F.softplus(scores['presence'])).sum(-1)/(n+.1*(k-n))
    count=F.cross_entropy(scores['count'],n,reduction='none')/math.log(k+1)
    extra=consistency(scores,t) if include_consistency else torch.zeros_like(field)
    values=field+presence+.1*count+.1*extra
    if not torch.isfinite(values).all(): raise ValueError('Nonfinite exact set loss')
    return values,dict(fields=field,presence=presence,count=count,consistency=extra,**parts)
