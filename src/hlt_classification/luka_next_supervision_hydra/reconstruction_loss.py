"""Exact rectangular set matching and graph-connected mixed-field reconstruction."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def huber(a,b):
    return F.smooth_l1_loss(a.expand_as(b),b,reduction='none',beta=1.)


def tensor(value,device,dtype=torch.float32):
    return torch.as_tensor(np.array(value,copy=True) if isinstance(value,np.ndarray) else value,
        device=device,dtype=dtype).detach()


def target(value,device):
    result={k:tensor(v,device,torch.bool if k=='valid' else torch.float64 if k in ('frame','aggregate') else torch.float32) for k,v in value.items()}
    x=result['values']; valid=result['valid']; mean=result['mean']; scale=result['scale']
    if (x.ndim!=2 or x.shape[-1]!=22 or valid.shape!=(len(x),4) or mean.shape!=(22,) or scale.shape!=(22,)
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<.001).any()):
        raise ValueError('Invalid reconstruction target shape/normalization')
    # Invalid field sentinels never enter a residual, including the detached cost.
    x=x.clone(); x[:,14:18]=torch.where(valid,x[:,14:18],0.)
    if (not torch.isfinite(x).all() or not torch.isin(x[:,5:14],x.new_tensor([0.,1.])).all()
            or not (x[:,5:11].sum(-1)==1).all() or not (x[:,11:14].sum(-1)==1).all()
            or not torch.equal(x[:,18:22].bool(),valid) or not torch.isin(x[:,18:22],x.new_tensor([0.,1.])).all()):
        raise ValueError('Invalid real measured fields')
    # Deterministic lexicographic target order, without perturbing an optimal cost.
    key=x.cpu().numpy(); order=np.lexsort(tuple(key[:,j] for j in range(21,-1,-1)))
    ix=torch.as_tensor(order,device=device); result['values']=x[ix]; result['valid']=valid[ix]
    return result


def kinematics(score,t):
    angles=torch.stack((score['angle'].sin(),score['angle'].cos()),-1)
    return torch.cat((score['kinematic'],(angles-t['mean'][3:5])/t['scale'][3:5]),-1)


def fields(score,t,*,pairwise):
    x=t['values']; valid=t['valid']; kin=kinematics(score,t)
    # Pairwise K x N is used only under no_grad; matched tensors have N rows.
    expand=lambda a:a[:,None] if pairwise else a
    shape=(len(kin),len(x)) if pairwise else (len(x),)
    def smooth(a,b):
        a,b=torch.broadcast_tensors(expand(a),b)
        return F.smooth_l1_loss(a,b,reduction='none',beta=1.)
    lk=smooth(kin,x[:,:5]).mean(-1)
    cat=score['category'].log_softmax(-1); charge=score['charge'].log_softmax(-1)
    lc=-.5*((expand(cat)*x[:,5:11]).sum(-1)/math.log(6)+(expand(charge)*x[:,11:14]).sum(-1)/math.log(3))
    logits=expand(score['validity']).expand(*shape,4); flags=valid.float().expand_as(logits)
    lv=F.binary_cross_entropy_with_logits(logits,flags,reduction='none').mean(-1)
    pred=expand(score['track']).expand(*shape,4); actual=x[:,14:18].expand_as(pred)
    safe_pred=torch.where(flags.bool(),pred,0.); safe_actual=torch.where(flags.bool(),actual,0.)
    regression=F.smooth_l1_loss(safe_pred,safe_actual,reduction='none').sum(-1)/flags.sum(-1).clamp(min=1)
    lt=.5*lv+.5*regression
    return (lk+lc+lt)/3,dict(kinematic=lk,composition=lc,track=lt)


def match(score,t):
    k=score['presence'].numel(); n=len(t['values'])
    if n>k: raise ValueError('Target count exceeds fixed TRAIN support; never crop')
    if n==0: return np.empty(0,np.int64),np.empty(0,np.int64)
    with torch.no_grad():
        field,_=fields(score,t,pairwise=True)
        positive=F.softplus(-score['presence']); negative=F.softplus(score['presence'])
        cost=field/n+(positive-.1*negative)[:,None]/(n+.1*(k-n))
        if not torch.isfinite(cost).all(): raise ValueError('Nonfinite assignment cost')
        return linear_sum_assignment(cost.double().cpu().numpy())


def physical(score,t):
    # Known HLT coordinate transforms are used only by the loss, never as
    # extra decoder input. FP64 inverse and sums retain their gradients.
    ref=t['frame'].double(); q=ref[2]
    if ref.shape!=(3,) or not torch.isfinite(ref).all() or q<=0: raise ValueError('Invalid observable HLT frame')
    xyz=score['kinematic'].double()*t['scale'][:3].double()+t['mean'][:3].double()
    pt=xyz[:,0].exp()*q; energy=xyz[:,1].exp()*q; eta=xyz[:,2]+ref[0]; phi=score['angle'].double()+ref[1]
    p4=torch.stack((pt*phi.cos(),pt*phi.sin(),pt*eta.sinh(),energy),-1)
    if not torch.isfinite(p4).all(): raise ValueError('Nonfinite inverse predicted four-vector')
    return p4


def consistency(score,t):
    p4=physical(score,t); weights=score['presence'].double().sigmoid(); q=t['frame'][2].double()
    summed=(weights[:,None]*p4).sum(0); truth=t['aggregate'].double()
    if truth.shape!=(5,) or not torch.isfinite(truth).all(): raise ValueError('Invalid full-set aggregate')
    components=F.smooth_l1_loss(torch.asinh(summed/q),torch.asinh(truth[:4]/q),reduction='mean')
    mass=lambda v:(v[3].square()-v[:3].square().sum())/q.square()
    mass_term=F.smooth_l1_loss(torch.asinh(mass(summed)),torch.asinh(mass(truth[:4])))
    charged=(weights*p4[:,3]*(1-score['charge'].double().softmax(-1)[:,1])).sum()
    fraction=charged/summed[3].clamp(min=q*1e-12)
    fraction_term=F.smooth_l1_loss(fraction,truth[4]/truth[3].clamp(min=q*1e-12))
    return (components+mass_term+fraction_term)/3


def row(score,t,include_consistency=False,*,timings=None):
    k=score['presence'].numel(); n=len(t['values'])
    if any(not torch.isfinite(v).all() for v in score.values()): raise ValueError('Nonfinite reconstruction prediction')
    if score['count'].shape!=(k+1,): raise ValueError('Fixed count support differs')
    began=time.perf_counter(); slots,columns=match(score,t)
    if timings is not None: timings['assignment_seconds']=timings.get('assignment_seconds',0.)+time.perf_counter()-began
    chosen=torch.zeros(k,device=score['presence'].device,dtype=torch.bool); chosen[slots]=True
    if n:
        s={name:v[slots] for name,v in score.items() if name!='count'}
        matched=dict(t,values=t['values'][columns],valid=t['valid'][columns])
        field,parts=fields(s,matched,pairwise=False); field=field.mean(); parts={name:v.mean() for name,v in parts.items()}
    else:
        field=score['kinematic'].sum()*0; parts={name:field for name in ('kinematic','composition','track')}
    presence=(F.softplus(-score['presence'][chosen]).sum()+.1*F.softplus(score['presence'][~chosen]).sum())/(n+.1*(k-n))
    count=F.cross_entropy(score['count'][None],torch.tensor([n],device=score['count'].device))/math.log(k+1)
    extra=consistency(score,t) if include_consistency else field.new_zeros(())
    loss=field+presence+.1*count+.1*extra
    if not torch.isfinite(loss): raise ValueError('Nonfinite exact set loss')
    return loss,dict(fields=field,presence=presence,count=count,consistency=extra,**parts)


def rows(scores,targets,include_consistency=False,*,timings=None):
    if len(targets)!=len(scores['presence']): raise ValueError('Reconstruction batch join differs')
    values=[]; details=[]
    for i,t in enumerate(targets):
        value,parts=row({k:v[i] for k,v in scores.items()},target(t,scores['presence'].device),include_consistency,timings=timings)
        values.append(value); details.append(parts)
    return torch.stack(values),{k:torch.stack([r[k] for r in details]) for k in details[0]}
