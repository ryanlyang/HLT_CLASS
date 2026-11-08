"""Masked TRAIN moments and equal-block per-jet reductions."""
import numpy as np
import torch
from torch.nn import functional as F
from .contracts import artifact, validate


class Moments:
    def __init__(self,width):
        self.count=np.zeros(width,np.int64); self.mean=np.zeros(width,np.float64); self.m2=np.zeros(width,np.float64)
    def add(self,values,valid):
        x=np.asarray(values,np.float64).reshape(-1,len(self.mean)); ok=np.asarray(valid,bool).reshape(x.shape)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite normalization input')
        n=ok.sum(0); mean=np.where(ok,x,0.).sum(0)/np.maximum(n,1)
        m2=np.where(ok,(x-mean)**2,0.).sum(0); total=self.count+n; delta=mean-self.mean
        self.m2+=m2+delta**2*self.count*n/np.maximum(total,1)
        self.mean+=delta*n/np.maximum(total,1); self.count=total
    def finish(self,*,parents,binary,blocks):
        sd=np.sqrt(np.maximum(self.m2/np.maximum(self.count,1),0.)); binary=np.asarray(binary,bool)
        mean=np.where(binary,0.,self.mean); scale=np.where(binary,1.,np.maximum(sd,.001))
        active=(self.count>0)&(sd>0)
        return artifact('NORMALIZATION',parents=parents,role='train',mean=mean.tolist(),scale=scale.tolist(),
            count=self.count.tolist(),active=active.tolist(),binary=binary.tolist(),blocks=list(map(int,blocks)))


def rows(pred,values,valid,norm,*,keep=None):
    validate(norm,'NORMALIZATION')
    if norm['role']!='train': raise PermissionError('Statistics are TRAIN-only')
    x=torch.as_tensor(values,device=pred.device,dtype=torch.float32).detach()
    ok=torch.as_tensor(valid,device=pred.device,dtype=torch.bool)
    mean=torch.tensor(norm['mean'],device=pred.device); scale=torch.tensor(norm['scale'],device=pred.device)
    active=torch.tensor(norm['active'],device=pred.device); binary=torch.tensor(norm['binary'],device=pred.device)
    if pred.shape!=x.shape or ok.shape!=x.shape or not torch.isfinite(pred).all() or not torch.isfinite(x).all():
        raise ValueError('Invalid scalar target/prediction')
    if keep is not None: active=active&torch.as_tensor(keep,device=pred.device)
    ok=ok&active
    target=(x-mean)/scale
    loss=torch.where(binary,F.binary_cross_entropy_with_logits(pred,x,reduction='none'),
                     F.smooth_l1_loss(pred,target,reduction='none'))
    blocks=torch.tensor(norm['blocks'],device=pred.device); sums=[]; enabled=[]
    for block in sorted(set(norm['blocks'])):
        use=ok&(blocks==block); n=use.sum(-1); sums.append((loss*use).sum(-1)/n.clamp(min=1)); enabled.append(n>0)
    stacked=torch.stack(sums,-1); support=torch.stack(enabled,-1)
    return stacked.sum(-1)/support.sum(-1).clamp(min=1)
