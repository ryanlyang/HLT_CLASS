"""One ordinary backward over CE, fixed U100-w5 and conditional subclass KL."""
import itertools
import torch
from torch.nn import functional as F
from hlt_classification.luka_fullstat_subclass.model import conditional_kl
from .contracts import coefficient
from .model import shared_parameters


def objective(task, logits, labels, prediction, target, subclass_logits, q, position):
    n = len(labels)
    if (labels.shape != (n,) or labels.dtype != torch.int64 or ((labels < 0) | (labels > 10)).any()
            or logits.shape != (n,11) or prediction.shape != (n,128) or target.shape != prediction.shape
            or subclass_logits.shape != (n,60) or q.shape != (n,10)
            or any(not torch.isfinite(v).all() for v in (logits,prediction,target,subclass_logits,q))
            or (q < 0).any() or not torch.allclose(q.sum(1), torch.ones(n,device=q.device),atol=2e-6,rtol=0)
            or (q[labels!=0,5:] != 0).any()):
        raise ValueError('Invalid objective shapes, labels or conditional targets')
    parts = dict(ce=F.cross_entropy(logits.float(),labels),
        embedding=F.smooth_l1_loss(prediction.float(),target.detach().float(),beta=1.,reduction='mean'),
        subclass=conditional_kl(subclass_logits,labels,q))
    parts['weighted_embedding'] = 5.*parts['embedding']
    parts['weighted_subclass'] = coefficient(task,position)*parts['subclass']
    parts['total'] = parts['ce']+parts['weighted_embedding']+parts['weighted_subclass']
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite objective')
    return parts['total'], parts


def backward(model, loss, parts, *, diagnose=False):
    diagnostic = None
    if diagnose:
        params = shared_parameters(model)
        vectors = {}
        for name in ('ce','weighted_embedding','weighted_subclass'):
            grads = torch.autograd.grad(parts[name],params,retain_graph=True,allow_unused=True)
            vectors[name] = [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)]
        norms = {k: sum(g.double().square().sum() for g in gs).sqrt() for k,gs in vectors.items()}
        diagnostic = {k+'_norm':float(v) for k,v in norms.items()}
        for a,b in itertools.combinations(vectors,2):
            dot = sum((x.double()*y.double()).sum() for x,y in zip(vectors[a],vectors[b]))
            diagnostic[a+'__'+b+'_cosine'] = float(dot/(norms[a]*norms[b]).clamp_min(1e-30))
        if not all(torch.isfinite(g).all() for gs in vectors.values() for g in gs):
            raise ValueError('Nonfinite diagnostic gradients')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite training gradients')
    return diagnostic


@torch.no_grad()
def statistics(logits, labels, q):
    offsets = torch.where(labels==0,0,10+5*(labels-1))
    columns = torch.arange(10,device=logits.device)[None,:]
    valid = (labels[:,None]==0) | (columns<5)
    z = logits.float().gather(1,(offsets[:,None]+columns).clamp_max(59)).masked_fill(~valid,-torch.inf)
    p = z.softmax(1)
    return dict(agreement=float((p.argmax(1)==q.argmax(1)).float().mean()),
        entropy=float(-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(1).mean()))
