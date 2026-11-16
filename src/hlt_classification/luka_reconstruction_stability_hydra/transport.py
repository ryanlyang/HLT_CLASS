"""Full-support entropy transport; detached optimum gives the envelope gradient."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as hard
from .contracts import SOFT_PROTOCOL


class ScalingConvergenceError(RuntimeError):
    def __init__(self, residual, gap, potential):
        super().__init__(f'Soft assignment did not converge: residual={residual:.6g}, gap={gap:.6g}; no hard fallback')
        self.potential=potential.detach().clone()


def temperature_at(scale,position):
    if not math.isfinite(scale) or scale<=0 or not 3.<position<9.:raise ValueError('Invalid soft interval/scale')
    return scale*.1*(.1**((position-3.)/6.))


def costs(scores,targets):
    k=scores['presence'].shape[1];packed,counts=hard.pack(targets,k,scores['presence'].device)
    fields,_=hard.fields(scores,packed,pairwise=True)
    values=[];masses=[]
    for i,n in enumerate(counts):
        if not n:values.append(None);masses.append(None);continue
        denominator=float(n+.1*(k-n))
        real=fields[i,:,:n]/float(n)+F.softplus(-scores['presence'][i])[:,None]/denominator
        mass=torch.ones(n,device=real.device,dtype=torch.float64)
        if n<k:
            real=torch.cat((real,.1*F.softplus(scores['presence'][i])[:,None]/denominator),dim=1)
            mass=torch.cat((mass,mass.new_tensor([k-n])))
        values.append(real);masses.append(mass)
    return values,masses,counts


@torch.no_grad()
def solve(cost,masses,temperature,*,iterations=None):
    """FP64 log-domain Sinkhorn; residual AND objective gap must pass."""
    if cost.ndim!=2 or not cost.numel() or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid transport problem')
    c=cost.detach().double();b=masses.detach().to(c).double();k,n=c.shape
    if b.shape!=(n,) or (b<=0).any() or float(b.sum())!=k or not torch.isfinite(c).all():
        raise ValueError('Invalid transport marginal/cost')
    logq=(b/k).log()[None].expand_as(c);kernel=logq-c/temperature;v=torch.zeros_like(b);u=c.new_zeros(k)
    for iteration in range(1,(iterations or SOFT_PROTOCOL['iterations'])+1):
        # Damped initial equilibration followed by log-potential overrelaxation.
        # This changes the numerical path only: the exact original marginal
        # and primal/dual tests still accept the same regularized optimum.
        omega=SOFT_PROTOCOL['log_relaxation'] if iteration>SOFT_PROTOCOL['relaxation_start'] else 1.
        u=(1.-omega)*u-omega*torch.logsumexp(kernel+v[None],dim=1)
        v=(1.-omega)*v+omega*(b.log()-torch.logsumexp(kernel+u[:,None],dim=0))
        if iteration%10 and iteration!=(iterations or SOFT_PROTOCOL['iterations']):continue
        logp=kernel+u[:,None]+v[None];p=logp.exp()
        entropy=(p*(logp-logq)).sum();primal=(p*c).sum()+temperature*entropy
        dual=temperature*(u.sum()+(b*v).sum()+k-p.sum())
        residual=torch.maximum((p.sum(1)-1).abs().max(),(p.sum(0)-b).abs().max())
        gap=(primal-dual).abs()/primal.abs().clamp_min(1.)
        if not torch.isfinite(torch.stack((primal,dual,residual,gap))).all():raise ValueError('Nonfinite transport solution')
        if float(residual)<=SOFT_PROTOCOL['marginal_tolerance'] and float(gap)<=SOFT_PROTOCOL['gap_tolerance']:
            return p,dict(marginal_residual=float(residual),normalized_gap=float(gap),iterations=iteration,
                primal=float(primal),dual=float(dual),entropy=float(entropy))
    raise ScalingConvergenceError(float(residual),float(gap),v)


def rows(scores,targets,temperature,*,robust=False):
    matrices,masses,counts=costs(scores,targets);k=scores['presence'].shape[1];losses=[];details=[]
    if not all(torch.isfinite(v).all() for v in scores.values()):raise ValueError('Nonfinite reconstruction prediction')
    for i,n in enumerate(counts):
        count=F.cross_entropy(scores['count'][i:i+1],torch.tensor([n],device=scores['count'].device))/math.log(k+1)
        if not n:
            losses.append(F.softplus(scores['presence'][i]).mean()+.1*count)
            details.append(dict(marginal_residual=0.,normalized_gap=0.,iterations=0.,entropy=0.))
            continue
        solver=solve
        if robust:
            from .transport_v2 import solve as solver
        plan,record=solver(matrices[i],masses[i],temperature)
        # The solver is detached, but this expression differentiates the original
        # cost tensor. The entropy scalar completes the regularized objective.
        value=(plan*matrices[i].double()).sum()+temperature*record['entropy']+.1*count
        if not torch.isfinite(value):raise ValueError('Nonfinite regularized objective')
        losses.append(value);details.append(record)
    return torch.stack(losses),dict(transport_residual_max=max(d['marginal_residual'] for d in details),
        transport_gap_max=max(d['normalized_gap'] for d in details),
        transport_iterations_max=max(d['iterations'] for d in details),
        transport_entropy_mean=float(np.mean([d['entropy'] for d in details])))
