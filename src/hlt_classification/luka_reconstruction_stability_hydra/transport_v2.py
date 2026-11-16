"""Versioned numerical recovery of the original entropy-regularized assignment.

Keep the v1 fast path. If scaling stalls, minimize the row-normalized dual with
safeguarded Newton steps on CPU in FP64. This is the same soft objective, not a
hard assignment or an acceptance of the unconverged scaling iterate.
"""
import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from . import transport

PROTOCOL = dict(name='soft_dual_newton/v2', fast_iterations=500,
    newton_iterations=256, line_search_steps=40, potential_step_cap=50.,
    hessian_regularization=1e-12, marginal_tolerance=1e-4, gap_tolerance=1e-4)


class ConvergenceError(RuntimeError):
    """Retain the exact numerical input so the worker can publish a replay case."""
    def __init__(self, cost, masses, temperature, record):
        super().__init__('Soft v2 did not converge; no hard fallback: '+str(record))
        self.problem = dict(cost=cost.detach().cpu(), masses=masses.detach().cpu(),
                            temperature=temperature, record=record, protocol=PROTOCOL)


def _newton(cost, masses, temperature, potential):
    c=cost.detach().double().cpu().numpy(); b=masses.detach().double().cpu().numpy()
    k,n=c.shape; logq=np.broadcast_to(np.log(b/k),c.shape)
    # Removing row constants improves conditioning without changing the plan.
    offsets=c.min(1)/temperature; kernel=logq-(c-c.min(1)[:,None])/temperature
    gauge=int(b.argmax()); free=np.arange(n)!=gauge
    v=potential.detach().double().cpu().numpy().copy(); v-=v[gauge]

    def evaluate(v):
        z=kernel+v; normalizer=logsumexp(z,axis=1)
        logp=z-normalizer[:,None]; p=np.exp(logp)
        return p,logp,float(normalizer.sum()-b@v),normalizer

    for iteration in range(PROTOCOL['newton_iterations']+1):
        p,logp,objective,normalizer=evaluate(v)
        entropy=float(np.sum(p*(logp-logq))); primal=float(np.sum(p*c)+temperature*entropy)
        u=offsets-normalizer
        dual=float(temperature*(u.sum()+b@v+k-p.sum()))
        residual=float(max(np.abs(p.sum(1)-1).max(),np.abs(p.sum(0)-b).max()))
        gap=abs(primal-dual)/max(abs(primal),1.)
        record=dict(marginal_residual=residual,normalized_gap=gap,primal=primal,
            dual=dual,entropy=entropy,iterations=PROTOCOL['fast_iterations']+iteration,
            newton_iterations=iteration,solver=PROTOCOL['name'])
        if not np.isfinite([objective,primal,dual,residual,gap]).all():
            raise ValueError('Nonfinite soft dual solve')
        if residual<=PROTOCOL['marginal_tolerance'] and gap<=PROTOCOL['gap_tolerance']:
            return torch.as_tensor(p,device=cost.device),record
        if iteration==PROTOCOL['newton_iterations']:break
        gradient=p.sum(0)-b
        hessian=np.diag(p.sum(0))-p.T@p
        h=hessian[np.ix_(free,free)]
        h.flat[::len(h)+1]+=PROTOCOL['hessian_regularization']
        direction=np.zeros(n)
        try:direction[free]=np.linalg.solve(h,-gradient[free])
        except np.linalg.LinAlgError:break
        maximum=np.abs(direction).max()
        if maximum>PROTOCOL['potential_step_cap']:direction*=PROTOCOL['potential_step_cap']/maximum
        slope=float(gradient@direction)
        if not np.isfinite(direction).all() or slope>=0:break
        accepted=False
        for backtrack in range(PROTOCOL['line_search_steps']):
            step=2.**(-backtrack); candidate=v+step*direction
            new_p,_,new_objective,_=evaluate(candidate)
            # Near machine precision, permit equal dual objective only with
            # strictly improved marginal error. Never bypass final gap checks.
            improves=np.abs(new_p.sum(0)-b).max()<residual
            if (new_objective<=objective+1e-4*step*slope or
                (abs(new_objective-objective)<=1e-12*max(1.,abs(objective)) and improves)):
                v=candidate; accepted=True; break
        if not accepted:break
    raise ConvergenceError(cost,masses,temperature,record)


@torch.no_grad()
def solve(cost,masses,temperature):
    try:
        plan,record=transport.solve(cost,masses,temperature)
        return plan,dict(record,solver=PROTOCOL['name'],newton_iterations=0)
    except transport.ScalingConvergenceError as exc:
        potential=exc.potential
    with threadpool_limits(limits=1):
        return _newton(cost,masses,temperature,potential)
