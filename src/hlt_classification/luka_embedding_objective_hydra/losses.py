"""Frozen geometry losses and asymmetric first-order CE gradient protection."""
import torch
from torch.nn import functional as F
from .contracts import cell, recipe, TERMS
from .model import shared_parameters


def importance(raw):
    """Bound relative sensitivity before mean-one normalization; zeros are uniform."""
    raw = raw.clamp_min(0)
    mean = raw.mean(dim=-1, keepdim=True)
    normalized = raw / mean.clamp_min(torch.finfo(raw.dtype).tiny)
    low, high = recipe()['importance_clip']
    bounded = normalized.clamp(low, high)
    return bounded / bounded.mean(dim=-1, keepdim=True)


def component(name, prediction, target, geometry):
    e = prediction.float() - target.detach().float()
    if name == 'base':
        return F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=1., reduction='mean')
    if name in ('coordinate', 'fisher'):
        if name == 'coordinate':
            weights = geometry.coordinate
        else:
            with torch.no_grad():
                a = geometry.readout
                t = recipe()['fisher_temperature']
                p = torch.softmax((target.detach().float() @ a.T + geometry.bias)/t, dim=1)
                weights = importance(((p @ a.square()) - (p @ a).square()).clamp_min(0)/(t*t))
        return (weights * F.smooth_l1_loss(e, torch.zeros_like(e), reduction='none', beta=1.)).mean()
    transform = {'class_span': geometry.class_projector, 'discriminant': geometry.discriminant_projector,
                 'whitened': geometry.whitener}
    if name not in transform:
        raise ValueError('Unknown embedding term')
    transformed = e @ transform[name]
    return F.smooth_l1_loss(transformed, torch.zeros_like(transformed), beta=1., reduction='mean')


def objective(task, logits, labels, prediction, target, geometry):
    setting = cell(task)
    if (logits.shape != (len(labels), 11) or prediction.shape != (len(labels), 128)
            or target.shape != prediction.shape
            or any(not torch.isfinite(x).all() for x in (logits, prediction, target))):
        raise ValueError('Invalid/nonfinite objective inputs')
    base = component('base', prediction, target, geometry)
    aux = base
    if setting['term'] is not None:
        name, f = setting['term'], setting['fraction']
        alternative = component(name, prediction, target, geometry) * geometry.scales[TERMS.index(name)]
        aux = (1-f)*base + f*alternative
    terms = dict(ce=F.cross_entropy(logits.float(), labels.long()), u100=5.*aux)
    if any(not torch.isfinite(v) for v in terms.values()):
        raise ValueError('Nonfinite loss')
    return sum(terms.values()), terms


def project(ce, aux, epsilon=1e-20):
    """Lists cover identical shared parameters, using zeros for unused gradients."""
    dot = sum((c.double()*a.double()).sum() for c, a in zip(ce, aux))
    norm2 = sum(c.double().square().sum() for c in ce)
    aux2 = sum(a.double().square().sum() for a in aux)
    coefficient = torch.where((dot < 0) & (norm2 > epsilon), dot / norm2.clamp_min(epsilon), 0.)
    adjusted = [a - coefficient.to(a.dtype)*c for c, a in zip(ce, aux)]
    after = sum((c.double()*a.double()).sum() for c, a in zip(ce, adjusted))
    return adjusted, dict(ce_norm=float(norm2.sqrt()), auxiliary_norm=float(aux2.sqrt()),
        dot_before=float(dot), dot_after=float(after), projected=bool(coefficient < 0), coefficient=float(coefficient))


def backward(model, loss, terms, *, protect=False, diagnose=False):
    shared = shared_parameters(model)
    diagnostic = None
    if protect:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        ce_raw = torch.autograd.grad(terms['ce'], params, retain_graph=True, allow_unused=True)
        aux_raw = torch.autograd.grad(terms['u100'], params, allow_unused=True)
        selected = {n for n, _ in shared}
        ids = [i for i, (n, _) in enumerate(named) if n in selected]
        ce = [torch.zeros_like(params[i]) if ce_raw[i] is None else ce_raw[i].detach() for i in ids]
        aux = [torch.zeros_like(params[i]) if aux_raw[i] is None else aux_raw[i].detach() for i in ids]
        adjusted, diagnostic = project(ce, aux, recipe()['projection_epsilon'])
        diagnostic['protection_applied'] = True
        projected = dict(zip(ids, [c+a for c, a in zip(ce, adjusted)]))
        for i, (_, p) in enumerate(named):
            c, a = ce_raw[i], aux_raw[i]
            if c is None and a is None:
                p.grad = None
            elif i in projected:
                p.grad = projected[i]
            else:
                p.grad = a if c is None else c if a is None else c+a
        if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
            raise ValueError('Nonfinite projected gradients')
        return diagnostic
    if diagnose:
        params = [p for _, p in shared]
        c = torch.autograd.grad(terms['ce'], params, retain_graph=True, allow_unused=True)
        a = torch.autograd.grad(terms['u100'], params, retain_graph=True, allow_unused=True)
        cg = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, c)]
        ag = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, a)]
        if not all(torch.isfinite(g).all() for g in (*cg, *ag)):
            raise ValueError('Nonfinite task gradients')
        adjusted, diagnostic = project(cg, ag, recipe()['projection_epsilon'])
        diagnostic['protection_applied'] = protect
    # Ordinary runs retain the original single backward; optional probes do not
    # mutate parameters, RNG, buffers or accumulated .grad tensors.
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite accumulated gradients')
    return diagnostic
