"""Physical four-vectors and differentiable native pair features, last axis = 4."""
import math
import torch
from torch.nn import functional as F

EPS = 1e-8


def wrap(phi):
    return (phi + math.pi) % (2*math.pi) - math.pi


def physical_audit(p):
    if p.shape[-1] != 4 or not torch.isfinite(p).all():
        raise ValueError('Nonfinite/invalid physical four-vector')
    pt2 = p[..., :2].square().sum(-1)
    p2 = p[..., :3].square().sum(-1); e2 = p[..., 3].square()
    tolerance = 1e-6*torch.maximum(torch.maximum(p2, e2), torch.ones_like(e2))
    if (pt2 <= 0).any() or (p[..., 3] <= 0).any() or ((e2-p2) < -tolerance).any():
        raise ValueError('Nonphysical p4: require positive pt/E and m2 >= -roundoff tolerance')
    return int(((e2-p2) < 0).sum())


def coordinates(p):
    pt2 = p[..., :2].square().sum(-1).clamp_min(EPS**2)
    pt = pt2.sqrt()
    # Same rapidity convention as pinned Weaver, not pseudorapidity.
    rapidity = .5*torch.log1p(2*p[..., 2]/(p[..., 3]-p[..., 2]).clamp_min(1e-20))
    mass2 = (p[..., 3].square()-p[..., :3].square().sum(-1)).clamp_min(0)
    return torch.stack((pt.log(), rapidity, torch.atan2(p[..., 1], p[..., 0]),
                        torch.log1p(mass2/pt2)), -1)


def difference(a, b):
    d = a-b
    return torch.stack((d[..., 0], d[..., 1], wrap(d[..., 2]), d[..., 3]), -1)


def pair_features(a, b):
    """Native values, with finite derivatives for exactly collinear real pairs.

    Preserve native kt=0 at delta=0. Flooring delta before kt would change it
    for pt>1. Below eps the native value is retained with a stopped derivative.
    """
    qa, qb = coordinates(a), coordinates(b)
    pt_a = a[..., :2].square().sum(-1).sqrt()
    pt_b = b[..., :2].square().sum(-1).sqrt()
    d2 = (qa[..., 1]-qb[..., 1]).square()+wrap(qa[..., 2]-qb[..., 2]).square()
    delta = torch.where(d2 >= EPS**2, d2.clamp_min(EPS**2).sqrt(), d2.detach().sqrt())
    minimum = torch.minimum(pt_a, pt_b)
    summed = a+b
    m2 = summed[..., 3].square()-summed[..., :3].square().sum(-1)
    return torch.stack(((minimum*delta).clamp_min(EPS).log(),
                        (minimum/(pt_a+pt_b).clamp_min(EPS)).clamp_min(EPS).log(),
                        delta.clamp_min(EPS).log(), m2.clamp_min(EPS).log()), -1)


def decode(p, raw, scale, eta_sign):
    q = coordinates(p)
    a = 6*torch.tanh(raw/6)
    log_pt = q[..., 0]+scale[0]*a[..., 0]
    y = q[..., 1]+eta_sign*scale[1]*a[..., 1]
    phi = q[..., 2]+scale[2]*a[..., 2]
    u = q[..., 3]+EPS
    inverse_softplus = u+torch.log(-torch.expm1(-u))
    mu = F.softplus(inverse_softplus+a[..., 3])
    pt = log_pt.exp(); mt = pt*torch.exp(mu/2)
    result = torch.stack((pt*phi.cos(), pt*phi.sin(), mt*y.sinh(), mt*y.cosh()), -1)
    if not torch.isfinite(result).all() or (pt <= 0).any() or (result[..., 3] <= 0).any():
        raise ValueError('Nonfinite/nonpositive corrected p4; no clipping or fallback')
    return result


def pair_indices(mask):
    """Unique real pairs, packed across jets; padding/diagonal never evaluated."""
    n = mask.shape[1]
    i, j = torch.triu_indices(n, n, 1, device=mask.device)
    batch, pair = torch.where(mask[:, i] & mask[:, j])
    return batch, i[pair], j[pair]
