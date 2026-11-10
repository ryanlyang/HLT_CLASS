"""Decision sensitivities at both deterministic native inputs; no learned tap targets."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .contracts import recipe


@contextmanager
def explanation_mode(model):
    modes = {m: m.training for m in model.modules()}
    capture = getattr(model, 'capture', None)
    model.eval()
    if capture is not None: model.capture = False
    try:
        with torch.autocast(next(model.parameters()).device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        for m, mode in modes.items(): m.training = mode
        if capture is not None: model.capture = capture


def scales(reference):
    particle, pair = reference.mod.embed.input_bn, reference.mod.pair_embed.embed[0]
    if not isinstance(particle, nn.BatchNorm1d) or not isinstance(pair, nn.BatchNorm1d):
        raise ValueError('Native input normalization taps differ')
    result = [torch.sqrt(m.running_var.detach()+m.eps).clamp_min(.001) for m in (particle,pair)]
    if tuple(x.numel() for x in result) != (17,4) or any(not torch.isfinite(x).all() for x in result):
        raise ValueError('Reference scales differ')
    return result


def pair_layout(model, mask):
    p = model.mod.pair_embed
    if not p.is_symmetric or p.pairwise_lv_dim != 4: raise ValueError('Expected symmetric native four-physics pair stream')
    batch, _, length = mask.shape
    valid = mask[:,0].bool()
    if p.sparse_eval[1]:
        owners, ii, jj = (valid[:,:,None] & valid[:,None,:]).tril(-1 if p.remove_self_pair else 0).nonzero(as_tuple=True)
        return owners, torch.ones(len(owners),dtype=torch.bool,device=mask.device), (1,4,len(owners))
    ii,jj = torch.tril_indices(length,length,offset=-1 if p.remove_self_pair else 0,device=mask.device)
    owners = torch.arange(batch,device=mask.device)[:,None].expand(-1,len(ii)).reshape(-1)
    return owners,(valid[:,ii] & valid[:,jj]).reshape(-1),(batch,4,len(ii))


def margin(logits, labels):
    other = logits.masked_fill(F.one_hot(labels.long(),11).bool(), -torch.inf)
    return logits.gather(1,labels.long()[:,None])[:,0]-torch.logsumexp(other,dim=1)


def sensitivity(model, raw, scale, *, create_graph=False):
    """Per-coordinate scaled derivatives and jet owner indices, excluding padding."""
    features = raw['features'].detach().float().requires_grad_(True)
    owners, valid_pairs, expected = pair_layout(model,raw['mask'])
    captured = []
    def tap(_module, args):
        x = args[0]
        if tuple(x.shape) != expected or captured: raise ValueError('Pair tap shape/call count differs')
        leaf = x.detach().float().requires_grad_(True); captured.append(leaf)
        return (leaf,)
    handle = model.mod.pair_embed.embed.register_forward_pre_hook(tap)
    try:
        with explanation_mode(model), torch.enable_grad():
            logits = model(features,raw['vectors'].detach().float(),raw['mask'])
            if len(captured) != 1 or not torch.isfinite(logits).all(): raise ValueError('Missing native pair stream/nonfinite logits')
            grads = torch.autograd.grad(margin(logits,raw['labels']).sum(),(features,captured[0]),create_graph=create_graph)
    finally: handle.remove()
    mask = raw['mask'][:,0].bool(); batch = len(features)
    particle = (grads[0]*scale[0][None,:,None]).transpose(1,2)[mask]
    particle_owners = torch.arange(batch,device=features.device)[:,None].expand_as(mask)[mask]
    pair = (grads[1]*scale[1][None,:,None]).transpose(1,2).reshape(-1,4)[valid_pairs]
    owner = torch.cat((particle_owners[:,None].expand(-1,17).reshape(-1), owners[valid_pairs,None].expand(-1,4).reshape(-1)))
    g = torch.cat((particle.reshape(-1),pair.reshape(-1)))
    if not torch.isfinite(g).all(): raise ValueError('Nonfinite input sensitivity')
    return g, owner


def mask_and_energy(g, owner, batch):
    maximum = torch.zeros(batch,device=g.device,dtype=g.dtype).scatter_reduce_(0,owner,g.detach().abs(),reduce='amax')
    mask = (g.detach().abs() >= .67*maximum[owner]) & (g.detach().abs() > 0)
    energy = torch.zeros_like(maximum).scatter_add_(0,owner,mask*g.detach().square()).clamp_min(1e-4)
    return mask, energy


def penalty(model, reference, raw):
    scale = scales(reference)
    g0, owner = sensitivity(reference,raw,scale)
    mask, energy = mask_and_energy(g0,owner,len(raw['features']))
    g, actual_owner = sensitivity(model,raw,scale,create_graph=True)
    if not torch.equal(owner,actual_owner): raise ValueError('Explanation coordinate alignment differs')
    per_jet = torch.zeros_like(energy).scatter_add(0,owner,mask*g.square())/energy
    loss = per_jet.mean()*recipe()['explanation']['coefficient']
    if not torch.isfinite(loss): raise ValueError('Nonfinite explanation penalty')
    return loss, dict(mask_fraction=float(mask.float().mean()), reference_energy=float(energy.mean()))


def finite_response(model, reference, raw):
    """Finite .05-SD displacement along T0's marked sensitivity signs at both taps."""
    scale=scales(reference); g,owner=sensitivity(reference,raw,scale)
    marked,_=mask_and_energy(g,owner,len(raw['features']))
    mask=raw['mask'][:,0].bool(); particle_count=int(mask.sum())*17
    direction=(marked*g.sign()).detach(); pf=torch.zeros_like(raw['features']).transpose(1,2)
    pf[mask]=direction[:particle_count].reshape(-1,17)*scale[0]
    _,valid_pairs,shape=pair_layout(reference,raw['mask'])
    pair=torch.zeros((shape[0]*shape[2],4),device=g.device)
    pair[valid_pairs]=direction[particle_count:].reshape(-1,4)*scale[1]
    pair=pair.reshape(shape[0],shape[2],4).transpose(1,2)
    def response(m):
        with explanation_mode(m),torch.no_grad():
            before=margin(m(raw['features'],raw['vectors'],raw['mask']),raw['labels'])
            def tap(_m,args): return (args[0]+.05*pair,)
            hook=m.mod.pair_embed.embed.register_forward_pre_hook(tap)
            try: after=margin(m(raw['features']+.05*pf.transpose(1,2),raw['vectors'],raw['mask']),raw['labels'])
            finally: hook.remove()
        if not torch.isfinite(after).all(): raise ValueError('Nonfinite finite-response diagnostic')
        return float((after-before).abs().mean())
    return dict(reference=response(reference), diverse=response(model), step_standard_deviations=.05,
                interpretation='transformed_coordinate_intervention_not_detector_perturbation')
