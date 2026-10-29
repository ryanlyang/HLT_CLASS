"""Five differentiable full-student AdamW steps; no persistent virtual mutations."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .model import objective, virtual_backend


def finite_gradients(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise ValueError('Missing or nonfinite required gradient')


def clone_optimizer(model, optimizer):
    """Detached starting moments and exact per-parameter AdamW configuration."""
    names = {id(p): n for n,p in model.named_parameters()}
    states, configs = {}, {}
    for group in optimizer.param_groups:
        if group.get('amsgrad') or group.get('maximize'):
            raise ValueError('Unregistered virtual optimizer')
        for p in group['params']:
            name = names[id(p)]; state = optimizer.state.get(p,{})
            states[name] = (int(state.get('step',0)),
                state.get('exp_avg',torch.zeros_like(p)).detach().clone(),
                state.get('exp_avg_sq',torch.zeros_like(p)).detach().clone())
            configs[name] = {k: group[k] for k in ('betas','eps','weight_decay')}
    return states, configs


def adam_step(params, grads, states, configs, lr):
    result, next_states = {}, {}
    for (name,p), g in zip(params.items(), grads):
        t,m,v = states[name]
        if g is None:
            result[name], next_states[name] = p, (t,m,v)
            continue
        c = configs[name]; b1,b2 = c['betas']; t += 1
        m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g.square()
        # Exact sqrt forward, finite derivative convention at v=0. Avoid 0*inf
        # in inactive parameters; no epsilon-inside-sqrt optimizer substitution.
        positive = v > 0
        sqrt_v = torch.where(positive, torch.where(positive,v,torch.ones_like(v)).sqrt(), torch.zeros_like(v))
        denominator = sqrt_v/(1-b2**t)**.5+c['eps']
        result[name] = p*(1-lr*c['weight_decay'])-(lr/(1-b1**t))*m/denominator
        next_states[name] = (t,m,v)
    return result, next_states


@contextmanager
def isolated(model, device, seed):
    modes = {m: m.training for m in model.modules()}
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), virtual_backend(), torch.autocast(device_type=device.type,enabled=False):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]): torch.cuda.manual_seed(seed)
            yield
    finally:
        for m, mode in modes.items(): m.training = mode


def virtual_query(model, optimizer, adapter, support, query, rates, *, seed):
    """Batches contain HLT tensors, labels and TRAIN target only for supports."""
    if len(support) != 5 or len(rates) != 5 or 'target' in query:
        raise PermissionError('Exactly five supports and HLT-only query required')
    device = next(model.parameters()).device
    params = {n:p.detach().clone().requires_grad_(True) for n,p in model.named_parameters()}
    buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    states, configs = clone_optimizer(model,optimizer)
    if params.keys() != states.keys():
        raise ValueError('Virtual update must cover the entire student')
    with isolated(model,device,seed):
        model.train()
        for batch,lr in zip(support,rates):
            # Clone between forwards: BN's in-place counters/stats must not
            # invalidate tensors saved by an earlier differentiable forward.
            buffers = {n:b.clone() for n,b in buffers.items()}
            logits,prediction = torch.func.functional_call(model,(params,buffers),(),
                dict(batch['inputs'],auxiliary=True), strict=True)
            loss,_,_ = objective(logits,batch['labels'],prediction,adapter(batch['target']),differentiable_target=True)
            grads = torch.autograd.grad(loss,tuple(params.values()),create_graph=True,allow_unused=True)
            params,states = adam_step(params,grads,states,configs,lr)
        model.eval()
        logits = torch.func.functional_call(model,(params,buffers),(),query['inputs'],strict=True)
        return F.cross_entropy(logits.float(),query['labels'])


def update(model, optimizer, adapter, adapter_optimizer, classifier, mean, scale, support, query, rates, *, seed):
    if any(p.requires_grad for p in classifier.parameters()):
        raise PermissionError('Teacher classifier must remain frozen')
    adapter_optimizer.zero_grad(set_to_none=True)
    query_ce = virtual_query(model,optimizer,adapter,support,query,rates,seed=seed)
    z = torch.cat([b['target'] for b in support])
    adapted = adapter(z)
    with torch.no_grad(): original = F.softmax(classifier(z*scale+mean),dim=-1)
    # This is teacher-side preservation, NOT logit KD into the HLT classifier.
    # Unlike another teacher CE objective, it is stationary at identity.
    teacher_kl = F.kl_div(F.log_softmax(classifier(adapted*scale+mean),dim=-1),original,
                         reduction='batchmean').clamp_min(0.)
    anchor = (adapted-z).square().mean()
    loss = query_ce+.1*teacher_kl+anchor
    if not torch.isfinite(loss): raise ValueError('Nonfinite adapter objective')
    params = tuple(adapter.parameters())
    grads = torch.autograd.grad(loss,params)
    for p,g in zip(params,grads): p.grad=g
    finite_gradients(params)
    norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    adapter_optimizer.step()
    return dict(query_ce=float(query_ce.detach()),teacher_kl=float(teacher_kl.detach()),
                anchor=float(anchor.detach()),total=float(loss.detach()),grad_norm=float(norm))
