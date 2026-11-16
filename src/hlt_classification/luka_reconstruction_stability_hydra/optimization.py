"""Parameter ownership, globally reduced auxiliary transforms and isolated RNG."""
from contextlib import contextmanager
from copy import deepcopy
import math
import torch
import torch.distributed as dist
from hlt_classification.luka_cross_message_hydra.contracts import recipe
from .contracts import encoder_multiplier, TRAIN_ROWS


def owner(name):
    if name.startswith('mod.fc.'): return 'classifier'
    if name.startswith('mod.'): return 'encoder'
    if name.startswith('predictor.'): return 'predictor'
    if name.startswith('aux.'): return 'decoder'
    raise ValueError('Undeclared parameter ownership: '+name)


def optimizer(model):
    config=recipe()['base'];excluded=model.no_weight_decay();groups=[];seen=set()
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if len({id(p) for _,p in named})!=len(named):raise ValueError('Duplicate parameter')
    for group in ('encoder','classifier','predictor','decoder'):
        for omit in (False,True):
            items=[(n,p) for n,p in named if owner(n)==group and (n in excluded)==omit]
            if not items:continue
            groups.append(dict(params=[p for _,p in items],names=[n for n,_ in items],owner=group,
                weight_decay=0. if omit else config['weight_decay']))
            seen.update(n for n,_ in items)
    if seen!={n for n,_ in named} or not any(g['owner']=='encoder' for g in groups):
        raise ValueError('Incomplete parameter partition')
    opt=torch.optim.AdamW(groups,lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'])
    opt.stability_state={}
    return opt


def set_lr(opt,task,position,lr):
    for group in opt.param_groups:
        group['lr']=lr*(encoder_multiplier(task,position) if group['owner']=='encoder' else 1.)


def reduce_values(values,world):
    """SUM already sample-weighted gradients, bounded transfers and matching inventory."""
    if world==1:return values
    buffers=[];size=0;limit=1048576
    def flush():
        if not buffers:return
        flat=torch.cat(buffers);dist.all_reduce(flat)
        offset=0
        for part in buffers:
            part.copy_(flat[offset:offset+part.numel()]);offset+=part.numel()
    for tensor in values:
        if tensor is not None:
            for part in tensor.view(-1).split(limit):
                if size+part.numel()>limit:flush();buffers=[];size=0
                buffers.append(part);size+=part.numel()
    flush()
    return values


def transform(base,aux,weight,rule,state,examples):
    if len(base)!=len(aux) or not base:raise ValueError('Shared gradient inventory differs')
    device=base[0].device
    norm=lambda xs:sum((v.double().square().sum() for v in xs),torch.zeros((),device=device,dtype=torch.float64))
    b2=norm(base);a2=norm(aux);dot=sum((b.double().mul(a.double()).sum() for b,a in zip(base,aux)),b2.new_zeros(()))
    if not torch.isfinite(torch.stack((b2,a2,dot))).all():raise ValueError('Nonfinite global gradient statistics')
    bn,an=float(b2.sqrt()),float(a2.sqrt());alpha=weight;projected=False;removed=0.
    if rule=='G30' and weight>0:
        decay=2.**(-examples/(.1*TRAIN_ROWS))
        if 'base_norm_ema' not in state:state.update(base_norm_ema=bn,aux_norm_ema=an)
        else:
            state['base_norm_ema']=decay*state['base_norm_ema']+(1-decay)*bn
            state['aux_norm_ema']=decay*state['aux_norm_ema']+(1-decay)*an
        alpha=0. if bn==0. else min(weight,.3*state['base_norm_ema']/(state['aux_norm_ema']+1e-12),.3*bn/(an+1e-12))
    elif rule=='PROJ' and weight>0 and float(b2)>1e-20 and float(dot)<0:
        factor=float(dot/b2);aux=[a-factor*b for a,b in zip(aux,base)];projected=True
        removed=abs(factor)*bn/(an+1e-12)
    elif rule not in ('ordinary','G30','PROJ'):raise ValueError('Unknown gradient rule')
    result=[b+alpha*a for b,a in zip(base,aux)]
    post_dot=float(sum((b.double().mul(a.double()).sum() for b,a in zip(base,aux)),b2.new_zeros(())))
    return result,dict(base_gradient_norm=bn,auxiliary_gradient_norm=an,gradient_dot=float(dot),
        post_projection_gradient_dot=post_dot,injected_gradient_dot=alpha*post_dot,
        base_auxiliary_cosine=float(dot)/(bn*an+1e-12),
        encoder_coefficient=alpha,capped=alpha<weight,projected=projected,removed_fraction=removed,
        injected_to_base_norm=alpha*float(norm(aux).sqrt())/(bn+1e-12),**state)


@contextmanager
def isolated_rng(device,state=None,seed=None):
    """Run an independent stream and restore the native training stream exactly."""
    device=torch.device(device);devices=[device] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        if state and 'cpu' in state:
            torch.set_rng_state(state['cpu'])
            if devices:torch.cuda.set_rng_state(state['cuda'],device)
        elif seed is not None:
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):torch.cuda.manual_seed(seed)
        try:yield
        finally:
            if state is not None:
                state['cpu']=torch.get_rng_state().clone()
                if devices:state['cuda']=torch.cuda.get_rng_state(device).clone()


def runtime_state(opt):return deepcopy(getattr(opt,'stability_state',{}))
