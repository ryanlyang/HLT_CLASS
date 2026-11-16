"""Parameter ownership, the original AdamW recipe and isolated RNG."""
from contextlib import contextmanager
from copy import deepcopy
import torch
from hlt_classification.luka_cross_message_hydra.contracts import recipe
from .contracts import encoder_multiplier


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
