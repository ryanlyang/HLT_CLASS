"""Single-device FP32 CE/U100 objectives and chunked R24 decoder VJP."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_next_supervision_hydra.model import Student
from .contracts import cell, weights, scientific_cfg, seeds
from hlt_classification.luka_next_supervision_hydra import reconstruction_model
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as batched
from .optimization import optimizer
from . import optimization as optim, decoder
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module,nn.Dropout): module.p=0.
        if isinstance(getattr(module,'dropout',None),(float,int)): module.dropout=0.


def clear(model):
    for name in ('prediction','jet','ce_representation','particle_tokens','particle_mask'):
        setattr(model,name,None)


def build(spec,task,rank,world,device,*,parity=False,fresh=False):
    if rank!=0 or world!=1:raise ValueError('This campaign uses one GPU per fit')
    torch.manual_seed(seeds(spec,task)['native_initialization'])
    model=Student('C0' if cell(task)['control'] else 'R24');model.cfg=scientific_cfg(task)
    if not cell(task)['control']:reconstruction_model.attach(model,spec['maximum'])
    if not fresh:
        from .identity import load_start
        load_start(spec,task,model)
    if parity:disable_dropout(model)
    model=model.to(device);opt=optimizer(model)
    opt.stability_state.update(decoder_rng={},decoder_seed=seeds(spec,task)['native_initialization']+7000000007)
    torch.manual_seed(seeds(spec,task)['dropout_base'])
    return model,opt


def configure(world,cpus):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    # Process-local runtime override; immutable donor sources are never edited.
    if batched.solver_pool.cache_info().currsize: batched.solver_pool().shutdown(wait=True)
    batched.ASSIGNMENT_WORKERS=cpus//world; batched.solver_pool.cache_clear()


def native(model,raw,device):
    if set(raw)-{'features','vectors','mask','identities','labels'}: raise PermissionError('Unexpected HLT input')
    with torch.autocast(torch.device(device).type,enabled=False):
        return model(**{k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')}).float()


def update(model,opt,batch,task,rank,world,device,position,*,probe=False,measure=False):
    return update_group(model,opt,[batch],task,rank,world,device,position,probe=probe,measure=measure)


def update_group(model,opt,batches,task,rank,world,device,position,*,probe=False,measure=False):
    if rank!=0 or world!=1 or len(batches)!=1:raise ValueError('One physical batch on one GPU is required')
    raw,embedding,targets=batches[0];c=cell(task);n=len(raw['features'])
    if not 0<n<=128:raise ValueError('Invalid physical batch')
    if c['embedding_weight']==0 and (embedding is not None or targets is not None):
        raise PermissionError('CE-only updates must not receive privileged targets')
    if c['control'] and targets is not None:raise PermissionError('Reconstruction targets only belong to R24')
    model.train();model.capture=True;opt.zero_grad(set_to_none=True)
    optim.set_lr(opt,task,position,learning_rate(position))
    encoder=[p for name,p in model.named_parameters() if optim.owner(name)=='encoder']
    before=[p.detach().clone() for p in encoder] if probe else None
    logits=native(model,raw,device)
    ce=F.cross_entropy(logits,torch.as_tensor(raw['labels'],device=device).long())
    emb=ce.new_zeros(());base=ce
    if c['embedding_weight']:
        if embedding is None:raise ValueError('Missing U100 TRAIN target')
        target=torch.as_tensor(embedding,device=device).float().detach()
        if target.shape!=model.prediction.shape or not torch.isfinite(target).all():raise ValueError('Invalid U100 target')
        emb=F.smooth_l1_loss(model.prediction,target);base=ce+c['embedding_weight']*emb
    w=weights(task,position)[0];aux=0.;parts={};gh=gt=None
    if not c['control']:
        if targets is None:raise ValueError('Missing full OFFLINE TRAIN targets')
        state=opt.stability_state
        with optim.isolated_rng(device,state['decoder_rng'],state['decoder_seed']):
            aux,parts,gh,gt,_=decoder.backward(model.aux,model.jet,model.particle_tokens,model.particle_mask,targets,w)
    if not torch.isfinite(base) or not np.isfinite(aux):raise ValueError('Nonfinite objective')
    outputs=[base];gradients=[torch.ones_like(base)]
    if gh is not None:outputs.extend([model.jet,model.particle_tokens]);gradients.extend([gh,gt])
    torch.autograd.backward(outputs,gradients)
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
    if not c['embedding_weight'] and any(p.grad is not None for p in model.predictor.parameters()):
        raise ValueError('CE-only predictor unexpectedly received gradients')
    opt.step()
    diagnostics=dict(encoder_weight=w,decoder_weight=w,extra_decoder_steps=0,examples=n,
        optimizer_updates=1,encoder_lr_multiplier=1.)
    if before is not None:
        diagnostics['encoder_parameter_update_norm']=float(torch.sqrt(sum(
            ((p.detach().double()-v.double()).square().sum() for p,v in zip(encoder,before)),
            torch.zeros((),device=device,dtype=torch.float64))))
    losses=dict(ce=float(ce.detach()),embedding=float(emb.detach()),reconstruction=aux,
        weighted_embedding=c['embedding_weight']*float(emb.detach()),
        encoder_weighted_reconstruction=w*aux,decoder_weighted_reconstruction=w*aux)
    clear(model)
    return losses,dict(probe_seconds=0.,diagnostics=diagnostics,reconstruction_losses=parts)


def snapshot(model,opt,loss):
    ps=list(model.parameters())
    return dict(loss=loss,parameters=torch.cat([p.detach().cpu().flatten() for p in ps]),
        gradients=torch.cat([(p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device='cpu')).flatten() for p in ps]),
        buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()},
        moments=torch.cat([opt.state[p][k].detach().cpu().flatten() for p in ps if p in opt.state for k in ('exp_avg','exp_avg_sq')]))


def qualification_state(model,opt):
    """CPU copies of the complete start state, including Adam steps and BN counters."""
    def copy(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {key:copy(item) for key,item in value.items()}
        if isinstance(value,list):return [copy(item) for item in value]
        if isinstance(value,tuple):return tuple(copy(item) for item in value)
        return value
    return copy(dict(model=model.state_dict(),optimizer=opt.state_dict(),runtime=optim.runtime_state(opt)))


def restore_qualification_state(model,opt,state):
    if set(state)!={'model','optimizer','runtime'}:raise ValueError('Incomplete qualification start state')
    model.load_state_dict(state['model'],strict=True)
    opt.load_state_dict(deepcopy(state['optimizer']))
    opt.stability_state=deepcopy(state['runtime'])
    opt.zero_grad(set_to_none=True);clear(model)


def buffer_differences(reference,actual):
    if reference.keys()!=actual.keys():raise ValueError('Buffer inventory differs')
    rows={}
    for name,ref in reference.items():
        other=actual[name]
        if ref.shape!=other.shape or ref.dtype!=other.dtype:raise ValueError('Buffer shape/dtype differs: '+name)
        if not torch.isfinite(ref).all() or not torch.isfinite(other).all():raise ValueError('Nonfinite buffer: '+name)
        a,b=ref.double(),other.double()
        rows[name]=dict(absolute=float((a-b).abs().max()) if a.numel() else 0.,
            relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
            reference_abs_max=float(a.abs().max()) if a.numel() else 0.,
            actual_abs_max=float(b.abs().max()) if b.numel() else 0.,
            dtype=str(ref.dtype),shape=list(ref.shape),
            integer_equal=True if ref.is_floating_point() else torch.equal(ref,other))
    return rows


def compare(ref,actual,tolerances):
    def relative(a,b):
        if a.shape!=b.shape: raise ValueError('Snapshot shape differs')
        return float((a.double()-b.double()).norm()/a.double().norm().clamp_min(1e-12))
    buffers=buffer_differences(ref['buffers'],actual['buffers'])
    result=dict(loss_abs=abs(ref['loss']-actual['loss']),
        gradient_relative_l2=relative(ref['gradients'],actual['gradients']),
        parameters_relative_l2=relative(ref['parameters'],actual['parameters']),
        adamw_moments_relative_l2=relative(ref['moments'],actual['moments']),
        buffer_abs=max((row['absolute'] for row in buffers.values()),default=0.))
    if not np.isfinite(list(result.values())).all(): raise ValueError('Nonfinite parity metrics')
    return dict(result,buffers=buffers,
        passed=all(result[k]<=v for k,v in tolerances.items()) and all(row['integer_equal'] for row in buffers.values()))
