"""Declared FP32/BF16 native forward, chunked decoder VJP, then sample-weighted gradient SUM."""
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from hlt_classification.luka_next_supervision_hydra.model import Student
from hlt_classification.luka_next_supervision_hydra.contracts import seed
from .contracts import cell, weights, scientific_cfg, seeds
from hlt_classification.luka_next_supervision_hydra import reconstruction_model, reconstruction_training as rt
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as batched
from hlt_classification.luka_cross_message_hydra.training import optimizer
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate


def shard(batch,rank,world):
    raw,emb,targets=batch; n=len(raw['features'])
    if not 0<=rank<world<=n: raise ValueError('Nonempty local shards required')
    ix=np.array_split(np.arange(n),world)[rank]
    return ({k:v[ix] for k,v in raw.items()},None if emb is None else emb[ix],
            None if targets is None else [targets[i] for i in ix]),len(ix)/n


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module,nn.Dropout): module.p=0.
        if isinstance(getattr(module,'dropout',None),(float,int)): module.dropout=0.


def clear(model):
    for name in ('prediction','jet','ce_representation','particle_tokens','particle_mask'):
        setattr(model,name,None)


def build(spec,task,rank,world,device,*,parity=False):
    cfg=scientific_cfg(task)
    torch.manual_seed(seeds(spec,task)['native_initialization'])
    model=Student('C0' if cell(task)['control'] else 'R24');model.cfg=cfg
    if not cell(task)['control']: reconstruction_model.attach(model,spec['maximum'])
    if parity: disable_dropout(model)
    if world>1: model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model=model.to(device); opt=optimizer(model)
    model._diagnostic_precision=cell(task)['precision']
    torch.manual_seed(seeds(spec,task)['dropout_base']+rank)
    return model,opt


def configure(world,cpus):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    # Process-local runtime override; immutable donor sources are never edited.
    if batched.solver_pool.cache_info().currsize: batched.solver_pool().shutdown(wait=True)
    batched.ASSIGNMENT_WORKERS=cpus//world; batched.solver_pool.cache_clear()


def reduce_gradients(model,world,bucket_elements=1048576):
    if world==1: return
    parameters=[p for p in model.parameters() if p.requires_grad]
    presence=torch.tensor([p.grad is not None for p in parameters],device=parameters[0].device,dtype=torch.int32)
    dist.all_reduce(presence)
    if any(x not in (0,world) for x in presence.tolist()): raise ValueError('Rank gradient inventories differ')
    buffers=[]; size=0
    def flush():
        if not buffers: return
        flat=torch.cat([g.reshape(-1) for g in buffers]); dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        start=0
        for g in buffers: g.copy_(flat[start:start+g.numel()].view_as(g)); start+=g.numel()
    for p in parameters:
        if p.grad is None: continue
        # Split large parameters as well: the bucket is an actual memory bound.
        for part in p.grad.view(-1).split(bucket_elements):
            if size+part.numel()>bucket_elements: flush(); buffers=[]; size=0
            buffers.append(part); size+=part.numel()
    flush()


def native(model,raw,device):
    if set(raw)-{'features','vectors','mask','identities','labels'}: raise PermissionError('Unexpected HLT input')
    bf16=getattr(model,'_diagnostic_precision','fp32')=='bf16_native'
    if bf16 and torch.device(device).type!='cuda': raise PermissionError('BF16-native arm requires CUDA')
    with torch.autocast(torch.device(device).type,enabled=bf16,dtype=torch.bfloat16):
        return model(**{k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')}).float()


def update(model,opt,batch,task,rank,world,device,position,*,probe=False,measure=False):
    (raw,embedding,targets),fraction=shard(batch,rank,world)
    if embedding is None: raise ValueError('TRAIN embedding targets required')
    is_control=cell(task)['control']
    if not is_control and targets is None:raise ValueError('TRAIN reconstruction targets required')
    model.train();model.capture=True;opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:group['lr']=learning_rate(position)
    logits=native(model,raw,device);target=torch.as_tensor(embedding,device=device).float().detach()
    if target.shape!=model.prediction.shape or not torch.isfinite(target).all():raise ValueError('Invalid U100 target')
    ce=F.cross_entropy(logits,torch.as_tensor(raw['labels'],device=device).long())
    emb=F.smooth_l1_loss(model.prediction,target);base=ce+5*emb
    encoder_weight,decoder_weight=weights(task,position)
    gh=gt=None;aux=0.;parts={};times={};diagnostics={};probe_seconds=0.
    if not is_control:
        aux,parts,gh,gt,times=rt.decode_backward(model.aux,model.jet,model.particle_tokens,model.particle_mask,
            targets,decoder_weight*fraction,consistency=False,measure=measure)
        # Decoder gradients are already accumulated. Scale only the VJPs injected
        # into the native encoder; this is intentional asymmetric training.
        ratio=encoder_weight/decoder_weight if decoder_weight else 0.
        gh=gh*ratio;gt=None if gt is None else gt*ratio
    if probe:
        if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
        began=time.perf_counter()
        if not is_control:diagnostics=rt.probes(model,targets,raw,encoder_weight,gh/fraction,None if gt is None else gt/fraction)
        for name,term,condition in (('ce',ce,model.ce_representation),('u100_w5',5*emb,model.jet)):
            g=torch.autograd.grad(term,condition,retain_graph=True)[0].detach().float()
            diagnostics[name+'_jet_gradient_norm']=float(g.norm())
            if gh is not None:diagnostics[name+'_aux_jet_gradient_cosine']=float(F.cosine_similarity(g.flatten(),(gh/fraction).flatten(),dim=0,eps=1e-12))
        diagnostics['decoder_parameter_gradient_norm']=float(torch.sqrt(sum((p.grad.detach().double().square().sum() for p in model.aux.parameters() if p.grad is not None),torch.zeros((),device=device,dtype=torch.float64)))/fraction) if model.aux is not None else 0.
        diagnostics['encoder_weight']=encoder_weight;diagnostics['decoder_weight']=decoder_weight
        if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
        probe_seconds=time.perf_counter()-began
    if not torch.isfinite(base) or not np.isfinite(aux):raise ValueError('Nonfinite objective')
    outputs=[base*fraction];gradients=[torch.ones_like(base)]
    if gh is not None:outputs.append(model.jet);gradients.append(gh)
    if gt is not None:outputs.append(model.particle_tokens);gradients.append(gt)
    torch.autograd.backward(outputs,gradients);reduce_gradients(model,world)
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
    opt.step()
    names=('ce','embedding','reconstruction','weighted_embedding','encoder_weighted_reconstruction','decoder_weighted_reconstruction')
    values=torch.tensor([float(ce.detach()),float(emb.detach()),aux,5*float(emb.detach()),encoder_weight*aux,decoder_weight*aux],device=device,dtype=torch.float64)*fraction
    if world>1:dist.all_reduce(values)
    losses=dict(zip(names,values.cpu().tolist()));clear(model)
    return losses,dict(probe_seconds=probe_seconds,diagnostics=diagnostics,decoder_timing=times,
        reconstruction_losses=parts,encoder_weight=encoder_weight,decoder_weight=decoder_weight)


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
    return copy(dict(model=model.state_dict(),optimizer=opt.state_dict()))


def restore_qualification_state(model,opt,state):
    if set(state)!={'model','optimizer'}:raise ValueError('Incomplete qualification start state')
    model.load_state_dict(state['model'],strict=True)
    opt.load_state_dict(deepcopy(state['optimizer']))
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


def clock(call,device,world):
    if world>1: dist.barrier()
    torch.cuda.synchronize(device); start=time.perf_counter(); result=call(); torch.cuda.synchronize(device)
    elapsed=torch.tensor(time.perf_counter()-start,device=device,dtype=torch.float64)
    if world>1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    return float(elapsed),result
