"""FP32 native forward, chunked decoder VJP, then sample-weighted gradient SUM."""
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from hlt_classification.luka_next_supervision_hydra.model import new_model
from hlt_classification.luka_next_supervision_hydra import reconstruction_model, reconstruction_training as rt
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as batched
from hlt_classification.luka_next_supervision_hydra.contracts import coefficient
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


def build(old,maximum,rank,world,device,*,parity=False):
    model=new_model(old['imported'],'R24'); reconstruction_model.attach(model,maximum)
    if parity: disable_dropout(model)
    if world>1: model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model=model.to(device); opt=optimizer(model)
    # Initialization is identical; only ordinary stochastic training differs by rank.
    torch.manual_seed(2026111201+rank)
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
    with torch.autocast(torch.device(device).type,enabled=False):
        return model(**{k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')}).float()


def update(model,opt,batch,rank,world,device,position,*,probe=False,measure=False):
    (raw,embedding,targets),fraction=shard(batch,rank,world)
    if embedding is None or targets is None: raise ValueError('TRAIN targets required')
    model.train(); model.capture=True; opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=learning_rate(position)
    logits=native(model,raw,device); target=torch.as_tensor(embedding,device=device).float().detach()
    if target.shape!=model.prediction.shape or not torch.isfinite(target).all(): raise ValueError('Invalid U100 target')
    ce=F.cross_entropy(logits,torch.as_tensor(raw['labels'],device=device).long())
    emb=F.smooth_l1_loss(model.prediction,target); base=ce+5*emb; weight=coefficient(model.cfg,position)
    aux,parts,gh,gt,times=rt.decode_backward(model.aux,model.jet,model.particle_tokens,model.particle_mask,
        targets,weight*fraction,consistency=model.cfg['consistency'],
        consistency_weight=model.cfg.get('consistency_weight',.1),measure=measure)
    diagnostics={}; probe_seconds=0.
    if probe:
        if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
        start=time.perf_counter()
        diagnostics=rt.probes(model,targets,raw,weight,gh/fraction,None if gt is None else gt/fraction)
        for name,term,condition in (('ce',ce,model.ce_representation),('u100_w5',5*emb,model.jet)):
            g=torch.autograd.grad(term,condition,retain_graph=True)[0].detach().float()
            diagnostics[name+'_jet_gradient_norm']=float(g.norm())
            diagnostics[name+'_aux_jet_gradient_cosine']=float(F.cosine_similarity(g.flatten(),(gh/fraction).flatten(),dim=0,eps=1e-12))
        if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
        probe_seconds=time.perf_counter()-start
    if not torch.isfinite(base) or not np.isfinite(aux): raise ValueError('Nonfinite objective')
    outputs=[base*fraction,model.jet]; gradients=[torch.ones_like(base),gh]
    if gt is not None: outputs.append(model.particle_tokens); gradients.append(gt)
    torch.autograd.backward(outputs,gradients)
    reduce_gradients(model,world)
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise ValueError('Nonfinite gradient')
    opt.step()
    scalar=(base.detach()+weight*aux)*fraction
    if world>1: dist.all_reduce(scalar)
    return float(scalar),dict(probe_seconds=probe_seconds,diagnostics=diagnostics,decoder_timing=times,
        reconstruction_losses=parts,weight=weight)


def snapshot(model,opt,loss):
    ps=list(model.parameters())
    return dict(loss=loss,parameters=torch.cat([p.detach().cpu().flatten() for p in ps]),
        gradients=torch.cat([(p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device='cpu')).flatten() for p in ps]),
        buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()},
        moments=torch.cat([opt.state[p][k].detach().cpu().flatten() for p in ps if p in opt.state for k in ('exp_avg','exp_avg_sq')]))


def compare(ref,actual,tolerances):
    def relative(a,b):
        if a.shape!=b.shape: raise ValueError('Snapshot shape differs')
        return float((a.double()-b.double()).norm()/a.double().norm().clamp_min(1e-12))
    if ref['buffers'].keys()!=actual['buffers'].keys(): raise ValueError('Buffer inventory differs')
    result=dict(loss_abs=abs(ref['loss']-actual['loss']),
        gradient_relative_l2=relative(ref['gradients'],actual['gradients']),
        parameters_relative_l2=relative(ref['parameters'],actual['parameters']),
        adamw_moments_relative_l2=relative(ref['moments'],actual['moments']),
        buffer_abs=max((float((v.double()-actual['buffers'][k].double()).abs().max()) for k,v in ref['buffers'].items() if v.numel()),default=0.))
    if not np.isfinite(list(result.values())).all(): raise ValueError('Nonfinite parity metrics')
    return dict(result,passed=all(result[k]<=v for k,v in tolerances.items()))


def clock(call,device,world):
    if world>1: dist.barrier()
    torch.cuda.synchronize(device); start=time.perf_counter(); result=call(); torch.cuda.synchronize(device)
    elapsed=torch.tensor(time.perf_counter()-start,device=device,dtype=torch.float64)
    if world>1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    return float(elapsed),result
