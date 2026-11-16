"""FP32 native forward, chunked decoder VJP, sample-weighted gradient SUM."""
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
from .optimization import optimizer
from . import optimization as optim, decoder
from .contracts import TRAIN_ROWS
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
    opt.stability_state["decoder_rng"]={}
    opt.stability_state["decoder_seed"]=seeds(spec,task)["native_initialization"]+7000000007
    if cell(task)['soft_assignment']:
        from .calibration import load
        opt.stability_state['soft_scale']=load(spec)['scale']
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
    with torch.autocast(torch.device(device).type,enabled=False):
        return model(**{k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')}).float()


def update(model,opt,batch,task,rank,world,device,position,*,probe=False,measure=False):
    return update_group(model,opt,[batch],task,rank,world,device,position,probe=probe,measure=measure)


def update_group(model,opt,batches,task,rank,world,device,position,*,probe=False,measure=False):
    """One optimizer boundary; every microbatch keeps the registered SyncBN128 population."""
    c=cell(task);total=sum(len(b[0]['features']) for b in batches)
    if not batches or len(batches)>c['accumulation'] or any(len(b[0]['features'])>128 for b in batches):
        raise ValueError('Accumulation group differs from registered microbatch policy')
    if not hasattr(opt,'stability_state'):opt.stability_state={}
    state=opt.stability_state;model.train();model.capture=True;opt.zero_grad(set_to_none=True)
    optim.set_lr(opt,task,position,learning_rate(position))
    w=weights(task,position)[0];special=c['gradient_rule']!='ordinary'
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and optim.owner(n)!='decoder']
    shared=[i for i,(n,p) in enumerate(named) if optim.owner(n)=='encoder']
    params=[p for _,p in named];base_acc=[torch.zeros_like(p) for p in params] if special else None
    aux_acc=[torch.zeros_like(p) for p in params] if special else None
    used=torch.zeros(len(params),device=device,dtype=torch.int32) if special else None
    loss_values=torch.zeros(6,device=device,dtype=torch.float64);stats={};diagnostics={};extra_steps=0
    temperature=None
    if c['soft_assignment'] and 3.<position<9.:
        from .transport import temperature_at
        temperature=temperature_at(state['soft_scale'],position)
    before=[p.detach().clone() for i,p in enumerate(params) if i in shared] if probe or special else None
    rng=state.setdefault('decoder_rng',{});decoder_seed=state.get('decoder_seed',7000000007)
    for batch in batches:
        (raw,embedding,targets),_=shard(batch,rank,world);fraction=len(raw['features'])/total
        if embedding is None or (not c['control'] and targets is None):raise ValueError('Missing TRAIN targets')
        logits=native(model,raw,device);target=torch.as_tensor(embedding,device=device).float().detach()
        if target.shape!=model.prediction.shape or not torch.isfinite(target).all():raise ValueError('Invalid U100 target')
        ce=F.cross_entropy(logits,torch.as_tensor(raw['labels'],device=device).long())
        emb=F.smooth_l1_loss(model.prediction,target);base=(ce+5*emb)*fraction
        # DEC2 has its own decoder parameter groups. AdamW only advances states
        # with a gradient; all native/predictor gradients remain None here.
        if c['decoder_extra_steps'] and 3.<position<=9.:
            if len(batches)!=1:raise ValueError('DEC2 accumulation is not registered')
            for _ in range(c['decoder_extra_steps']):
                opt.zero_grad(set_to_none=True)
                with optim.isolated_rng(device,rng,decoder_seed):
                    decoder.backward(model.aux,model.jet,model.particle_tokens,model.particle_mask,targets,w*fraction)
                reduce_gradients(model,world)
                if any(p.grad is not None for p in params):raise ValueError('Decoder-only update touched native state')
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite decoder gradient')
                opt.step();extra_steps+=1
            opt.zero_grad(set_to_none=True)
        aux=0.;gh=gt=None
        if not c['control']:
            # Extract unweighted condition VJPs for interventions. Decoder receives
            # its original coefficient even when the encoder gradient is changed.
            previous=[None if p.grad is None else p.grad.clone() for p in model.aux.parameters()] if special else None
            with optim.isolated_rng(device,rng,decoder_seed):
                aux,parts,gh,gt,_=decoder.backward(model.aux,model.jet,model.particle_tokens,
                    model.particle_mask,targets,fraction if special else w*fraction,temperature=temperature)
            if special:
                for p,old in zip(model.aux.parameters(),previous):
                    if p.grad is not None:p.grad.mul_(w)
                    if old is not None:p.grad.add_(old,alpha=1.-w)
            for key,value in parts.items():
                stats[key]=max(stats.get(key,0.),value) if key.endswith('_max') else stats.get(key,0.)+value*fraction
        if not torch.isfinite(base) or not np.isfinite(aux):raise ValueError('Nonfinite objective')
        if special:
            bg=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            outputs=[model.jet,model.particle_tokens];vjp=[gh,gt]
            ag=torch.autograd.grad(outputs,params,grad_outputs=vjp,allow_unused=True)
            for i,(b,a) in enumerate(zip(bg,ag)):
                if b is not None or a is not None:used[i]=1
                if b is not None:base_acc[i].add_(b)
                if a is not None:aux_acc[i].add_(a)
        else:
            outputs=[base];gradients=[torch.ones_like(base)]
            if gh is not None:outputs.extend([model.jet,model.particle_tokens]);gradients.extend([gh,gt])
            torch.autograd.backward(outputs,gradients)
        loss_values+=torch.tensor([float(ce.detach()),float(emb.detach()),aux,5*float(emb.detach()),w*aux,w*aux],device=device,dtype=torch.float64)*fraction
        clear(model)
    if special:
        optim.reduce_values(base_acc,world);optim.reduce_values(aux_acc,world)
        if world>1:dist.all_reduce(used,op=dist.ReduceOp.MAX)
        transformed,diagnostics=optim.transform([base_acc[i] for i in shared],[aux_acc[i] for i in shared],
            w,c['gradient_rule'],state.setdefault('gradient_statistics',{}),total)
        # Reduce decoder gradients only: native arrays are already global.
        reduce_gradients(model,world)
        for i,p in enumerate(params):p.grad=base_acc[i]+w*aux_acc[i] if used[i] else None
        for i,g in zip(shared,transformed):
            if used[i]:params[i].grad=g
    else:reduce_gradients(model,world)
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
    opt.step()
    if before is not None:
        diagnostics['encoder_parameter_update_norm']=float(torch.sqrt(sum((
            (params[i].detach().double()-old.double()).square().sum() for i,old in zip(shared,before)),torch.zeros((),device=device,dtype=torch.float64))))
    diagnostics.update(encoder_weight=w,decoder_weight=w,extra_decoder_steps=extra_steps,examples=total,
        optimizer_updates=1,temperature=temperature,encoder_lr_multiplier=optim.encoder_multiplier(task,position))
    if world>1:dist.all_reduce(loss_values)
    names=('ce','embedding','reconstruction','weighted_embedding','encoder_weighted_reconstruction','decoder_weighted_reconstruction')
    return dict(zip(names,loss_values.cpu().tolist())),dict(probe_seconds=0.,diagnostics=diagnostics,reconstruction_losses=stats)


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


def clock(call,device,world):
    if world>1: dist.barrier()
    torch.cuda.synchronize(device); start=time.perf_counter(); result=call(); torch.cuda.synchronize(device)
    elapsed=torch.tensor(time.perf_counter()-start,device=device,dtype=torch.float64)
    if world>1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    return float(elapsed),result
