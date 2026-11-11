"""One global batch, two local ranks; observations never alter training state."""
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context, timed, memory, release_memory, merge_predictions
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from . import model, losses
from .registry import cell, teacher_condition, SCALES

KEYS=('features','vectors','mask')


def build(spec,task,ctx,*,parity=False):
    student=model.new_model(spec,task)
    if parity:model.disable_dropout(student)
    if ctx.world>1:student=nn.SyncBatchNorm.convert_sync_batchnorm(student)
    student=student.to(ctx.device);opt=optimizer(student)
    wrapped=DistributedDataParallel(student,device_ids=[ctx.rank] if ctx.device.type=='cuda' else None,
                                   broadcast_buffers=True) if ctx.world>1 else student
    torch.manual_seed(spec['nodes'][task]['initialization_seed']+ctx.rank)
    return student,wrapped,opt


def prepare(task,ds,banks,ix,epoch,batch,ctx,*,role='train'):
    if role!='train':raise PermissionError('Training iterator cannot read validation')
    c=cell(task);condition=None;crop_batches=None;valid=None
    if task=='TR':
        condition=teacher_condition(epoch,batch);raw=ds.condition(role,ix,condition)
    elif task=='TP':
        source,scale=teacher_condition(epoch,batch,regional=True)
        raw=ds.view(role,'D000' if source=='HLT' else 'OFFLINE').batch(ix)
        crop_batches,valid=ds.crops(role,ix,source,scale)
    else:raw=ds.view(role,c['input']).batch(ix)
    n=len(ix)
    if n<ctx.world:raise ValueError('Training requires nonempty local shards')
    local=np.array_split(np.arange(n),ctx.world)[ctx.rank];weight=len(local)*ctx.world/n
    tensors={k:torch.as_tensor(raw[k][local],device=ctx.device) for k in KEYS}
    target=banks.batch(ix[local])
    target={k:torch.as_tensor(v,device=ctx.device) if isinstance(v,np.ndarray) else v for k,v in target.items()}
    if task=='TR':target['condition']=condition
    if task=='TP':
        target['valid']=torch.as_tensor(valid[local],device=ctx.device)
        tensors.update(crops=[{k:torch.as_tensor(raw[k][local],device=ctx.device) for k in KEYS} for raw in crop_batches],valid=target['valid'])
    return tensors,torch.as_tensor(raw['labels'][local],device=ctx.device),target,weight


def probe(student,task,tensors,labels,target,position,weight,ctx):
    buffers={k:v.detach().clone() for k,v in student.named_buffers()};params=model.shared_parameters(student)
    devices=[ctx.device.index if ctx.device.index is not None else torch.cuda.current_device()] if ctx.device.type=='cuda' else []
    vectors={}
    try:
        with torch.random.fork_rng(devices=devices):
            _,parts=losses.objective(task,student(**tensors),labels,target,position)
            for key in ('ce','weighted_embedding','weighted_extra'):
                g=torch.autograd.grad(parts[key]*weight,params,retain_graph=True,allow_unused=True)
                flat=torch.cat([(torch.zeros_like(p) if grad is None else grad.detach()).flatten() for p,grad in zip(params,g)])
                ctx.average(flat);vectors[key]=flat.double()
    finally:
        with torch.no_grad():
            for k,v in student.named_buffers():v.copy_(buffers[k])
    norms={k:v.norm() for k,v in vectors.items()}
    result={k+'_norm':float(v) for k,v in norms.items()}
    for a,b in itertools.combinations(vectors,2):result[a+'__'+b+'_cosine']=float(vectors[a].dot(vectors[b])/(norms[a]*norms[b]).clamp_min(1e-30))
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite observation gradient')
    return result


def step(student,wrapped,opt,task,ds,banks,ix,epoch,batch,position,ctx,*,diagnose=False,coverage=None):
    wrapped.train();args,y,target,weight=prepare(task,ds,banks,ix,epoch,batch,ctx)
    for group in opt.param_groups:group['lr']=learning_rate(position)
    opt.zero_grad(set_to_none=True)
    diagnostic=probe(student,task,args,y,target,position,weight,ctx) if diagnose else None
    total,parts=losses.objective(task,wrapped(**args),y,target,position)
    (total*weight).backward()
    if not all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None):raise ValueError('Nonfinite required gradient')
    opt.step()
    values=torch.tensor([float(v.detach()) for v in parts.values()],device=ctx.device,dtype=torch.float64)*weight
    ctx.average(values)
    if coverage is not None:
        valid=target.get('valid',torch.ones((len(y),1),device=ctx.device,dtype=torch.bool))
        observed=torch.stack((valid.double().mean(),valid.any(-1).double().mean()))*weight
        ctx.average(observed)
        coverage.update(zip(('valid_query_fraction','valid_jet_fraction'),observed.cpu().tolist()))
    return dict(zip(parts,values.cpu().tolist())),diagnostic


def arrays(raw,ctx,indices=None):
    return {k:torch.as_tensor(raw[k] if indices is None else raw[k][indices],device=ctx.device) for k in KEYS}


def original_u100(student, view, ctx, indices=None):
    """Reproduce the frozen exporter batching, including original padding context.

    Assign whole consecutive batches to ranks. A panel row is evaluated with
    its original neighbours, not with other randomly selected panel rows.
    This is inference only: training still uses the registered split batch128.
    """
    ordered = np.arange(len(view), dtype=np.int64) if indices is None else np.asarray(indices)
    if (ordered.ndim != 1 or ordered.dtype.kind not in 'iu' or not len(ordered)
            or np.any(ordered < 0) or np.any(ordered >= len(view))
            or len(np.unique(ordered)) != len(ordered)):
        raise ValueError('Invalid frozen U100 extraction rows')
    positions = []; values = []; student.eval()
    with torch.inference_mode(), torch.autocast(ctx.device.type, enabled=False):
        for number, block in enumerate(np.unique(ordered // 128)):
            if number % ctx.world != ctx.rank:
                continue
            start = int(block) * 128
            ix = np.arange(start, min(start + 128, len(view)), dtype=np.int64)
            _, h = student.encode(**arrays(view.batch(ix), ctx))
            selected = np.flatnonzero(ordered // 128 == block)
            z = h[torch.as_tensor(ordered[selected] - start, device=ctx.device)].float()
            if z.shape != (len(selected), 128) or not torch.isfinite(z).all():
                raise ValueError('Invalid frozen U100 representation')
            positions.append(selected); values.append(z.cpu().numpy())
    local = (np.concatenate(positions) if positions else np.empty(0, np.int64),
             np.concatenate(values) if values else np.empty((0, 128), np.float32))
    shards = ctx.all(local); pos = np.concatenate([p for p, _ in shards])
    if not np.array_equal(np.sort(pos), np.arange(len(ordered))):
        raise ValueError('Frozen U100 extraction has missing or duplicate rows')
    return np.concatenate([v for _, v in shards])[np.argsort(pos)]


def predict(student,ds,role,ctx,*,coordinate='D000',condition=None,source=None,scale=None,indices=None,embedding=False,predicted=False):
    """Collect in ordered row positions. Evaluation has no DDP/SyncBN collectives."""
    if role not in ('train','validation'):raise PermissionError('No final-test inference')
    view=ds.view(role,'D000' if condition is not None or source=='HLT' else 'OFFLINE' if source=='OFFLINE' else coordinate)
    ordered=np.arange(len(view),dtype=np.int64) if indices is None else np.asarray(indices,np.int64)
    student.eval();local_positions=[];values=[];masks=[]
    with torch.inference_mode(),torch.autocast(ctx.device.type,enabled=False):
        for start in range(0,len(ordered),128):
            pos=np.arange(start,min(start+128,len(ordered)),dtype=np.int64);ix=ordered[pos]
            local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            if not len(local):continue
            raw=ds.condition(role,ix,condition) if condition is not None else view.batch(ix)
            if scale is None:
                logits,h=student.encode(**arrays(raw,ctx,local)) if hasattr(student,'encode') else (student(**arrays(raw,ctx,local)),None)
                z=student.predictor(h.float()) if predicted else h if embedding else logits.float().softmax(1)
                valid=np.ones(len(local),bool)
            else:
                batches,valid_all=ds.crops(role,ix,source,scale)
                zs=torch.stack([student.encode(**arrays(a,ctx,local))[1].float() for a in batches],1)
                valid=valid_all[local]
                if embedding:z=zs
                else:
                    mask=torch.as_tensor(valid,device=ctx.device)
                    avg=(zs*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
                    logits=student.mod.fc(avg)
                    # For an all-empty source follow the full-jet CE convention.
                    if not valid.all(1).all():
                        empty=~mask.any(1)
                        if empty.any():logits[empty]=student(**arrays(raw,ctx,local),auxiliary=False)[empty]
                    z=logits.float().softmax(1)
            if z is None or not torch.isfinite(z).all():raise ValueError('Nonfinite extracted representation or probability')
            local_positions.append(pos[local]);values.append(z.float().cpu().numpy());masks.append(valid)
    shape=(SCALES.index(scale) if scale else 0)
    width=128 if embedding or predicted else 11
    trailing=(2**(shape+1),width) if embedding and scale else (width,)
    vt=(2**(shape+1),) if scale else ()
    chunk=(np.concatenate(local_positions) if local_positions else np.empty(0,np.int64),
           np.concatenate(values) if values else np.empty((0,*trailing),np.float32),
           np.concatenate(masks) if masks else np.empty((0,*vt),bool))
    shards=ctx.all(chunk);positions=np.concatenate([a for a,_,_ in shards])
    if not np.array_equal(np.sort(positions),np.arange(len(ordered))):raise ValueError('Duplicate/missing distributed evaluation rows')
    order=np.argsort(positions)
    return np.concatenate([v for _,v,_ in shards])[order],np.concatenate([m for _,_,m in shards])[order]
