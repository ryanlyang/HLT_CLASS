"""Single GPU versus DDP/SyncBatchNorm, with global batch and padding preserved."""
from datetime import timedelta
from pathlib import Path
import gc
import math
import os
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from . import benchmark as b, model, losses


class Outputs(nn.Module):
    """Expose all auxiliary outputs to DDP; labels never enter the forward pass."""
    def __init__(self,student):
        super().__init__(); self.student=student

    def forward(self,features,vectors,mask):
        self.student.capture=self.training
        logits=self.student(features,vectors,mask)
        return logits,self.student.prediction,self.student.subclass_logits


def shard(raw,rank,world):
    n=len(raw['labels'])
    if not 0<=rank<world<=n: raise ValueError('Nonempty shards required')
    ix=np.array_split(np.arange(n),world)[rank]
    # Slice only the jet dimension: identical global sequence padding is vital
    # for the input and pairwise BatchNorm statistics.
    return {k:v[ix] for k,v in raw.items()},len(ix)*world/n


def disable_dropout(student):
    for module in student.modules():
        if isinstance(module,nn.Dropout): module.p=0.
        # Installed Weaver attention can store dropout as a float rather than
        # an nn.Dropout child. Do not replace actual module-valued attributes.
        if isinstance(getattr(module,'dropout',None),(float,int)): module.dropout=0.


def build(spec,task,rank,world,device,*,parity=False):
    student=model.new_model(spec,task)
    if parity: disable_dropout(student)
    if world>1: student=nn.SyncBatchNorm.convert_sync_batchnorm(student)
    student=student.to(device); opt=optimizer(student)
    for group in opt.param_groups: group['lr']=learning_rate(3.)
    wrapped=Outputs(student)
    if world>1: wrapped=DDP(wrapped,device_ids=[rank],output_device=rank,broadcast_buffers=True)
    torch.manual_seed(spec['nodes'][task]['initialization_seed']+rank)
    return student,wrapped,opt


def update(wrapped,student,opt,raw,task,rank,world,device):
    local,weight=shard(raw,rank,world)
    tensors={k:torch.from_numpy(local[k]).to(device) for k in ('features','vectors','mask','labels','u100','subclass')}
    wrapped.train(); opt.zero_grad(set_to_none=True)
    logits,prediction,subclass=wrapped(**{k:tensors[k] for k in ('features','vectors','mask')})
    total,parts=losses.objective(task,logits,tensors['labels'],prediction,tensors['u100'],subclass,tensors['subclass'],3.)
    (total*weight).backward()
    if not all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None):
        raise ValueError('Nonfinite FP32 gradients')
    opt.step()
    scalar=total.detach()*len(local['labels'])/len(raw['labels'])
    if world>1: dist.all_reduce(scalar)
    student.clear()
    return float(scalar)


def relative_l2(a,b):
    return float((a.double()-b.double()).norm()/a.double().norm().clamp_min(1e-12))


def compare(reference,actual,tolerances):
    if set(reference['buffers'])!=set(actual['buffers']): raise ValueError('Buffer inventory differs')
    buffer_abs=max((float((v.double()-actual['buffers'][k].double()).abs().max())
                    for k,v in reference['buffers'].items() if v.numel()),default=0.)
    loss_abs=abs(reference['loss']-actual['loss'])
    grad=relative_l2(reference['gradients'],actual['gradients'])
    result=dict(loss_abs=loss_abs,gradient_relative_l2=grad,buffer_abs=buffer_abs,
        parameters_relative_l2=relative_l2(reference['parameters'],actual['parameters']),
        adamw_moments_relative_l2=relative_l2(reference['moments'],actual['moments']))
    if not np.isfinite(list(result.values())).all(): raise ValueError('Nonfinite parity diagnostics')
    result['passed']=all(result[k]<=limit for k,limit in tolerances.items())
    return result


def parity_snapshot(student,opt,loss):
    parameters=list(student.parameters())
    return dict(loss=loss,parameters=torch.cat([p.detach().cpu().flatten() for p in parameters]),
        gradients=torch.cat([(p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device='cpu')).flatten()
                             for p in parameters]),
        buffers={k:v.detach().cpu().clone() for k,v in student.named_buffers()},
        moments=torch.cat([opt.state[p][key].detach().cpu().flatten() for p in parameters if p in opt.state
                           for key in ('exp_avg','exp_avg_sq')]))


def parity(spec,batches,rank,world,device):
    root=Path(spec['root']); entries={}; comparisons={}
    for n,task in zip(spec['parity_batches'],spec['tasks']):
        student,wrapped,opt=build(spec,task,rank,world,device,parity=True)
        value=update(wrapped,student,opt,batches[f'parity{n}'],task,rank,world,device)
        if rank==0:
            snapshot=parity_snapshot(student,opt,value); name=f'parity{n}_{task}'
            if world==1:
                path=root/(name+'.pt'); save_state(path,snapshot); entries[name]=file_ref(path,root=root)
            else:
                m=load_json(root/'parity_reference.json'); b.validate(m,'PARITY',dict(spec=spec['content_hash']))
                ref=m['snapshots'][name]
                if ref['path']!=name+'.pt': raise ValueError('Parity snapshot ownership differs')
                reference=torch.load(validate_file_ref(ref,root=root),map_location='cpu',weights_only=True)
                comparisons[name]=compare(reference,snapshot,spec['parity_tolerances'])
        if world>1: dist.barrier()
        del student,wrapped,opt; gc.collect(); torch.cuda.empty_cache()
    if rank==0 and world==1:
        write_immutable_json(root/'parity_reference.json',b.artifact('PARITY',parents=dict(spec=spec['content_hash']),snapshots=entries,
            dropout_disabled=True,training_batchnorm=True,partial_shards_sample_weighted=True))
    return comparisons


def clock_step(call,device,world):
    if world>1: dist.barrier()
    torch.cuda.synchronize(device); start=time.perf_counter(); result=call()
    torch.cuda.synchronize(device); elapsed=torch.tensor(time.perf_counter()-start,device=device,dtype=torch.float64)
    if world>1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    return float(elapsed),result


def measure_task(spec,task,batches,rank,world,device):
    student,wrapped,opt=build(spec,task,rank,world,device)
    torch.cuda.reset_peak_memory_stats(device)
    step=lambda raw:update(wrapped,student,opt,raw,task,rank,world,device)
    for i in range(spec['warmup_batches']): step(batches[f'train_{i:02}'])
    samples=[]
    for i in range(spec['warmup_batches'],spec['warmup_batches']+spec['measured_batches']):
        seconds,value=clock_step(lambda:step(batches[f'train_{i:02}']),device,world)
        samples.append(seconds)
        if rank==0: print(f'GPU_BENCH world={world} {task} batch={i+1} step={seconds:.3f}s loss={value:.6f}',flush=True)
    step(batches['partial11']); step(batches['tail'])
    wrapped.eval()
    def validation():
        with torch.no_grad():
            for i in range(spec['validation_batches']):
                raw,_=shard(batches[f'val_{i:02}'],rank,world)
                logits,_,_=wrapped(**{k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')})
                if not torch.isfinite(logits).all(): raise ValueError('Nonfinite validation forward')
    validation_seconds,_=clock_step(validation,device,world)
    longest=True
    try: step(batches['longest'])
    except torch.cuda.OutOfMemoryError:
        # A single-card OOM is useful benchmark evidence. Multi-rank failures
        # must terminate the process group instead of stranding a collective.
        if world>1: raise
        longest=False
    peaks=[None]*world; peak=torch.cuda.max_memory_allocated(device)/2**30
    if world>1: dist.all_gather_object(peaks,peak)
    else: peaks=[peak]
    row=dict(step_seconds=samples,mean_step_seconds=float(np.mean(samples)),peak_gpu_gib=peaks,
        longest_support_pass=longest,partial_batches=[11,len(batches['tail']['labels'])],
        validation_rows=128*spec['validation_batches'],validation_seconds=validation_seconds,
        train_only_30_epoch_hours=30*math.ceil(spec['counts']['train']/128)*float(np.mean(samples))/3600)
    del wrapped,student,opt; gc.collect(); torch.cuda.empty_cache()
    return row


def rank_main(spec):
    from . import benchmark_site, benchmark_submission
    rank=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    if (world not in (1,2) or not 0<=rank<world or int(os.environ.get('RANK','-1'))!=rank
            or int(os.environ.get('LOCAL_WORLD_SIZE','-1'))!=world or int(os.environ.get('GROUP_RANK','-1'))!=0):
        raise PermissionError('Only local one/two-GPU comparisons are registered')
    benchmark_submission.require_job(spec); devices=benchmark_site.allocation(spec)
    # The launcher authenticates the complete pinned source and parent before
    # entering torchrun. Every rank independently authenticates input payloads.
    m,batches=b.read_batches(spec)
    device=torch.device('cuda',rank); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    try:
        checks=parity(spec,batches,rank,world,device)
        rows={task:measure_task(spec,task,batches,rank,world,device) for task in spec['tasks']}
        if rank==0:
            result=b.artifact('PHASE',parents=dict(spec=spec['content_hash']),world_size=world,precision='fp32',
                batch_manifest=m['content_hash'],devices=devices[:world],global_batch=128,tasks=rows,parity=checks,
                dropout_enabled_for_timing=True,normalization='native' if world==1 else 'SyncBatchNorm')
            write_immutable_json(Path(spec['root'])/f'world{world}.json',result)
    finally:
        if world>1: dist.destroy_process_group()
