"""Global batch semantics, observation-only probes and exact distributed evaluation."""
from dataclasses import dataclass
from pathlib import Path
import gc
import itertools
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from hlt_classification.luka_subclass_hydra.benchmark_training import shard
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from . import model, losses


@dataclass
class Context:
    rank: int
    world: int
    device: torch.device

    def sync(self):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)

    def barrier(self):
        if self.world>1: dist.barrier()

    def all(self,value):
        if self.world==1:return [value]
        values=[None]*self.world;dist.all_gather_object(values,value);return values

    def main(self,fn):
        result=None;error=None
        if self.rank==0:
            try:result=fn()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        objects=[error,result]
        if self.world>1:dist.broadcast_object_list(objects,src=0)
        if objects[0] is not None:raise RuntimeError(objects[0])
        return objects[1]

    def average(self,tensor):
        if self.world>1:dist.all_reduce(tensor);tensor.div_(self.world)
        return tensor


def release_memory(ctx):
    gc.collect()
    if ctx.device.type=='cuda':torch.cuda.empty_cache()


def build(spec,task,ctx,*,parity=False):
    student=model.new_model(spec,task)
    if parity:model.disable_dropout(student)
    if ctx.world>1:student=nn.SyncBatchNorm.convert_sync_batchnorm(student)
    student=student.to(ctx.device);opt=optimizer(student);wrapped=model.Outputs(student)
    if ctx.world>1:
        wrapped=DistributedDataParallel(wrapped,device_ids=[ctx.rank] if ctx.device.type=='cuda' else None,
            broadcast_buffers=True)
    torch.manual_seed(spec['nodes'][task]['initialization_seed']+ctx.rank)
    return student,wrapped,opt


def prepare(raw,banks,indices,ctx):
    arrays={k:raw[k] for k in ('features','vectors','mask','labels')}
    arrays.update(u100=banks['u100'].targets[indices],subclass=banks['subclass'].targets[indices])
    local,weight=shard(arrays,ctx.rank,ctx.world)
    return {k:torch.as_tensor(v,device=ctx.device) for k,v in local.items()},weight


def forward_loss(module,task,data,position):
    logits,pred,q=module(**{k:data[k] for k in ('features','vectors','mask')})
    total,parts=losses.objective(task,logits,data['labels'],pred,data['u100'],q,data['subclass'],position)
    return total,parts,losses.statistics(q,data['labels'],data['subclass'])


def probe(student,task,data,position,weight,ctx):
    # Bypass DDP's reducer for autograd.grad. SyncBatchNorm still synchronizes
    # its forward/backward; weighted parameter gradients are explicitly reduced.
    buffers={k:v.detach().clone() for k,v in student.named_buffers()}
    params=model.shared_parameters(student);vectors={}
    devices=[ctx.device.index if ctx.device.index is not None else torch.cuda.current_device()] if ctx.device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            outputs=model.Outputs(student).train()
            _,parts,_=forward_loss(outputs,task,data,position)
            for name in ('ce','weighted_embedding','weighted_subclass'):
                grads=torch.autograd.grad(parts[name]*weight,params,retain_graph=True,allow_unused=True)
                flat=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten() for p,g in zip(params,grads)])
                ctx.average(flat)
                if not torch.isfinite(flat).all():raise ValueError('Nonfinite diagnostic gradient')
                vectors[name]=flat.double()
    finally:
        with torch.no_grad():
            for k,v in student.named_buffers():v.copy_(buffers[k])
        student.clear()
    norms={k:v.norm() for k,v in vectors.items()};result={k+'_norm':float(v) for k,v in norms.items()}
    for a,b in itertools.combinations(vectors,2):
        result[a+'__'+b+'_cosine']=float(vectors[a].dot(vectors[b])/(norms[a]*norms[b]).clamp_min(1e-30))
    return result


def step(student,wrapped,opt,task,raw,banks,indices,position,ctx,*,diagnose=False):
    wrapped.train();data,weight=prepare(raw,banks,indices,ctx)
    for group in opt.param_groups:group['lr']=learning_rate(position)
    opt.zero_grad(set_to_none=True)
    diagnostic=probe(student,task,data,position,weight,ctx) if diagnose else None
    total,parts,stats=forward_loss(wrapped,task,data,position)
    (total*weight).backward()
    if not all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None):
        raise ValueError('Nonfinite distributed gradient')
    opt.step()
    names=list(parts)+list(stats)
    values=torch.tensor([float(v.detach()) for v in parts.values()]+list(stats.values()),device=ctx.device,dtype=torch.float64)*weight
    ctx.average(values);values=values.cpu().tolist();student.clear()
    result=dict(zip(names,values))
    return {k:result[k] for k in parts},diagnostic,{k:result[k] for k in stats}


def merge_predictions(shards,count):
    indices=np.concatenate([a for a,_ in shards]);prob=np.concatenate([p for _,p in shards])
    if (indices.dtype!=np.int64 or prob.shape!=(count,11) or not np.array_equal(np.sort(indices),np.arange(count))
            or not np.isfinite(prob).all() or (prob<0).any() or not np.allclose(prob.sum(1),1,atol=2e-6,rtol=0)):
        raise ValueError('Prediction shards duplicate, omit, or corrupt validation rows')
    return prob[np.argsort(indices)]


def predict(student,cache,ctx):
    student.eval()
    if hasattr(student,'capture'):student.capture=False;student.clear()
    indices=[];predictions=[]
    with torch.no_grad():
        for start in range(0,len(cache),128):
            ix=np.arange(start,min(start+128,len(cache)),dtype=np.int64)
            local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            if not len(local):continue
            raw=cache.batch(ix)
            logits=student(**{k:torch.as_tensor(raw[k][local],device=ctx.device) for k in ('features','vectors','mask')})
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite native validation')
            indices.append(ix[local]);predictions.append(logits.float().softmax(1).cpu().numpy())
    pair=(np.concatenate(indices) if indices else np.empty(0,np.int64),
          np.concatenate(predictions) if predictions else np.empty((0,11),np.float32))
    shards=ctx.all(pair)
    # Every rank reconstructs identical, authenticated row order. This avoids
    # publishing metrics on one rank while another enters a training collective.
    return merge_predictions(shards,len(cache))


def host_usage():
    """No extra dependency in the pinned scientific environment."""
    if os.name=='posix':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def launcher_rss():
    if os.name!='posix':return 0 # Local Windows CPU fixtures have no Hydra launcher.
    pid=os.getppid();seen=set();total=0
    while pid and pid not in seen:
        seen.add(pid)
        values=dict(line.split(':',1) for line in (Path('/proc')/str(pid)/'status').read_text().splitlines())
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')[0].decode()
        if not Path(command).name.startswith(('python','torchrun')):break
        total+=int(values['VmRSS'].split()[0])*1024;pid=int(values['PPid'])
    return total


def memory(ctx):
    peak_rss=host_usage()
    local=dict(peak_rss_bytes=int(peak_rss),gpu_peak_bytes=int(torch.cuda.max_memory_allocated(ctx.device)) if ctx.device.type=='cuda' else 0)
    ranks=ctx.all(local)
    # Include torchrun/worker ancestors in addition to both rank peaks. Summed
    # RSS is conservative because shared pages may be counted more than once.
    ancestors=ctx.main(launcher_rss)
    return dict(ranks=ranks,host_peak_bytes=sum(x['peak_rss_bytes'] for x in ranks)+ancestors,
                accounting='sum_rank_high_water_RSS_plus_launcher_RSS')


def timed(fn,ctx):
    ctx.barrier();ctx.sync();started=time.perf_counter();value=fn();ctx.sync()
    seconds=time.perf_counter()-started
    return max(ctx.all(seconds)),value
