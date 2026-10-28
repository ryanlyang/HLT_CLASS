"""Bounded disposable in-worker checks, not a separate queued preflight."""
import time
import math
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_attention.banks import counts
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot,check_headroom
from .contracts import artifact,validate
from .model import new_model
from .training import batch_loss,optimizer,finite_gradients,clear,registered


def run(spec,tr,val,bank,node,device,*,elapsed_before_seconds=0.):
    started=time.perf_counter()
    if str(device).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    model=new_model(node).to(device).eval(); model.capture=True
    torch.manual_seed(node['initialization_seed']); native=DelphesParticleTransformer().to(device).eval()
    raw=tr.batch(np.arange(min(4,len(tr)),dtype=np.int64))
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            torch.testing.assert_close(_forward(model,raw,device,bf16=bf16),_forward(native,raw,device,bf16=bf16),rtol=0,atol=0)
    del native; clear(model)
    tr_counts=counts(tr); val_counts=counts(val)
    ix=np.full(128,int(np.argmax(tr_counts)),dtype=np.int64)
    model.train(); model.capture=True; opt=optimizer(model,registered(node['arm'])['base'])
    opt.zero_grad(set_to_none=True)
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    step_started=time.perf_counter()
    loss,terms,_=batch_loss(model,tr,bank,ix,device)
    head=tuple(model.mod.fc.parameters())
    expected=torch.autograd.grad(registered(node['arm'])['ce_coefficient']*terms['ce'],head,retain_graph=True)
    loss.backward(); finite_gradients(model)
    for p,g in zip(head,expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.neighborhood_head.parameters()):
        raise ValueError('Disconnected neighborhood supervision')
    opt.step()
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    step_seconds=time.perf_counter()-step_started
    clear(model); model.eval(); model.capture=False
    validation_started=time.perf_counter()
    with torch.inference_mode():
        out=_forward(model,val.batch(np.full(128,int(np.argmax(val_counts)),dtype=np.int64)),device,bf16=str(device).startswith('cuda'))
        if not torch.isfinite(out).all(): raise ValueError('Invalid maximum-validation output')
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    validation_seconds=time.perf_counter()-validation_started
    # Maximum-support probes admit memory, not a fictitious epoch containing
    # only copies of the longest jet. Time eight seeded real-population batches
    # separately, using the slowest sample and a 2x factor plus publication time.
    train_times=[]; validation_times=[]; rng=np.random.default_rng(94272)
    for _ in range(8):
        ix=rng.choice(len(tr),128,replace=len(tr)<128).astype(np.int64)
        model.train(); model.capture=True; opt.zero_grad(set_to_none=True)
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        tick=time.perf_counter()
        probe,probe_terms,_=batch_loss(model,tr,bank,ix,device)
        probe.backward(); finite_gradients(model); opt.step()
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        train_times.append(time.perf_counter()-tick)
        clear(model); del probe,probe_terms
        model.eval(); model.capture=False
        jx=rng.choice(len(val),128,replace=len(val)<128).astype(np.int64)
        tick=time.perf_counter()
        with torch.inference_mode():
            output=_forward(model,val.batch(jx),device,bf16=str(device).startswith('cuda'))
            if not torch.isfinite(output).all(): raise ValueError('Invalid timing output')
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        validation_times.append(time.perf_counter()-tick)
    projected_seconds=projection(len(tr),len(val),train_times,validation_times)
    if not np.isfinite(projected_seconds) or projected_seconds<=0: raise ValueError('Invalid runtime projection')
    elapsed_seconds=elapsed_before_seconds+time.perf_counter()-started
    if str(device).startswith('cuda') and projected_seconds+elapsed_seconds>spec['task_minutes'][node['arm']]*60:
        raise ValueError('Conservative measured fit projection exceeds registered walltime; preserve artifacts')
    memory=memory_snapshot(torch)
    if str(device).startswith('cuda'):
        check_headroom(memory,memory_mb=spec['memory_mb'],future_cache_bytes=0,gpu_bytes=torch.cuda.get_device_properties(torch.device(device)).total_memory)
    del model,opt,loss,terms,expected,out
    return artifact('STARTUP',parents=dict(spec=spec['content_hash'],bundle=bank.manifest['content_hash']),task=node['arm'],
        passed=True,native_parity=True,main_head_CE_only=True,discarded=True,batch_size=128,updates=9,
        maximum_train=int(tr_counts.max()),maximum_validation=int(val_counts.max()),memory=memory,step_seconds=step_seconds,
        validation_seconds=validation_seconds,projected_seconds=projected_seconds,head_connected=True,
        train_samples=train_times,validation_samples=validation_times,elapsed_seconds=elapsed_seconds,
        timing_seed=94272,projection_factor=2,publication_reserve_seconds=1200)


def projection(train_rows,validation_rows,train_times,validation_times):
    return 2*(30*math.ceil(train_rows/128)*max(train_times)
              +33*math.ceil(validation_rows/128)*max(validation_times))+1200


def check(value,spec,task,bundle_hash):
    validate(value,'STARTUP',dict(spec=spec['content_hash'],bundle=bundle_hash))
    if (value['task']!=task or any(value[k] is not True for k in ('passed','native_parity','main_head_CE_only','discarded','head_connected'))
            or (value['batch_size'],value['updates'])!=(128,9)
            or any(type(value[k])!=int or not 1<=value[k]<=512 for k in ('maximum_train','maximum_validation'))):
        raise ValueError('Startup evidence differs')
    if any(not math.isfinite(value[k]) or value[k]<=0 for k in ('step_seconds','validation_seconds','projected_seconds')):
        raise ValueError('Invalid startup timing')
    samples=value['train_samples']+value['validation_samples']
    if (len(value['train_samples'])!=8 or len(value['validation_samples'])!=8
            or any(not math.isfinite(v) or v<=0 for v in samples)
            or not math.isfinite(value['elapsed_seconds']) or value['elapsed_seconds']<0
            or (value['timing_seed'],value['projection_factor'],value['publication_reserve_seconds'])!=(94272,2,1200)
            or value['projected_seconds']!=projection(spec['counts']['train'],spec['counts']['validation'],
                                                     value['train_samples'],value['validation_samples'])):
        raise ValueError('Startup projection differs')
