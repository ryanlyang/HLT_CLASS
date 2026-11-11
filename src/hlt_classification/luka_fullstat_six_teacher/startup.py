"""Integrated acceptance measures the complete sampler and optimizer path."""
from pathlib import Path
from copy import copy
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_multiview.startup import stress_ids
from hlt_classification.luka_fullstat_multiview import views
from hlt_classification.luka_fullstat_multiview.resources import check_memory
from hlt_classification.luka_fullstat_retained.worker import sync
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_u100_multiview import teachers
from . import model,training
from .sampling import Lookahead, ReferenceSampler
from .contracts import artifact,validate,recipe,TASK


def timing(spec,step,val_step,preparation):
    if any(not math.isfinite(x) or x<=0 for x in (step,val_step)) or preparation<0 or not math.isfinite(preparation):
        raise ValueError('Invalid timing evidence')
    minutes=math.ceil(preparation/60+120+1.5*(20*math.ceil(spec['counts']['train']/128)*step
                      +22*math.ceil(spec['counts']['validation']/128)*val_step)/60)
    return dict(train_step=step,validation_step=val_step,preparation_seconds=preparation,
                conservative_minutes=minutes,full_epoch_measured=False)


def check(spec,proof):
    validate(proof,'STARTUP',dict(spec=spec['content_hash']))
    for k in ('installed_weaver_parity','exact_sampler_parity','connected_gradients','classifier_ce_only',
              'native_reload_parity','probe_discarded'):
        if proof[k] is not True:raise ValueError('Missing integrated check: '+k)
    for role in ('train','validation'):
        v=proof[role]
        if v['role']!=role or v['coordinate']!='D000' or v['rows']!=spec['counts'][role]:
            raise ValueError('Wrong startup input population')
    if proof['batch']!=128 or proof['candidates']!=1025 or proof['measured_steps']!=64:
        raise ValueError('Wrong measured workload')
    if proof['candidate_cpu_workers']!=3 or proof['cpu_worker_peak_rss_bytes']<0:
        raise ValueError('Wrong CPU candidate workload')
    t=proof['timing']
    if t!=timing(spec,t['train_step'],t['validation_step'],t['preparation_seconds']):
        raise ValueError('Timing identity differs')
    if t['conservative_minutes']>spec['minutes'][TASK]:
        raise ValueError(f'Measured estimate {t["conservative_minutes"]} min exceeds allocation; see startup measurements')
    check_memory(spec,proof['memory'])


def run(spec,tr,val,banks,sampler,device,started):
    folder=Path(spec['root'])/TASK
    try:
        return _run(spec,tr,val,banks,sampler,device,started)
    except Exception as exc:
        write_immutable_json(folder/'startup_failure.json',artifact('FAILURE',
            error_type=type(exc).__name__,message=str(exc),memory=memory_snapshot(torch)))
        raise


def _run(spec,tr,val,banks,sampler,device,started):
    with parity_backend(device):installed_parity(tr,device=device)
    anchors=np.r_[[rows[0] for rows in sampler.classes],np.random.default_rng(104).integers(0,len(tr),17)]
    reference=copy(sampler);reference.__class__=ReferenceSampler
    before=time.monotonic();expected,oldproof=reference.sample(anchors,1,'mix')
    reference_seconds=time.monotonic()-before
    before=time.monotonic();actual,newproof=sampler.sample(anchors,1)
    optimized_seconds=time.monotonic()-before
    if not np.array_equal(expected,actual) or oldproof!=newproof:
        raise ValueError('Optimized sampler differs from exact reference')
    net=model.new_model(spec['nodes'][TASK]).to(device);net.train()
    opt=training.optimizer(net,recipe()['multi3']['base'])
    ids=stress_ids(tr);candidates,_=sampler.sample(ids,1,evidence=False)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss,parts=model.objective(net,tr.batch(ids),ids,banks,sampler,candidates,.3,device,subclass_weight=.5,ramp=1.)
        fc=list(net.mod.fc.parameters());expected_grad=torch.autograd.grad(parts['ce'],fc,retain_graph=True)
        loss.backward();training.finite_gradients(net)
        for p,g in zip(fc,expected_grad):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        opt.step();net.clear()
    for head in (net.subclass,net.matching,*net.predictors.values(),*net.logit_heads.values()):
        if not any(p.grad is not None and torch.any(p.grad!=0) for p in head.parameters()):
            raise ValueError('Disconnected auxiliary head')
    opt.zero_grad(set_to_none=True);del loss,parts,expected_grad
    net.eval();net.capture=False;native=net.deployed().to(device)
    raw=tr.batch(np.arange(11))
    with parity_backend(device),torch.no_grad():
        a=teachers.forward(net,raw,device,bf16=False);b=teachers.forward(native,raw,device,bf16=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        out=teachers.forward(native,val.batch(stress_ids(val)),device,bf16=torch.device(device).type=='cuda')
        if not torch.isfinite(out).all():raise ValueError('Nonfinite validation stress')
    del native,a,b,out
    # Real global-bank indices, full candidates, native batch and full support.
    order=np.random.default_rng(73012).integers(0,len(tr),68*128)
    net.train();net.capture=True
    for indices in (order[:4*128],order[4*128:]):
        sync(device);tick=time.monotonic()
        with Lookahead(sampler,indices,2) as batches:
            for ix,ci in batches:
                opt.zero_grad(set_to_none=True)
                loss,parts=model.objective(net,tr.batch(ix),ix,banks,sampler,ci,.3,device,subclass_weight=.5,ramp=1.)
                loss.backward();training.finite_gradients(net);opt.step()
                torch.stack([x.detach() for x in parts.values()]).cpu().numpy();net.clear()
        sync(device);train_step=(time.monotonic()-tick)/(len(indices)/128)
    opt.zero_grad(set_to_none=True);del loss,parts
    net.eval();net.capture=False;rng=np.random.default_rng(73013);sync(device);tick=time.monotonic()
    with torch.no_grad():
        for _ in range(16):
            out=teachers.forward(net,val.batch(rng.integers(0,len(val),128)),device,bf16=torch.device(device).type=='cuda')
            if not torch.isfinite(out).all():raise ValueError('Nonfinite validation benchmark')
            out.float().softmax(1).cpu().numpy()
    sync(device);val_step=(time.monotonic()-tick)/16
    cpu=sampler._parallel_cpu;memory=memory_snapshot(torch)
    memory['peak_rss_bytes']+=cpu.memory();memory['rss_bytes']+=cpu.memory()
    proof=artifact('STARTUP',parents=dict(spec=spec['content_hash']),batch=128,candidates=1025,measured_steps=64,
        candidate_cpu_workers=cpu.workers,cpu_worker_peak_rss_bytes=cpu.memory(),
        installed_weaver_parity=True,exact_sampler_parity=True,connected_gradients=True,classifier_ce_only=True,
        native_reload_parity=True,probe_discarded=True,train=views.describe(tr),validation=views.describe(val),
        sampler_capacity=sampler.capacity,sampler_probe_seconds=dict(reference=reference_seconds,optimized=optimized_seconds),
        memory=memory,timing=timing(spec,train_step,val_step,time.monotonic()-started))
    # Evidence survives resource rejection. A finite poor AUC is never an admission criterion.
    write_immutable_json(Path(spec['root'])/TASK/'startup_measurements.json',proof)
    check(spec,proof)
    print(f'SIX TEACHER measured update={train_step:.4f}s; conservative request={proof["timing"]["conservative_minutes"]}min',flush=True)
    return proof
