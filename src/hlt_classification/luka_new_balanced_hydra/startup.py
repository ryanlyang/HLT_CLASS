"""Real acceptance inside each science allocation; probes never initialize a fit."""
from contextlib import nullcontext
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_u100_multiview.teachers import forward
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_new_fullstat_hydra.training import subset
from hlt_classification.luka_fullstat_balanced_ladder.artifacts import save_state
from hlt_classification.luka_u100_multiview.training import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from . import contracts as c, training, views
from hlt_classification.luka_fullstat_balanced_ladder import model, contrastive


def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def estimate(stage,train_step,val_step,export_step,elapsed,remaining_epochs=20):
    if (any(not math.isfinite(x) or x<=0 for x in (train_step,val_step))
            or not math.isfinite(export_step) or export_step<0 or not math.isfinite(elapsed) or elapsed<0
            or not 0<=remaining_epochs<=20 or (export_step==0)!=(stage=='DE000')):
        raise ValueError('Invalid timing witness')
    # Include all remaining epochs, one selected evaluation, export, CPU metric/IO margin.
    seconds=elapsed+3600+1.3*(remaining_epochs*c.STEPS*train_step+
        ((remaining_epochs+1)*math.ceil(c.COUNTS['validation']/c.BATCH)+math.ceil(c.OUTSIDE['validation']/c.BATCH))*val_step+c.STEPS*export_step)
    return dict(train_step=train_step,val_step=val_step,export_step=export_step,elapsed=elapsed,
        remaining_epochs=remaining_epochs,conservative_minutes=math.ceil(seconds/60),timing_factor=1.3,fixed_margin_minutes=60)


def run(spec,stage,tr,val,banks,sampler,matching,device,folder,started,remaining_epochs=20,outside=None):
    def record(name,**fields):
        value=c.artifact('STARTUP_PHASE',parents=dict(spec=spec['content_hash']),stage=stage,phase=name,**fields)
        meta.write_immutable_json(folder/'startup_diagnostics'/f'{name}.json',value)
        print('BALANCED startup '+name,flush=True)
    record('begin')
    with parity_backend(device):installed_parity(tr,device=device)
    net=model.new_model(stage).to(device).eval();native=net.deployed().to(device)
    raw=tr.batch(np.arange(min(4,len(tr))));errors={}
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            a=forward(net,raw,device,bf16=bf16);b=forward(native,raw,device,bf16=bf16)
            torch.testing.assert_close(a.float(),b.float(),rtol=1e-2 if bf16 else 2e-5,atol=1e-2 if bf16 else 2e-5)
            errors[str(bf16)]=float((a.float()-b.float()).abs().max())
    del native
    diagnostic=matching.diagnostic_sampler() if matching is not None else None
    if diagnostic is not None:
        serial=object.__new__(contrastive.SerialReference);serial.__dict__.update(diagnostic.__dict__)
        ids=np.array([0,0,1,2],np.int64);positions=np.arange(4,dtype=np.int64)
        a,_=serial.occurrences(ids,1,positions);b,_=diagnostic.sample(ids,1,positions)
        if not np.array_equal(a,b):raise ValueError('Serial/optimized occurrence candidate mismatch')
    record('parity',initial_errors=errors,installed_weaver_parity=True,matching_parity=diagnostic is not None)
    mini_ix=np.concatenate([np.flatnonzero(tr.labels==k)[:4] for k in range(11)])
    val_ix=np.concatenate([np.flatnonzero(val.labels==k)[:2] for k in range(11)])
    if len(mini_ix)!=44 or len(val_ix)!=22:raise ValueError('Integrated miniature lacks all classes')
    mini_val=subset(val,val_ix);mini_val.coordinate_name=val.coordinate_name
    mini_opt=training.optimizer(net,c.recipe());mini_rows=[];best_state=None
    for epoch in (1,2):
        net.train();mini_opt.zero_grad(set_to_none=True)
        candidates=diagnostic.sample(mini_ix,epoch,np.arange(len(mini_ix)),evidence=False)[0] if diagnostic else None
        loss,_=model.objective(net,tr.batch(mini_ix),mini_ix,banks,update=epoch,steps=1,device=device,
                               sampler=diagnostic,candidates=candidates)
        loss.backward();training.finite_gradients(net);mini_opt.step();net.clear();net.capture=False
        measured=evaluate_probabilities(mini_val.labels,predict(net,mini_val,device))
        row=dict(epoch=epoch,validation=measured);mini_rows.append(row)
        if max(mini_rows,key=lambda r:selection_key(r['validation'],r['epoch'])) is row:
            best_state=training.cpu_state(net.state_dict())
    save_state(folder/'miniature.pt',best_state)
    net.load_state_dict(torch.load(folder/'miniature.pt',map_location=device,weights_only=True),strict=True)
    chosen=max(mini_rows,key=lambda r:selection_key(r['validation'],r['epoch']))
    if evaluate_probabilities(mini_val.labels,predict(net,mini_val,device))!=chosen['validation']:
        raise ValueError('Integrated miniature checkpoint reload differs')
    record('miniature',history=mini_rows,selected_epoch=chosen['epoch'],checkpoint_reload_verified=True)
    del mini_opt,mini_val,best_state,loss
    ids=np.full(c.BATCH,int(np.argmax(views.lengths(tr))),np.int64)
    opt=training.optimizer(net,c.recipe());net.train()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        candidates=diagnostic.sample(ids,1,np.arange(step*c.BATCH,(step+1)*c.BATCH),evidence=False)[0] if diagnostic else None
        loss,parts=model.objective(net,tr.batch(ids),ids,banks,update=3*c.STEPS,steps=c.STEPS,
                                  device=device,sampler=diagnostic,candidates=candidates)
        grads=torch.autograd.grad(parts['ce'],tuple(net.mod.fc.parameters()),retain_graph=True)
        loss.backward();training.finite_gradients(net)
        for p,g in zip(net.mod.fc.parameters(),grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        opt.step();net.clear()
    record('stress',maximum_support=int(views.lengths(tr).max()),steps=3,classifier_ce_only=True)
    opt.zero_grad(set_to_none=True);del loss,parts,grads
    net.eval();net.capture=False;native=net.deployed().to(device)
    with parity_backend(device),torch.inference_mode():
        torch.testing.assert_close(forward(net,raw,device,bf16=False),forward(native,raw,device,bf16=False),rtol=0,atol=0)
        out=forward(native,val.batch(np.full(c.BATCH,int(np.argmax(views.lengths(val))),np.int64)),device,bf16=True)
        if not torch.isfinite(out).all():raise ValueError('Invalid max-support validation')
        if outside is not None and len(outside):
            extra=forward(native,outside.batch(np.full(c.BATCH,int(np.argmax(views.lengths(outside))),np.int64)),device,bf16=True)
            if not torch.isfinite(extra).all():raise ValueError('Invalid max-support outside validation')
            del extra
        if stage!='DE000':
            _,h=forward(native,tr.batch(ids),device,bf16=False,capture=True)
            if not torch.isfinite(h).all():raise ValueError('Invalid max-support FP32 export')
            del h
    del native,out
    record('deployment',native_parity=True,validation_and_export_full_support=True)
    order=sampler.draw(1,draws=68*c.BATCH);net.train()
    pool_context=contrastive.CPUWorkers(diagnostic,workers=spec['candidate_workers']) if diagnostic else nullcontext(None)
    with pool_context as pool:
        batches_context=contrastive.Lookahead(diagnostic,order,1,batch_size=c.BATCH) if diagnostic else nullcontext(
            ((ix,None) for _,ix in sampler.batches(order,batch_size=c.BATCH)))
        with batches_context as batches:
            for step,(ix,candidates) in enumerate(batches):
                if step==4:sync(device);begun=time.monotonic()
                opt.zero_grad(set_to_none=True)
                loss,parts=model.objective(net,tr.batch(ix),ix,banks,update=3*c.STEPS,steps=c.STEPS,
                    device=device,sampler=diagnostic,candidates=candidates)
                loss.backward();training.finite_gradients(net);opt.step();net.clear()
        sync(device);train_step=(time.monotonic()-begun)/64
        pool_bytes=pool.memory() if pool is not None else 0
    opt.zero_grad(set_to_none=True);del loss,parts
    rng=np.random.default_rng(84639);net.eval();net.capture=False;sync(device);begun=time.monotonic()
    with torch.inference_mode():
        for _ in range(16):
            logits=forward(net,val.batch(rng.integers(0,len(val),c.BATCH)),device,bf16=True)
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite validation benchmark')
            logits.float().softmax(-1).cpu().numpy()
    sync(device);val_step=(time.monotonic()-begun)/16;export_step=0.
    if stage!='DE000':
        sync(device);begun=time.monotonic()
        with torch.inference_mode():
            for ix in order[:16*c.BATCH].reshape(16,c.BATCH):
                raw=tr.batch(ix);_,h=forward(net,raw,device,bf16=False,capture=True)
                logits=forward(net,raw,device,bf16=True)
                if not torch.isfinite(h).all() or not torch.isfinite(logits).all():raise ValueError('Nonfinite export benchmark')
                h.float().cpu().numpy();(logits.float()/2).softmax(-1).cpu().numpy()
        sync(device);export_step=(time.monotonic()-begun)/16
    memory=memory_snapshot(torch);memory['candidate_worker_peak_bytes']=pool_bytes
    timing=estimate(stage,train_step,val_step,export_step,time.monotonic()-started,remaining_epochs)
    record('measurements',timing=timing,memory=memory)
    # Reserve temporary banks, normalization, selected states and OS overhead.
    if memory['peak_rss_bytes']+pool_bytes+12*2**30>spec['memory_mb']*1024**2*.95:
        raise MemoryError('Measured host memory plus export reserve exceeds allocation')
    if memory['gpu_peak_bytes']>torch.cuda.get_device_properties(device).total_memory*.92:
        raise MemoryError('Measured full-support CUDA peak lacks headroom')
    if timing['conservative_minutes']>spec['minutes']['train_'+stage]:
        raise RuntimeError('Measured remaining work exceeds walltime; retained diagnostic requires explicit resource revision')
    del net,opt,diagnostic;gc.collect();torch.cuda.empty_cache()
    return c.artifact('STARTUP',parents=dict(spec=spec['content_hash']),stage=stage,initial_errors=errors,
        installed_weaver_parity=True,finite_connected_gradients=True,classifier_ce_only=True,
        native_deployment_parity=True,full_support=True,matching_serial_parity=stage=='DE000',
        integrated_two_epoch_miniature=True,probe_discarded=True,memory=memory,timing=timing)
