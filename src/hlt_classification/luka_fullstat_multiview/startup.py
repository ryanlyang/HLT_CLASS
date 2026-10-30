"""Bounded new-graph checks inside the allocation, discarded before fresh fits."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_retained.worker import check_memory, sync
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_fullstat_readout import worker as rw, model as rm
from hlt_classification.luka_u100_multiview import teachers
from . import model, training, views
from .contracts import artifact, validate, recipe


def stress_ids(value):
    if isinstance(value,views.PairedCache):
        # One batch reaches BOTH stream maxima, even if they occur on different jets.
        a=int(np.argmax(views.lengths(value.primary)));b=int(np.argmax(views.lengths(value.context)))
        return np.r_[np.full(64,a,np.int64),np.full(64,b,np.int64)]
    return np.full(128,int(np.argmax(views.lengths(value))),np.int64)


def timing(spec,task,train_step,val_step,export_step,preparation):
    name=model.teacher_name(task)
    if (any(not math.isfinite(v) or v<=0 for v in (train_step,val_step))
            or not math.isfinite(export_step) or (export_step<=0 if name else export_step!=0)
            or not math.isfinite(preparation) or preparation<0): raise ValueError('Invalid timing witness')
    nt=math.ceil(spec['counts']['train']/128);nv=math.ceil(spec['counts']['validation']/128)
    minutes=math.ceil(preparation/60+120+1.5*(20*nt*train_step+22*nv*val_step+nt*export_step)/60)
    if minutes>spec['minutes'][task]: raise ValueError('Measured full20 plus export exceeds allocation')
    return dict(train_step=train_step,val_step=val_step,export_step=export_step,preparation=preparation,
                conservative_minutes=minutes,full_epoch_measured=False)


def run(spec,task,tr,val,banks,meta,device,started):
    name=model.teacher_name(task); witness=None
    if not name:
        readout=rm.make_readout(banks['U100'],Path(spec['bank_manifest']['path']).parent,spec['teacher_refs'],device)
        witness=rw.teacher_parity(spec,banks['U100'],readout,meta,device)
        del readout
    ordinary_input=tr.primary if name=='FUSION' else tr
    with parity_backend(device): installed_parity(ordinary_input,device=device)
    net=model.new_model(spec['nodes'][task],task).to(device).eval()
    native=DelphesParticleTransformer().to(device).eval()
    if name=='FUSION': native.mod.load_state_dict(net.primary_mod.state_dict(),strict=True)
    else: native.load_state_dict({k:v for k,v in net.state_dict().items() if k.startswith('mod.')},strict=True)
    raw=tr.batch(np.arange(min(4,len(tr))));errors={}
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            a=teachers.forward(net,raw,device,bf16=bf16,name=name)
            b=teachers.forward(native,raw,device,bf16=bf16)
            tol=1e-2 if bf16 else 2e-5
            torch.testing.assert_close(a.float(),b.float(),rtol=tol,atol=tol)
            errors[str(bf16)]=float((a.float()-b.float()).abs().max())
    del native
    net.train();opt=training.optimizer(net,recipe()['base']);ids=stress_ids(tr)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss,parts=model.loss_for(net,task,tr.batch(ids),ids,banks,device)
        fc=net.primary_mod.fc if name=='FUSION' else net.mod.fc
        ce_grad=torch.autograd.grad(parts['ce'],tuple(fc.parameters()),retain_graph=True)
        loss.backward();training.finite_gradients(net)
        for p,g in zip(fc.parameters(),ce_grad): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        opt.step()
        if not name: net.predictions={}
    if name=='FUSION' and not any(p.grad is not None and torch.any(p.grad!=0) for p in net.context_mod.parameters()):
        raise ValueError('Fusion context stayed disconnected')
    if not name:
        if not all(any(p.grad is not None and torch.any(p.grad!=0) for p in head.parameters()) for head in net.predictors.values()):
            raise ValueError('Embedding predictor disconnected')
        net.capture=False
    net.eval();deployed=model.deploy(net,task).to(device).eval()
    if name:
        restored=model.new_model(spec['nodes'][task],task).to(device).eval()
        restored.load_state_dict(training.cpu_state(deployed.state_dict()),strict=True)
    else:
        restored=DelphesParticleTransformer().to(device).eval()
        restored.load_state_dict(training.cpu_state(deployed.state_dict()),strict=True)
    with parity_backend(device),torch.inference_mode():
        a=teachers.forward(net,raw,device,bf16=False,name=name)
        b=teachers.forward(restored,raw,device,bf16=False,name=name)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    del restored,deployed
    with torch.inference_mode():
        out=teachers.forward(net,val.batch(stress_ids(val)),device,bf16=torch.device(device).type=='cuda',name=name)
        if not torch.isfinite(out).all(): raise ValueError('Invalid max-support validation')
        if name:
            out,h=teachers.forward(net,tr.batch(ids),device,bf16=False,name=name,capture=True)
            if not torch.isfinite(h).all() or not torch.isfinite(out).all(): raise ValueError('Invalid max-support FP32 export')
    rng=np.random.default_rng(73012);net.train()
    if not name: net.capture=True
    for step in range(68):
        if step==4: sync(device);begun=time.monotonic()
        ix=rng.integers(0,len(tr),128);opt.zero_grad(set_to_none=True)
        loss,_=model.loss_for(net,task,tr.batch(ix),ix,banks,device)
        loss.backward();training.finite_gradients(net);opt.step()
        if not name: net.predictions={}
    sync(device);train_step=(time.monotonic()-begun)/64
    net.eval()
    if not name: net.capture=False
    sync(device);begun=time.monotonic()
    with torch.inference_mode():
        for _ in range(16):
            out=teachers.forward(net,val.batch(rng.integers(0,len(val),128)),device,bf16=torch.device(device).type=='cuda',name=name)
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite validation timing')
    sync(device);val_step=(time.monotonic()-begun)/16;export_step=0.
    if name:
        sync(device);begun=time.monotonic()
        with torch.inference_mode():
            for _ in range(16):
                _,h=teachers.forward(net,tr.batch(rng.integers(0,len(tr),128)),device,bf16=False,name=name,capture=True)
                if not torch.isfinite(h).all(): raise ValueError('Nonfinite export timing')
                h.float().cpu().numpy()
        sync(device);export_step=(time.monotonic()-begun)/16
    memory=memory_snapshot(torch);check_memory(spec,memory)
    estimate=timing(spec,task,train_step,val_step,export_step,time.monotonic()-started)
    print(f'FULLSTAT MULTIVIEW integrated task={task} conservative_minutes={estimate["conservative_minutes"]}',flush=True)
    return artifact('STARTUP',parents=dict(spec=spec['content_hash']),task_id=task,
        teacher_witness=witness,installed_weaver_parity=True,initial_parity=True,initial_errors=errors,
        finite_connected_gradients=True,classifier_CE_only=True,deployment_readback=True,
        batch=128,stress_steps=3,train=views.describe(tr),validation=views.describe(val),
        train_stress_rows=ids.tolist(),validation_stress_rows=stress_ids(val).tolist(),
        probe_discarded=True,memory=memory,timing=estimate)


def check(spec,task,value):
    validate(value,'STARTUP',dict(spec=spec['content_hash']));name=model.teacher_name(task)
    if (value['task_id']!=task or value['batch']!=128 or value['stress_steps']!=3
            or any(value[k] is not True for k in ('installed_weaver_parity','initial_parity',
                'finite_connected_gradients','classifier_CE_only','deployment_readback','probe_discarded'))):
        raise ValueError('Startup graph checks differ')
    if set(value['initial_errors'])!={'False','True'} or any(not math.isfinite(v) or v<0 for v in value['initial_errors'].values()):
        raise ValueError('Invalid parity errors')
    for role in ('train','validation'):
        v=value[role];ids=value[role+'_stress_rows']
        if (v['role']!=role or v['coordinate']!=(name or 'D000') or v['rows']!=spec['counts'][role]
                or v['maximum']<1 or v['resident_bytes']<1 or len(ids)!=128
                or any(type(i)is not int or not 0<=i<v['rows'] for i in ids)):
            raise ValueError('Startup population/stress differs')
        from hlt_classification.data.cache_contracts import require_sha256
        require_sha256(v['input_sha256'],name='input fingerprint')
    if not name: rw.validate_teacher_witness(spec,value['teacher_witness'])
    elif value['teacher_witness'] is not None: raise ValueError('Oracle used unrelated supervision')
    check_memory(spec,value['memory']);t=value['timing']
    if t!=timing(spec,task,t['train_step'],t['val_step'],t['export_step'],t['preparation']):
        raise ValueError('Timing witness changed')
