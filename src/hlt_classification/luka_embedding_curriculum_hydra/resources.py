"""Warmed full/partial production transitions; no scientific-quality gate."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_u100_contrastive.training import peaks
from hlt_classification.luka_cross_message_hydra.resources import Subset
from hlt_classification.luka_particle_teacher_hydra.resources import miniature_rows
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_u100_embedding.training import objective as baseline_objective
from hlt_classification.jetclass2_delphes.runner import _forward
from .contracts import ORDER,artifact,validate
from . import training


def equal_state(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a) is type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def parity_record(spec,environment):
    return artifact('BASELINE_PARITY',parents=dict(spec=spec['content_hash']),max_abs=0.,prediction_max_abs=0.,
        native_tensors_equal=True,one_update_model_equal=True,one_update_optimizer_equal=True,
        installed_weaver=True,environment=environment,quality_gate=False)


def baseline(spec,train,bank,device):
    training.join(train,bank,'train')
    torch.manual_seed(spec['imported']['node']['initialization_seed']); original=AuxiliaryParT().to(device).eval()
    candidate=training.new_model(spec,device).eval(); original.capture=candidate.capture=True
    if any(not torch.equal(v,candidate.state_dict()[k]) for k,v in original.state_dict().items()): raise ValueError('Initialization differs')
    raw=train.batch(np.arange(min(128,len(train)))); data={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    with torch.no_grad(),torch.autocast(torch.device(device).type,enabled=False): a=original(**data); b=candidate(**data)
    if not torch.equal(a,b) or not torch.equal(original.prediction,candidate.prediction): raise ValueError('Native FP32 parity failed')
    # Compare the new E00 update against the original U100 objective and forward,
    # including dropout RNG, all model buffers and AdamW moments/step counters.
    opt=training.optimizer(original); candidate_opt=training.optimizer(candidate)
    cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []
    original.train(); opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=training.learning_rate('E00',1.)
    z=_forward(original,raw,device,bf16=str(device).startswith('cuda'))
    loss,_,_=baseline_objective(z,torch.from_numpy(raw['labels']).to(device),original.prediction,
        torch.as_tensor(bank.targets[:len(z)],device=device),coefficient=5.)
    loss.backward(); opt.step(); original.prediction=None
    torch.set_rng_state(cpu_rng)
    if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
    training.step(candidate,candidate_opt,raw,bank.targets[:len(z)],'E00',1.,device)
    if not equal_state(original.state_dict(),candidate.state_dict()) or not equal_state(opt.state_dict(),candidate_opt.state_dict()):
        raise ValueError('Original U100 one-update parity failed')
    return parity_record(spec,installed_environment())


def record(spec,task,report,warm,memory,gpu,preparation):
    for r in (report,warm):
        if r['task']!=task or r['technical'] is not True or r['epochs']!=3 or r['parents']['spec']!=spec['content_hash']:
            raise ValueError('Need three-stage production miniature')
    if report['counts']!=warm['counts'] or not math.isfinite(preparation) or preparation<0: raise ValueError('Bad miniature populations/timing')
    full=[]; evaluations=[]
    for row in report['validation_history']:
        times=row['batch_seconds']; sizes=row['batch_sizes']
        if (len(times)!=len(sizes) or not np.isfinite(times).all() or min(times)<=0 or 128 not in sizes
                or not any(2<=n<128 for n in sizes) or 1 in sizes): raise ValueError('Need warmed full and partial batches')
        full.extend(t for t,n in zip(times,sizes) if n==128)
        evaluations.append(row['validation_seconds']/math.ceil(report['counts']['validation']/128))
    vstep=max(evaluations); step=max(full)
    if not math.isfinite(vstep) or vstep<=0: raise ValueError('Bad validation timing')
    estimate=30*(step*math.ceil(spec['counts']['train']/128)+vstep*math.ceil(spec['counts']['validation']/128))
    minutes=max(30,math.ceil((2*estimate+preparation+900)/60))
    limits=dict(minutes=spec['fit_max_minutes'],gpu_peak_bytes=int(gpu['total_memory_bytes']*.85),peak_rss_bytes=int(spec['memory_mb']*1024**2*.85))
    quantities=dict(minutes=minutes,gpu_peak_bytes=memory['gpu_peak_bytes'],peak_rss_bytes=memory.get('peak_rss_bytes'))
    if any(v is not None and (not math.isfinite(v) or v<0) for v in quantities.values()): raise ValueError('Nonfinite resource measurement')
    violations=[dict(resource=k,measured=v,limit=limits[k]) for k,v in quantities.items() if v is not None and v>limits[k]]
    return artifact('RESOURCE_CHECK',parents=dict(spec=spec['content_hash'],report=report['content_hash'],warm=warm['content_hash']),
        task=task,minutes=minutes,train_step_seconds=step,validation_step_seconds=vstep,memory=memory,gpu=gpu,
        preparation_seconds=preparation,limits=limits,violations=violations,passed=not violations,quality_gate=False)


def subset_bank(bank,indices):
    return SimpleNamespace(identities=bank.identities[indices],labels=bank.labels[indices],targets=bank.targets[indices],
        manifest=bank.manifest,role=bank.role)


def measure(spec,train,val,bank,vbank,preparation,device='cuda'):
    folder=Path(spec['root'])/'setup'; ix=miniature_rows(train); vx=miniature_rows(val)
    tr=Subset(train,ix); va=Subset(val,vx); tb=subset_bank(bank,ix); vb=subset_bank(vbank,vx)
    before=peaks(device); measurements={}
    for task in ORDER:
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        warm=training.fit(folder/('warm_'+task),spec,task,tr,va,tb,vb,device=device,technical=True)
        gc.collect()
        report=training.fit(folder/('mini_'+task),spec,task,tr,va,tb,vb,device=device,technical=True)
        memory=peaks(device)
        for key in ('gpu_peak_bytes','peak_rss_bytes'):
            vs=[v for v in (memory.get(key),before.get(key)) if v is not None]; memory[key]=max(vs) if vs else None
        r=record(spec,task,report,warm,memory,gpu_identity(),preparation)
        write_immutable_json(folder/('resource_'+task+'.json'),r); measurements[task]=r
        print(f'CURRICULUM resource {task}: step={r["train_step_seconds"]:.3f}s request={r["minutes"]}min',flush=True)
        if not r['passed']: raise RuntimeError('Curriculum resource envelope exceeded: '+str(r['violations']))
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    a=artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],validation_bank=vbank.manifest['content_hash']),
        measurements=measurements,minutes={t:r['minutes'] for t,r in measurements.items()},
        environment=installed_environment(),gpu=gpu_identity(),quality_gate=False,technical_weights_discarded=True)
    write_immutable_json(folder/'acceptance.json',a)
    return a


def accepted(spec):
    folder=Path(spec['root'])/'setup'; m=load_json(folder/'validation_embeddings.json'); a=load_json(folder/'acceptance.json')
    validate(a,'ACCEPTANCE',dict(spec=spec['content_hash'],validation_bank=m['content_hash']))
    if (set(a['measurements'])!=set(ORDER) or set(a['minutes'])!=set(ORDER)
            or a['environment']!=spec['reference']['environment'] or a['gpu']['name']!=spec['execution_site']['gpu_name']
            or a['gpu']['compute_capability']!=[8,6] or a['quality_gate'] is not False or a['technical_weights_discarded'] is not True):
        raise ValueError('Incomplete acceptance or wrong backend')
    parity=load_json(folder/'baseline_parity.json'); validate(parity,'BASELINE_PARITY',dict(spec=spec['content_hash']))
    if parity!=parity_record(spec,a['environment']): raise ValueError('Baseline parity differs')
    for task,r in a['measurements'].items():
        report,_=training.validate_fit(folder/('mini_'+task),spec,task,technical=True)
        warm,_=training.validate_fit(folder/('warm_'+task),spec,task,technical=True)
        if (r!=record(spec,task,report,warm,r['memory'],a['gpu'],r['preparation_seconds']) or not r['passed']
                or r!=load_json(folder/('resource_'+task+'.json')) or a['minutes'][task]!=r['minutes']): raise ValueError('Measured resource evidence differs')
    return a
