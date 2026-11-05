"""Genuine warmed production paths, complete candidate universes and finite-only admission."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_u100_contrastive.training import peaks
from hlt_classification.luka_cross_message_hydra.resources import Subset
from hlt_classification.luka_particle_teacher_hydra.resources import miniature_rows
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from .contracts import artifact,validate,setting,seed
from . import inputs,training,model


def path_key(task):
    cfg=setting(task)
    # Same tensor support and compute graph; scalar weights and replicate seeds
    # do not create new execution paths. Source, sampler and interaction do.
    return tuple((k,str(v)) for k,v in cfg.items() if k not in ('coefficient','seed_index'))


def baseline(spec,train,device):
    raw=train.batch(np.arange(min(128,len(train)))); values={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    errors={}
    for task in ('C0','C1','P1','G1','J1','J5'):
        torch.manual_seed(seed(spec['imported'],setting(task),'initialization_seed'))
        original=AuxiliaryParT().to(device).eval(); original.capture=True
        candidate=model.new_model(spec['imported'],task).to(device).eval(); candidate.capture=True
        native=candidate.deployment_state()
        if native.keys()!=original.native_state().keys() or any(not torch.equal(v,original.native_state()[k]) for k,v in native.items()):
            raise ValueError('Native initialization changed')
        if any(not torch.equal(v,original.predictor.state_dict()[k]) for k,v in candidate.predictor.state_dict().items()):
            raise ValueError('Original U100 prediction head changed')
        with torch.no_grad(),torch.autocast(torch.device(device).type,enabled=False):
            a=original(**values); b=candidate(**values)
        if not torch.equal(a,b) or not torch.equal(original.prediction,candidate.prediction): raise ValueError('Installed-Weaver native FP32 parity failed')
        errors[task]=dict(max_abs=0.,prediction_max_abs=0.,native_tensors_equal=True)
        del original,candidate
    return artifact('BASELINE_PARITY',parents=dict(spec=spec['content_hash']),checks=errors,
        environment=installed_environment(),installed_weaver=True,quality_gate=False)


def resource_record(spec,task,report,warm,memory,gpu,preparation_seconds):
    for r in (report,warm):
        if r['task']!=task or r['technical'] is not True or r['epochs']!=1 or r['parents']['spec']!=spec['content_hash']:
            raise ValueError('Expected task-bound production miniatures')
    row=report['validation_history'][0]; times=row['batch_seconds']; sizes=row['batch_sizes']
    if (report['counts']!=warm['counts'] or len(times)!=len(sizes) or 128 not in sizes or 1 in sizes
            or not any(2<=n<128 for n in sizes) or not np.isfinite(times).all() or min(times)<=0
            or not math.isfinite(preparation_seconds) or preparation_seconds<0): raise ValueError('Need warmed full/partial timing')
    phases=row.get('batch_phase_seconds')
    if (not isinstance(phases,list) or len(phases)!=len(times)
            or any(set(p)!=set(training.PHASE_KEYS) or not np.isfinite(list(p.values())).all()
                or min(p.values())<0 or not math.isclose(sum(p.values()),t,rel_tol=1e-6,abs_tol=1e-6)
                for p,t in zip(phases,times))): raise ValueError('Missing complete production phase timings')
    # Probes run once per epoch, not once per optimizer update. The complete
    # miniature still measures their cost and memory; budget their actual use.
    perstep=max(t-p['diagnostics'] for t,p,n in zip(times,phases,sizes) if n==128)
    probe_seconds=max(p['diagnostics'] for p in phases)
    if not math.isfinite(perstep) or perstep<=0: raise ValueError('Invalid regular production timing')
    vstep=row['validation_seconds']/math.ceil(report['counts']['validation']/128)
    if not math.isfinite(vstep) or vstep<=0: raise ValueError('Invalid native validation timing')
    estimate=30*(perstep*math.ceil(spec['counts']['train']/128)+probe_seconds+vstep*math.ceil(spec['counts']['validation']/128))
    policy=spec['resource_policy']; minutes=max(30,math.ceil((2*estimate+preparation_seconds+900)/60))
    if policy!=dict(safety_multiplier=2.,fixed_overhead_seconds=900,memory_fraction=.85): raise ValueError('Resource policy drift')
    limits=dict(minutes=spec['fit_max_minutes'],gpu_peak_bytes=int(gpu['total_memory_bytes']*.85),peak_rss_bytes=int(spec['memory_mb']*1024**2*.85))
    for key in ('gpu_peak_bytes','peak_rss_bytes'):
        n=memory.get(key)
        if n is None and key=='peak_rss_bytes': continue  # Windows synthetic fixture only.
        if not isinstance(n,(int,float)) or not math.isfinite(n) or n<0: raise ValueError('Invalid resource measurement')
    violations=[dict(resource=k,measured=n,limit=limits[k]) for k,n in dict(minutes=minutes,
        gpu_peak_bytes=memory['gpu_peak_bytes'],peak_rss_bytes=memory.get('peak_rss_bytes')).items() if n is not None and n>limits[k]]
    return artifact('RESOURCE_CHECK',parents=dict(spec=spec['content_hash'],report=report['content_hash'],warm=warm['content_hash']),
        task=task,minutes=minutes,memory=memory,gpu=gpu,preparation_seconds=preparation_seconds,
        train_step_seconds=perstep,probe_seconds_per_epoch=probe_seconds,
        batch_phase_seconds=phases,validation_step_seconds=vstep,limits=limits,violations=violations,passed=not violations,quality_gate=False)


def measure(spec,train,val,bank,device='cuda',*,shared_preparation_seconds=0.):
    if not math.isfinite(shared_preparation_seconds) or shared_preparation_seconds<0: raise ValueError('Invalid shared preparation timing')
    root=Path(spec['root'])/'setup'; ix=miniature_rows(train); vx=miniature_rows(val)
    tr=Subset(train,ix); va=Subset(val,vx)
    tiny=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=bank.targets[ix],manifest=bank.manifest)
    records={}; representatives={}; paths={}
    for task in spec['remaining']:
        key=path_key(task)
        if key in representatives: paths[task]=representatives[key]; continue
        representatives[key]=task; paths[task]=task
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter(); source=inputs.candidates(spec,task,train,device)
        preparation=shared_preparation_seconds+time.perf_counter()-began; memory_before=peaks(device)
        if source is not None and hasattr(source,'capacity'):
            write_immutable_json(root/('capacity_'+task+'.json'),artifact('CAPACITY',parents=dict(spec=spec['content_hash']),task=task,**source.capacity))
        wrapped=None if source is None else inputs.SubsetCandidates(source,ix)
        try:
            warm=training.fit(root/('warm_'+task),spec,task,tr,va,tiny,candidates=wrapped,device=device,technical=True)
            gc.collect()
            report=training.fit(root/('mini_'+task),spec,task,tr,va,tiny,candidates=wrapped,device=device,technical=True)
        except BaseException:
            if hasattr(source,'close'): source.close()
            raise
        memory=peaks(device)
        for k in ('gpu_peak_bytes','peak_rss_bytes'):
            vs=[v for v in (memory.get(k),memory_before.get(k)) if v is not None]; memory[k]=max(vs) if vs else None
        children=source.child_pss_bytes() if hasattr(source,'child_pss_bytes') else 0
        memory['sampler_child_pss_bytes']=children
        if memory.get('peak_rss_bytes') is not None: memory['peak_rss_bytes']+=children
        record=resource_record(spec,task,report,warm,memory,gpu_identity(),preparation)
        write_immutable_json(root/('resource_'+task+'.json'),record)
        print(f'NEXT_SUPERVISION resource {task}: step={record["train_step_seconds"]:.3f}s '
            f'limit={record["minutes"]}min host={memory.get("peak_rss_bytes")} '
            f'phases={record["batch_phase_seconds"]}',flush=True)
        if not record['passed']:
            if hasattr(source,'close'): source.close()
            raise RuntimeError('Measured Group A resource envelope exceeded: '+str(record['violations'])+'; preserve diagnostics')
        records[task]=record
        if hasattr(source,'close'): source.close()
        del source,wrapped; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    result=artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash']),paths=paths,measurements=records,
        minutes={t:records[p]['minutes'] for t,p in paths.items()},environment=installed_environment(),gpu=gpu_identity(),
        quality_gate=False,science_dependencies='none',measured_universe='full_TRAIN',technical_weights_discarded=True)
    write_immutable_json(root/'acceptance.json',result)
    return result


def accepted(spec):
    root=Path(spec['root'])/'setup'; a=load_json(root/'acceptance.json'); validate(a,'ACCEPTANCE',dict(spec=spec['content_hash']))
    if (set(a['paths'])!=set(spec['remaining']) or set(a['minutes'])!=set(spec['remaining'])
            or a['environment']!=spec['reference']['environment'] or a['gpu']!=spec['qualification']['acceptance']['gpu']
            or a['quality_gate'] is not False or a['science_dependencies']!='none' or a['measured_universe']!='full_TRAIN'
            or a['technical_weights_discarded'] is not True or set(a['measurements'])!=set(a['paths'].values())):
        raise ValueError('Acceptance coverage/environment differs')
    parity=load_json(root/'baseline_parity.json'); validate(parity,'BASELINE_PARITY',dict(spec=spec['content_hash']))
    if (parity['environment']!=a['environment'] or set(parity['checks'])!={'C0','C1','P1','G1','J1','J5'}
            or parity['installed_weaver'] is not True or parity['quality_gate'] is not False
            or any(v['max_abs']!=0 or v['prediction_max_abs']!=0 or not v['native_tensors_equal'] for v in parity['checks'].values())):
        raise ValueError('Native baseline parity differs')
    for task,path in a['paths'].items():
        if path_key(task)!=path_key(path): raise ValueError('Resource path does not match task support')
    for path in a['measurements']:
        capacity=load_json(root/('capacity_'+path+'.json')); validate(capacity,'CAPACITY',dict(spec=spec['content_hash']))
        if (capacity['task']!=path or capacity['rows']!=spec['counts']['train']
                or sum(capacity['class_counts'])!=spec['counts']['train']): raise ValueError('Capacity population differs')
        if setting(path)['family'] in ('P','G') and capacity.get('sampler_workers')!=spec['candidate_workers']:
            raise ValueError('Measured CPU sampler-worker support differs')
        report,_=training.validate_fit(root/('mini_'+path),spec,path,technical=True)
        warm,_=training.validate_fit(root/('warm_'+path),spec,path,technical=True)
        r=a['measurements'][path]
        if (r!=resource_record(spec,path,report,warm,r['memory'],a['gpu'],r['preparation_seconds'])
                or load_json(root/('resource_'+path+'.json'))!=r or not r['passed']
                or any(a['minutes'][task]!=r['minutes'] for task,p in a['paths'].items() if p==path)):
            raise ValueError('Measured resource evidence differs')
    return a
