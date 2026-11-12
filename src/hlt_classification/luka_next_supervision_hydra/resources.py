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
    from .reconstruction_followup import SETTING_TASKS,execution_key
    if task in SETTING_TASKS: return execution_key(task)
    cfg=setting(task)
    # Same tensor support and compute graph; scalar weights and replicate seeds
    # do not create new execution paths. Source, sampler and interaction do.
    return tuple((k,str(v)) for k,v in cfg.items() if k not in
        ('coefficient','seed_index','ramp_epochs','start_after_epochs','consistency_weight'))


def baseline(spec,train,device):
    raw=train.batch(np.arange(min(128,len(train)))); values={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    errors={}
    for task in baseline_tasks(spec):
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


def baseline_tasks(spec):
    if spec.get('campaign_kind')=='R8_FOLLOWUP_V2': return ('C2','C3','R13','R22','R26','R27','R28')
    if spec.get('scope')=='direct_reconstruction_only': return ('C0','C1','R1','R2','R3','R4')
    if spec.get('scope')=='group_C_only': return ('C0','C1','D1','D5','S1','S5')
    return ('C0','C1','Q1','Q6','V1','V7') if spec.get('scope')=='group_B_only' else ('C0','C1','P1','G1','J1','J5')


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
    if spec.get('scope')=='direct_reconstruction_only' and setting(task)['family']=='R':
        detail=row.get('batch_reconstruction_seconds')
        keys={'decoder_forward_seconds','assignment_seconds','decoder_backward_seconds','native_backward_optimizer_seconds'}
        if (not isinstance(detail,list) or len(detail)!=len(times) or any(set(d)!=keys
                or not np.isfinite(list(d.values())).all() or min(d.values())<0 for d in detail)):
            raise ValueError('Missing exact reconstruction execution timing')
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
        batch_phase_seconds=phases,validation_step_seconds=vstep,limits=limits,violations=violations,passed=not violations,quality_gate=False,
        **(dict(batch_reconstruction_seconds=row['batch_reconstruction_seconds']) if spec.get('scope')=='direct_reconstruction_only' and setting(task)['family']=='R' else {}))


def measure(spec,train,val,bank,device='cuda',*,shared_preparation_seconds=0.):
    if not math.isfinite(shared_preparation_seconds) or shared_preparation_seconds<0: raise ValueError('Invalid shared preparation timing')
    root=Path(spec['root'])/'setup'; ix=miniature_rows(train); vx=miniature_rows(val)
    va=Subset(val,vx)
    records={}; representatives={}; paths={}
    for task in spec['remaining']:
        key=path_key(task)
        if key in representatives: paths[task]=representatives[key]; continue
        representatives[key]=task; paths[task]=task
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        began=time.perf_counter(); source=inputs.candidates(spec,task,train,device)
        preparation=shared_preparation_seconds+time.perf_counter()-began; memory_before=peaks(device)
        selected=ix
        if spec.get('scope')=='direct_reconstruction_only' and source is not None:
            from .reconstruction_targets import miniature_indices
            selected=miniature_indices(source,ix)
        elif spec.get('scope')=='group_B_only':
            from .structured_targets import miniature_indices
            selected=miniature_indices(source,ix)
        elif spec.get('scope')=='group_C_only':
            from .probabilistic_targets import miniature_indices
            selected=miniature_indices(source,ix)
        tr=Subset(train,selected)
        tiny=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=bank.targets[selected],manifest=bank.manifest)
        if source is not None and hasattr(source,'capacity'):
            write_immutable_json(root/('capacity_'+task+'.json'),artifact('CAPACITY',parents=dict(spec=spec['content_hash']),task=task,**source.capacity))
        wrapped=None if source is None else inputs.SubsetCandidates(source,selected)
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
        if spec.get('scope')=='direct_reconstruction_only':
            from .reconstruction_targets import authenticate
            preparation_peak=authenticate(spec)['statistics_peak_host_bytes']
            if preparation_peak is not None:
                memory['peak_rss_bytes']=max(memory.get('peak_rss_bytes') or 0,preparation_peak)
        elif spec.get('scope')=='group_B_only':
            from .structured_targets import authenticate
            preparation_peak=authenticate(spec)['statistics_peak_host_bytes']
            if preparation_peak is not None:
                memory['peak_rss_bytes']=max(memory.get('peak_rss_bytes') or 0,preparation_peak)
        elif spec.get('scope')=='group_C_only':
            from .probabilistic_targets import authenticate
            preparation_peak=authenticate(spec)['statistics_peak_host_bytes']
            if preparation_peak is not None:
                memory['peak_rss_bytes']=max(memory.get('peak_rss_bytes') or 0,preparation_peak)
        record=resource_record(spec,task,report,warm,memory,gpu_identity(),preparation)
        write_immutable_json(root/('resource_'+task+'.json'),record)
        print(f'NEXT_SUPERVISION resource {task}: step={record["train_step_seconds"]:.3f}s '
            f'limit={record["minutes"]}min host={memory.get("peak_rss_bytes")} '
            f'phases={record["batch_phase_seconds"]}',flush=True)
        if not record['passed']:
            if hasattr(source,'close'): source.close()
            raise RuntimeError('Measured supervision resource envelope exceeded: '+str(record['violations'])+'; preserve diagnostics')
        records[task]=record
        if hasattr(source,'close'): source.close()
        del source,wrapped; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    result=artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash']),paths=paths,measurements=records,
        minutes={t:records[p]['minutes'] for t,p in paths.items()},environment=installed_environment(),gpu=gpu_identity(),
        quality_gate=False,science_dependencies='none',measured_universe='full_TRAIN',technical_weights_discarded=True,
        **structured_parent(spec))
    write_immutable_json(root/'acceptance.json',result)
    return result


def accepted(spec):
    root=Path(spec['root'])/'setup'; a=load_json(root/'acceptance.json'); validate(a,'ACCEPTANCE',dict(spec=spec['content_hash']))
    if (set(a['paths'])!=set(spec['remaining']) or set(a['minutes'])!=set(spec['remaining'])
            or a['environment']!=spec['reference']['environment'] or a['gpu']!=spec['qualification']['acceptance']['gpu']
            or a['quality_gate'] is not False or a['science_dependencies']!='none' or a['measured_universe']!='full_TRAIN'
            or a['technical_weights_discarded'] is not True or set(a['measurements'])!=set(a['paths'].values())):
        raise ValueError('Acceptance coverage/environment differs')
    if any(a.get(k)!=v for k,v in structured_parent(spec).items()): raise ValueError('Frozen structured target parent changed')
    parity=load_json(root/'baseline_parity.json'); validate(parity,'BASELINE_PARITY',dict(spec=spec['content_hash']))
    if (parity['environment']!=a['environment'] or set(parity['checks'])!=set(baseline_tasks(spec))
            or parity['installed_weaver'] is not True or parity['quality_gate'] is not False
            or any(v['max_abs']!=0 or v['prediction_max_abs']!=0 or not v['native_tensors_equal'] for v in parity['checks'].values())):
        raise ValueError('Native baseline parity differs')
    for task,path in a['paths'].items():
        if path_key(task)!=path_key(path): raise ValueError('Resource path does not match task support')
    for path in a['measurements']:
        if setting(path)['family']!='C':
            capacity=load_json(root/('capacity_'+path+'.json')); validate(capacity,'CAPACITY',dict(spec=spec['content_hash']))
            if (capacity['task']!=path or capacity['rows']!=spec['counts']['train']
                    or sum(capacity['class_counts'])!=spec['counts']['train']): raise ValueError('Capacity population differs')
            if setting(path)['family'] in ('P','G','Q') and capacity.get('sampler_workers')!=spec['candidate_workers']:
                raise ValueError('Measured CPU sampler-worker support differs')
        report,_=training.validate_fit(root/('mini_'+path),spec,path,technical=True)
        warm,_=training.validate_fit(root/('warm_'+path),spec,path,technical=True)
        r=a['measurements'][path]
        if (r!=resource_record(spec,path,report,warm,r['memory'],a['gpu'],r['preparation_seconds'])
                or load_json(root/('resource_'+path+'.json'))!=r or not r['passed']
                or any(a['minutes'][task]!=r['minutes'] for task,p in a['paths'].items() if p==path)):
            raise ValueError('Measured resource evidence differs')
    return a


def structured_parent(spec):
    if spec.get('scope')=='direct_reconstruction_only':
        from .reconstruction_targets import authenticate
        prepared=authenticate(spec); root=Path(spec['root'])/'setup'
        review=dict(one_time_preparation_seconds=prepared['preparation_seconds'],
            statistics_peak_host_bytes=prepared['statistics_peak_host_bytes'],fixed_output_slots=prepared['maximum'],
            decoder=spec['recipe']['decoder'],storage_budget_bytes=spec['storage_budget_bytes'],
            sources={s:{k:load_json(root/f'reconstruction_{s}.json')[k] for k in
                ('content_hash','rows','tokens','maximum_count','count_quantiles','support')} for s in ('OFFLINE','HLT')})
        return dict(reconstruction_targets=prepared['content_hash'],reconstruction_review=review)
    if spec.get('scope')=='group_C_only':
        from .probabilistic_targets import authenticate
        prepared=authenticate(spec); root=Path(spec['root'])/'setup'
        review=dict(one_time_preparation_seconds=prepared['preparation_seconds'],
            statistics_peak_host_bytes=prepared['statistics_peak_host_bytes'],density=spec['recipe']['density'],
            sets={s:{k:load_json(root/f'set_{s}.json')[k] for k in ('content_hash','rows','tokens','maximum_count','count_quantiles','support')}
                for s in ('OFFLINE','HLT')})
        return dict(probabilistic_targets=prepared['content_hash'],probabilistic_review=review)
    if spec.get('scope')!='group_B_only': return {}
    from .structured_targets import authenticate
    prepared=authenticate(spec); root=Path(spec['root'])/'setup'
    catalog=load_json(root/'question_catalog.json')
    dictionaries={s:load_json(root/('dictionary_'+s+'.json')) for s in ('OFFLINE','HLT')}
    review=dict(catalog=catalog['content_hash'],coverage=catalog['coverage'],
        one_time_preparation_seconds=prepared['preparation_seconds'],statistics_peak_host_bytes=prepared['statistics_peak_host_bytes'],
        dictionaries={s:dict(hash=d['content_hash'],duplicate_centers=d['diagnostic']['duplicate_centers'],
            empty_centers=sum(n==0 for n in d['diagnostic']['occupancy']),mean_squared_distance=d['diagnostic']['mean_squared_distance'])
            for s,d in dictionaries.items()})
    return dict(structured_targets=prepared['content_hash'],structured_review=review)
