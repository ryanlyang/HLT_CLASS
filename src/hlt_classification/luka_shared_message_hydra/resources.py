"""Warmed production miniatures; measured limits independent of model quality."""
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
from hlt_classification.luka_cross_message_ce_hydra.worker import miniature_rows as donor_miniature_rows
from .contracts import TASKS,ARMS,artifact,validate
from . import training,inputs


def miniature_rows(cache):
    """Keep the largest128 first and a non-singleton partial native BN batch."""
    indices=donor_miniature_rows(cache)
    if len(indices)%128==1:
        selected=set(indices.tolist())
        extra=next((i for i in range(len(cache)) if i not in selected),None)
        if extra is not None:
            indices=np.append(indices,extra).astype(np.int64)
        else:
            # Tiny local populations may have no unused row. Retain the full
            # largest128 and all classes, removing two redundant tail rows.
            for _ in range(2):
                labels=cache.labels[indices]
                position=next((i for i in reversed(range(128,len(indices)))
                    if np.count_nonzero(labels==labels[i])>1),None)
                if position is None: raise ValueError('Need full and non-singleton partial miniature batches')
                indices=np.delete(indices,position)
    if len(indices)<=128 or not 2<=len(indices)%128<=127:
        raise ValueError('Need full and non-singleton partial miniature batches')
    return indices


def resource_record(spec,task,report,warmup,memory,gpu,preparation_seconds):
    from .campaign import registration
    if spec['resource_policy']!=registration()['resource_policy']: raise ValueError('Resource policy differs')
    for r in (report,warmup):
        if r['task']!=task or not r['technical'] or r['epochs']!=1 or r['parents']['spec']!=spec['content_hash']:
            raise ValueError('Expected task-bound production miniatures')
    if report['counts']!=warmup['counts']: raise ValueError('Warmup shape mismatch')
    row=report['validation_history'][0]
    times=row['batch_seconds']; sizes=row['batch_sizes']
    if (len(times)!=len(sizes) or not times or any(not math.isfinite(t) or t<=0 for t in times)
            or 128 not in sizes or not any(2<=s<128 for s in sizes) or 1 in sizes
            or not math.isfinite(preparation_seconds) or preparation_seconds<0
            or not math.isfinite(row['validation_seconds']) or row['validation_seconds']<=0):
        raise ValueError('Need finite warmed full/partial-batch timing')
    perstep=max(t for t,s in zip(times,sizes) if s==128)
    vstep=row['validation_seconds']/math.ceil(report['counts']['validation']/128)
    estimate=30*(perstep*math.ceil(spec['counts']['train']/128)+vstep*math.ceil(spec['counts']['validation']/128))
    policy=spec['resource_policy']
    minutes=max(30,math.ceil((policy['safety_multiplier']*estimate+preparation_seconds+policy['fixed_overhead_seconds'])/60))
    limits=dict(minutes=spec['fit_max_minutes'],gpu_peak_bytes=int(gpu['total_memory_bytes']*policy['memory_fraction']),
        peak_rss_bytes=int(spec['memory_mb']*1024**2*policy['memory_fraction']))
    measured=dict(minutes=minutes,gpu_peak_bytes=memory['gpu_peak_bytes'],peak_rss_bytes=memory.get('peak_rss_bytes'))
    violations=[]
    for key,n in measured.items():
        if n is None and key=='peak_rss_bytes': continue  # Non-POSIX local fixture only.
        if not isinstance(n,(int,float)) or not math.isfinite(n) or n<0: raise ValueError('Invalid memory measurement')
        if n>limits[key]: violations.append(dict(resource=key,measured=n,limit=limits[key]))
    return artifact('RESOURCE_CHECK',parents=dict(spec=spec['content_hash'],report=report['content_hash'],warmup=warmup['content_hash']),
        task=task,minutes=minutes,memory=memory,gpu=gpu,limits=limits,violations=violations,passed=not violations,
        preparation_seconds=preparation_seconds,train_step_seconds=perstep,validation_step_seconds=vstep,
        resource_policy=policy,memory_scope='preparation_and_warmed_fit',quality_gate=False)


def measure(spec,train,val,oracle,bank,preparation_seconds,device='cuda'):
    root=Path(spec['root'])/'setup'; ix=miniature_rows(train); vx=miniature_rows(val)
    tr=Subset(train,ix); va=Subset(val,vx); ot=Subset(oracle,ix)
    tiny=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=bank.targets[ix],manifest=bank.manifest)
    records={}; preparation_memory=peaks(device)
    for task in TASKS:
        cfg=ARMS[task]; teacher=inputs.frozen_teacher(spec,device) if cfg['mode']=='frozen' else None
        extra=dict(oracle_train=None if cfg['mode']=='control' else ot,teacher=teacher)
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        warm=training.fit(root/('warm_'+task),spec,task,tr,va,tiny,device=device,technical=True,**extra)
        gc.collect()
        report=training.fit(root/('mini_'+task),spec,task,tr,va,tiny,device=device,technical=True,**extra)
        memory=peaks(device)
        for key in ('gpu_peak_bytes','peak_rss_bytes'):
            samples=[v for v in (memory.get(key),preparation_memory.get(key)) if v is not None]
            memory[key]=max(samples) if samples else None
        record=resource_record(spec,task,report,warm,memory,gpu_identity(),preparation_seconds)
        write_immutable_json(root/('resource_'+task+'.json'),record)
        if not record['passed']: raise RuntimeError('Measured resource envelope exceeded: '+str(record['violations'])+'; preserve diagnostics')
        records[task]=record; del teacher; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    value=artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash']),measurements=records,
        minutes={task:r['minutes'] for task,r in records.items()},environment=installed_environment(),gpu=gpu_identity(),quality_gate=False)
    write_immutable_json(root/'acceptance.json',value)
    return value


def accepted(spec):
    root=Path(spec['root'])/'setup'; value=load_json(root/'acceptance.json')
    validate(value,'ACCEPTANCE',dict(spec=spec['content_hash']))
    if (set(value['minutes'])!=set(TASKS) or set(value['measurements'])!=set(TASKS)
            or value['quality_gate'] is not False or value['environment']!=spec['reference']['environment']
            or value['gpu']!=spec['qualification']['acceptance']['gpu']): raise ValueError('Acceptance coverage/environment differs')
    for task in TASKS:
        r=value['measurements'][task]
        report,_=training.validate_fit(root/('mini_'+task),spec,task,technical=True)
        warm,_=training.validate_fit(root/('warm_'+task),spec,task,technical=True)
        expected=resource_record(spec,task,report,warm,r['memory'],value['gpu'],r['preparation_seconds'])
        if (r!=expected or load_json(root/('resource_'+task+'.json'))!=r or not r['passed'] or value['minutes'][task]!=r['minutes']):
            raise ValueError('Measured miniature/limit evidence differs')
    return value
