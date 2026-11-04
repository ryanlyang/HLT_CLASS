"""Measured real kernels and publication miniatures inside the shared setup."""
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_u100_contrastive.training import peaks
from .contracts import TASKS, TEACHERS, settings, artifact, validate, teacher_task
from .model import new_model
from . import training


class Subset:
    def __init__(self,cache,indices):
        self.cache,self.indices=cache,np.asarray(indices,dtype=np.int64)
        self.identities=cache.identities[self.indices]; self.labels=cache.labels[self.indices]
    def __len__(self): return len(self.indices)
    def __getattr__(self,key): return getattr(self.cache,key)
    def batch(self,indices): return self.cache.batch(self.indices[indices])


def miniature_indices(cache):
    # One example of every class, plus the largest real multiplicities. Includes
    # a full worst-shape batch and a partial final batch, independent of quality.
    sizes=np.concatenate([np.diff(b.offsets) for b in cache.blocks])
    chosen=[int(np.flatnonzero(cache.labels==label)[0]) for label in range(11)]
    largest=np.argsort(-sizes,kind='stable').tolist()
    return np.array(list(dict.fromkeys(largest[:128]+chosen+largest[128:256]))[:267],dtype=np.int64)


def measure(spec,train,val,oracle_train,oracle_val,bank,load_seconds,device='cuda'):
    root=Path(spec['root'])/'setup'; ix=miniature_indices(train); vx=miniature_indices(val)
    tr,va=Subset(train,ix),Subset(val,vx); ot,ov=Subset(oracle_train,ix),Subset(oracle_val,vx)
    tinybank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=bank.targets[ix],manifest=bank.manifest)
    minutes={}; details={}
    for task in TASKS:
        setting=settings(task); isteacher=task in TEACHERS; teacher=None; scale=None; parent=None
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        if setting['mode'] in ('aux','both'):
            name=teacher_task(task); folder=root/('mini_'+name)
            report,_=training.validate_fit(folder,spec,name,technical=True)
            state=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
            teacher=training.deployment(settings(name),state,device).requires_grad_(False)
            scale=torch.tensor(report['statistics']['scale'],device=device); parent=report['content_hash']
        model=new_model(spec['imported'],setting,None)
        report=training.fit(root/('mini_'+task),spec,task,model,ot if isteacher else tr,ov if isteacher else va,
            device=device,bank=tinybank if setting['embedding'] else None,teacher=teacher,
            oracle_train=ot if teacher is not None else None,scale=scale,teacher_parent=parent,technical=True)
        row=report['validation_history'][0]; memory=peaks(device)
        perstep=row['train_seconds']/math.ceil(len(tr)/128)
        vstep=row['validation_seconds']/math.ceil(len(va)/128)
        estimate=30*(perstep*math.ceil(spec['counts']['train']/128)+vstep*math.ceil(spec['counts']['validation']/128))
        # Conservative multiplier, reload/statistics allowance, and measured load.
        requested=math.ceil((2*estimate+2*load_seconds+900)/60)
        if (requested>spec['fit_max_minutes'] or memory['gpu_peak_bytes']>gpu_identity()['total_memory_bytes']*.85
                or (memory.get('peak_rss_bytes') or 0)>spec['memory_mb']*1024**2*.85):
            write_immutable_json(root/'resource_failure.json',artifact('RESOURCE_FAILURE',task=task,minutes=requested,memory=memory))
            raise MemoryError('Measured adapter family exceeds reviewed resource envelope; preserve diagnostics')
        minutes[task]=max(30,requested)
        details[task]=dict(report=report['content_hash'],train_step_seconds=perstep,validation_step_seconds=vstep,memory=memory)
        del model,teacher; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    value=artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash']),minutes=minutes,measurements=details,
        train_rows=len(tr),validation_rows=len(va),environment=installed_environment(),gpu=gpu_identity(),
        miniature='production_kernels_one_epoch_per_task_with_publication_reload',quality_gate=False)
    write_immutable_json(root/'acceptance.json',value)
    return value


def accepted(spec):
    root=Path(spec['root'])/'setup'; value=load_json(root/'acceptance.json')
    validate(value,'ACCEPTANCE',dict(spec=spec['content_hash']))
    if (set(value['minutes'])!=set(TASKS) or set(value['measurements'])!=set(TASKS) or value['quality_gate'] is not False
            or any(type(m) is not int or not 30<=m<=spec['fit_max_minutes'] for m in value['minutes'].values())):
        raise ValueError('Resource acceptance coverage differs')
    for task in TASKS:
        report,_=training.validate_fit(root/('mini_'+task),spec,task,technical=True)
        if value['measurements'][task]['report']!=report['content_hash']: raise ValueError('Miniature report changed')
    return value
