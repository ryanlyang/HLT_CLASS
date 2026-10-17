"""Authenticated training/export/report workers; no source or parent writes."""
from pathlib import Path
import os
import time
import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.luka_fullsim import regular_cache, alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot, require_backend
from . import campaign, banks, training
from .contracts import artifact, validate, recipe
from .model import new_model


def data(gate, role, coordinate, authenticated=None):
    if role not in ('train','validation') or coordinate not in ('U100','D066','D033','D000'):
        raise PermissionError('Only registered D-ladder TRAIN/validation views allowed')
    root = gate['anchor']['evidence']['prepared_root']
    meta = regular_cache.metadata(root) if authenticated is None else authenticated
    if (meta[0]['content_hash']!=gate['anchor']['evidence']['cache']
            or meta[1]['content_hash']!=gate['anchor']['evidence']['prepared']):
        raise ValueError('Imported cache lineage differs')
    result = regular_cache.load(root, role=role, coordinate=coordinate,
                                max_ram_bytes=int(gate['memory_mb']*1024**2*.3), authenticated=meta)
    if len(result)!=gate['counts'][role]:
        raise ValueError('Imported cache row counts differ')
    return result


def execution(gate, measured=None):
    require_backend()
    job,cpus,memory = allocation(gate['execution_site'])
    if (cpus,memory)!=(gate['cpus'],gate['memory_mb']):
        raise PermissionError('Attention allocation differs')
    if measured and (measured['gpu']!=gpu_identity() or measured['environment']!=installed_environment()):
        raise PermissionError('Measured environment/GPU changed')
    return job


def report_valid(report, node, bank_hash, prepared, validation_rows, *, technical=False):
    validate(report,'TRAINING_REPORT',dict(recipe=recipe()['content_hash'],prepared=prepared,bank=bank_hash))
    history=report['validation_history']
    if (report['node']!=node or report['scientific_fit'] is not (not technical)
            or report['passes']!=(1 if technical else 30) or len(history)!=report['passes']
            or [r['pass_number'] for r in history]!=list(range(1,len(history)+1))
            or not report['selected_weights_restored']
            or any(r['validation']['rows']!=validation_rows for r in history)):
        raise ValueError('Training report population/recipe/scientific status differs')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Wrong selected checkpoint report')


def fit(directory, gate, node, train_cache, val, bank, *, device, technical=False):
    model=new_model(node)
    report,state=training.train(model,train_cache,val,bank=bank,node=node,device=device,technical=technical)
    report_valid(report,node,bank.manifest['content_hash'],train_cache.foundation_sha256,len(val),technical=technical)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if any((directory/f).exists() for f in ('selected.pt','training_report.json','validation_predictions.npz')):
        raise FileExistsError('Preserve partial fit; no automatic restart')
    save_state(directory/'selected.pt',state)
    restored=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    if set(restored)!=set(state) or any(not torch.equal(restored[n],state[n]) for n in state):
        raise ValueError('Checkpoint roundtrip differs')
    model.load_state_dict(restored,strict=True)
    # Native HLT deployment has no readout/target capability.
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    native=DelphesParticleTransformer().to(device)
    native.load_state_dict(restored,strict=True)
    native.eval(); model.eval(); model.capture_attention=False
    raw=val.batch(np.arange(min(4,len(val))))
    from hlt_classification.jetclass2_delphes.runner import _forward
    with torch.inference_mode():
        torch.testing.assert_close(_forward(native,raw,device,bf16=str(device).startswith('cuda')),
                                   _forward(model,raw,device,bf16=str(device).startswith('cuda')),rtol=0,atol=0)
    native_parameters=sum(p.numel() for p in native.parameters())
    with torch.inference_mode():
        for _ in range(3): _forward(native,raw,device,bf16=str(device).startswith('cuda'))
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        latency_start=time.monotonic()
        for _ in range(20): _forward(native,raw,device,bf16=str(device).startswith('cuda'))
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        batch_ms=(time.monotonic()-latency_start)*1000/20
    del native
    prediction_start=time.monotonic()
    arrays=dict(identities=val.identities,labels=val.labels,
                probabilities=predict(model,val,device=device,batch_size=128))
    costs=dict(parameters=native_parameters,device=str(device),training_validation_seconds=report['runtime_seconds'],
               training_validation_gpu_hours=report['runtime_seconds']/3600 if str(device).startswith('cuda') else None,
               selected_prediction_seconds=time.monotonic()-prediction_start,selected_prediction_jets=len(val),
               native_batch_latency_ms=batch_ms,latency_batch_rows=len(raw['labels']),
               latency_padded_width=raw['mask'].shape[-1],latency_scope='20 warmed label-free forwards; input transfer included')
    recomputed=evaluate_probabilities(arrays['labels'],arrays['probabilities'])
    if recomputed!=report['validation']:
        raise ValueError('Selected validation checkpoint readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json',report)
    write_immutable_json(directory/'costs.json',costs)
    return report,model


def checkpoint(gate, spec, name):
    if name=='ANCHOR':
        row=gate['anchor']['evidence']['models']['COARSE_U100']
        return Path(row['checkpoint']['path']),dict(checkpoint=row['checkpoint']['sha256'],report=row['report']['sha256'])
    directory=Path(spec['root'])/('train_'+name)
    return directory/'selected.pt',dict(checkpoint=file_ref(directory/'selected.pt')['sha256'],
                                         report=file_ref(directory/'training_report.json')['sha256'])


def completed(spec,gate,name,seen=None):
    seen={} if seen is None else seen
    if name in seen: return seen[name]
    task=next((t for t in gate['tasks'] if t['task_id']==name),None)
    if task is None: raise ValueError('Unregistered attention task')
    root=Path(spec['root']); path=root/name/'receipt.json'
    if not path.is_file(): return None
    row=load_json(path)
    validate(row,'TASK',dict(spec=spec['content_hash']))
    deps={}
    for dep in task['dependencies']:
        value=completed(spec,gate,dep,seen)
        if value is None: raise ValueError('Missing completed dependency')
        deps[dep]=value['content_hash']
    if row['task_id']!=name or row['dependencies']!=deps:
        raise ValueError('Task/dependency receipt mismatch')
    paths=[r['path'] for r in row['outputs']]
    if len(paths)!=len(set(paths)) or any(not p.startswith(name+'/') for p in paths):
        raise ValueError('Output paths escape task')
    for ref in row['outputs']: validate_file_ref(ref,root=root)
    if task['kind']=='train':
        required={name+'/'+f for f in ('selected.pt','training_report.json','validation_predictions.npz','costs.json')}
        if not required<=set(paths): raise ValueError('Missing bound training outputs')
        node=next(n for n in gate['nodes'] if n['node_id']==task['node_id'])
        bank=load_json(root/('export_'+node['teacher'])/'bank/manifest.json')
        report=load_json(root/name/'training_report.json')
        report_valid(report,node,bank['content_hash'],gate['anchor']['evidence']['prepared'],gate['counts']['validation'])
        arrays=load_npz_arrays(root/name/'validation_predictions.npz')
        if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:
            raise ValueError('Prediction/report metrics differ')
    elif task['kind']=='export':
        if name+'/bank/manifest.json' not in paths: raise ValueError('Missing bound bank')
        bank=load_json(root/name/'bank/manifest.json')
        _,parents=checkpoint(gate,spec,task['node_id'])
        validate(bank,'BANK',parents)
        expected_attention=task['node_id']=='ANCHOR' or task['node_id'].startswith('ATTN_')
        if bank['teacher']!=task['node_id'] or bank['attention'] is not expected_attention:
            raise ValueError('Bank teacher/arm differs')
        required=[bank['population']]+[s['payload'] for s in bank['shards']]
        for ref in required:
            bound=dict(ref,path=name+'/bank/'+ref['path'])
            if bound not in row['outputs']: raise ValueError('Unbound bank payload')
    else:
        if name+'/results.json' not in paths:
            raise ValueError('Missing aggregate output')
        result=load_json(root/name/'results.json')
        validate(result,'RESULTS',dict(spec=spec['content_hash']))
        expected=['M0HLT','OFFLINE','DIRECT_D000','COARSE_U100']+[n['node_id'] for n in gate['nodes']]
        if ([r['node_id'] for r in result['rows']]!=expected
                or any(r['validation'] is None or r['original_validation'] is None for r in result['rows'])
                or result['primary_contrast']!='ATTN_D000 minus LOGIT_D000' or result['single_seed'] is not True):
            raise ValueError('Aggregate coverage/contrast differs')
        by_name={r['node_id']:r for r in result['rows']}
        expected_contrasts={field:{metric:by_name['ATTN_D000'][field][metric]-by_name['LOGIT_D000'][field][metric]
                            for metric in ('accuracy','macro_ovr_auc','cross_entropy')}
                          for field in ('validation','original_validation')}
        if result['contrasts']!=expected_contrasts:
            raise ValueError('Aggregate contrasts differ')
    seen[name]=row
    return row


def result_rows(spec,gate):
    rows=[]; seen={}
    meta=regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
    for name in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100'):
        refs=gate['anchor']['evidence']['models'][name]
        for ref in refs.values(): validate_file_ref(ref)
        report=load_json(refs['report']['path']); diag=load_json(refs['diagnostics']['path'])
        rows.append(dict(node_id=name,origin='parent_fixed30',validation=report['validation'],
                         original_validation=diag['original_validation']))
    for n in gate['nodes']:
        receipt=completed(spec,gate,'train_'+n['node_id'],seen)
        row=dict(node_id=n['node_id'],origin='new',validation=None,original_validation=None)
        if receipt:
            p=load_npz_arrays(Path(spec['root'])/('train_'+n['node_id'])/'validation_predictions.npz')
            ix=alljets_data.comparison_indices(meta[2],meta[3],p['identities'],p['labels'])
            row.update(validation=evaluate_probabilities(p['labels'],p['probabilities']),
                       original_validation=evaluate_probabilities(p['labels'][ix],p['probabilities'][ix]),
                       costs=load_json(Path(spec['root'])/('train_'+n['node_id'])/'costs.json'))
        rows.append(row)
    for row in rows:
        row['recovery']={field:None if row[field] is None else recovery(row[field],rows[0][field],rows[1][field])
                         for field in ('validation','original_validation')}
    return rows


def run(spec,name):
    gate,measured=campaign.check_spec(spec)
    task=next((t for t in gate['tasks'] if t['task_id']==name),None)
    if task is None: raise ValueError('Unregistered task; no test capability')
    seen={}
    previous=completed(spec,gate,name,seen)
    if previous: return previous
    deps={}
    for dep in task['dependencies']:
        row=completed(spec,gate,dep,seen)
        if row is None: raise ValueError('Missing teacher/dependency')
        deps[dep]=row['content_hash']
    job=execution(gate,measured) if task['kind']!='aggregate' else os.environ.get('SLURM_JOB_ID')
    root=Path(spec['root']); directory=root/name
    directory.mkdir(exist_ok=False)
    log=PhaseLog(directory,lambda:memory_snapshot(torch))
    if task['kind']=='aggregate':
        rows=result_rows(spec,gate)
        if any(r['validation'] is None for r in rows): raise ValueError('Every fit required regardless of quality')
        by_name={r['node_id']:r for r in rows}
        contrasts={field:{metric:by_name['ATTN_D000'][field][metric]-by_name['LOGIT_D000'][field][metric]
                          for metric in ('accuracy','macro_ovr_auc','cross_entropy')}
                   for field in ('validation','original_validation')}
        result=artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,
                        primary_contrast='ATTN_D000 minus LOGIT_D000',contrasts=contrasts,single_seed=True)
        write_immutable_json(directory/'results.json',result)
    else:
        anchor=task['node_id']=='ANCHOR'
        node=(dict(gate['nodes'][0],coordinate='U100',node_id='ANCHOR') if anchor else
              next(n for n in gate['nodes'] if n['node_id']==task['node_id']))
        with log.phase('load_train_cache'):
            tr=data(gate,'train',node['coordinate'])
        if task['kind']=='train':
            with log.phase('load_validation_and_teacher'):
                val=data(gate,'validation',node['coordinate'])
                _,parents=checkpoint(gate,spec,node['teacher'])
                bank=banks.Bank(root/('export_'+node['teacher'])/'bank',tr,parents=parents,
                                teacher=node['teacher'],require_attention=node['branch']=='ATTN')
            with log.phase('scientific_fit'):
                fitted,model=fit(directory,gate,node,tr,val,bank,device='cuda')
                del model
        else:
            with log.phase('teacher_export'):
                path,parents=checkpoint(gate,spec,task['node_id'])
                model=new_model(node)
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                banks.export(directory/'bank',model,tr,parents=parents,teacher=task['node_id'],
                              attention=anchor or node['branch']=='ATTN',device='cuda')
                banks.Bank(directory/'bank',tr,parents=parents,teacher=task['node_id'],
                           require_attention=anchor or node['branch']=='ATTN')
        execution(gate,measured)
    campaign.check_spec(spec)
    if any(completed(spec,gate,dep)['content_hash']!=h for dep,h in deps.items()):
        raise ValueError('Dependency changed during execution')
    row=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=name,job_id=job,dependencies=deps,
                  outputs=[file_ref(p,root=root) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'receipt.json',row)
    return completed(spec,gate,name)
