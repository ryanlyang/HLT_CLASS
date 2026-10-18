"""Authenticated gate fits/exports, warm starts and development-only reporting."""
from pathlib import Path
import os
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,
    write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_fullsim import regular_cache,alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from hlt_classification.luka_attention.worker import data,execution
from . import banks,training,campaign
from .contracts import artifact,validate,recipe,anchor_node,input_label
from .model import new_model


def checkpoint(gate,spec,name):
    if name=='ANCHOR':
        refs=gate['anchor']['evidence']['models']['COARSE_U100']
        for ref in (refs['checkpoint'],refs['report']): validate_file_ref(ref)
        return Path(refs['checkpoint']['path']),dict(checkpoint=refs['checkpoint']['sha256'],report=refs['report']['sha256'])
    root=Path(spec['root'])/('train_'+name)
    return root/'selected.pt',dict(checkpoint=file_ref(root/'selected.pt')['sha256'],
                                 report=file_ref(root/'training_report.json')['sha256'])


def report_valid(report,node,bank_hash,warm_parents,prepared,rows,*,technical=False):
    validate(report,'TRAINING_REPORT',dict(recipe=recipe()['content_hash'],prepared=prepared,
                                          bank=bank_hash,**warm_parents))
    history=report['validation_history']; passes=1 if technical else 30
    if (report['node']!=node or report['scientific_fit'] is not (not technical)
            or report['passes']!=passes or len(history)!=passes
            or [r['pass_number'] for r in history]!=list(range(1,passes+1))
            or not report['selected_weights_restored'] or report['warm_start_policy']!=recipe()['warm_start']
            or any(r['validation']['rows']!=rows or r['backbone_frozen']!=(r['pass_number']<=node['freeze_epochs']) for r in history)):
        raise ValueError('Fit population/budget/mode differs')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Selected checkpoint differs')


def fit(directory,gate,node,tr,val,tr_context,val_context,bank,state,*,warm_parents,device,technical=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if any((directory/f).exists() for f in ('selected.pt','training_report.json','validation_predictions.npz')):
        raise FileExistsError('Preserve partial fit; no automatic overwrite')
    model=new_model(node,state)
    report,selected=training.train(model,tr,val,tr_context,val_context,bank=bank,node=node,
                                  warm_parents=warm_parents,device=device,technical=technical)
    report_valid(report,node,bank.manifest['content_hash'],warm_parents,tr.foundation_sha256,len(val),technical=technical)
    save_state(directory/'selected.pt',selected)
    restored=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    if set(restored)!=set(selected) or any(not torch.equal(restored[k],selected[k]) for k in selected):
        raise ValueError('Checkpoint roundtrip differs')
    # A fresh deployment adapter, not the training instance's retained context.
    deployed=new_model(node,restored).to(device).eval()
    model.eval(); ix=np.arange(min(4,len(val))); raw=val.batch(ix)
    ctx=None if val_context is None else val_context.batch(ix)
    with torch.inference_mode():
        a=banks.forward(model,raw,ctx,device); b=banks.forward(deployed,raw,ctx,device)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    del model
    begin=time.monotonic(); p=banks.predict(deployed,val,val_context,device=device)
    if evaluate_probabilities(val.labels,p)!=report['validation']:
        raise ValueError('Selected validation readback differs')
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=p)
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json',report)
    write_immutable_json(directory/'costs.json',dict(parameters=sum(p.numel() for p in deployed.parameters()),
        fit_seconds=report['runtime_seconds'],readback_seconds=time.monotonic()-begin,
        deployment_input=input_label(node),checkpoint_roundtrip=True,external_context_required=node['gate_mode']=='privileged'))
    return report,deployed


def completed(spec,gate,name,seen=None,*,technical=False):
    seen={} if seen is None else seen
    if name in seen: return seen[name]
    task=next((t for t in gate['tasks'] if t['task_id']==name),None)
    if task is None: raise ValueError('Unknown gate task')
    root=Path(spec['root']); path=root/name/'receipt.json'
    if not path.is_file(): return None
    r=load_json(path); validate(r,'TASK',dict(spec=spec['content_hash']))
    deps={}
    for dep in task['dependencies']:
        d=completed(spec,gate,dep,seen,technical=technical)
        if d is None: raise ValueError('Missing dependency receipt')
        deps[dep]=d['content_hash']
    if r['task_id']!=name or r['dependencies']!=deps or r['technical'] is not technical:
        raise ValueError('Task lineage/mode differs')
    paths=[f['path'] for f in r['outputs']]
    if len(paths)!=len(set(paths)) or any(not p.startswith(name+'/') for p in paths):
        raise ValueError('Output path coverage differs')
    for ref in r['outputs']: validate_file_ref(ref,root=root)
    if task['kind']=='train':
        required={name+'/'+f for f in ('selected.pt','training_report.json','validation_predictions.npz','costs.json')}
        if not required<=set(paths): raise ValueError('Missing bound fit payload')
        node=next(n for n in gate['nodes'] if n['node_id']==task['node_id'])
        bank=load_json(root/('export_'+node['teacher'])/'bank/manifest.json')
        _,warm=checkpoint(gate,spec,node['warm_start'])
        warm={'warm_checkpoint':warm['checkpoint'],'warm_report':warm['report']}
        report=load_json(root/name/'training_report.json')
        report_valid(report,node,bank['content_hash'],warm,gate['anchor']['evidence']['prepared'],gate['counts']['validation'],technical=technical)
        p=load_npz_arrays(root/name/'validation_predictions.npz')
        if evaluate_probabilities(p['labels'],p['probabilities'])!=report['validation']:
            raise ValueError('Bound predictions differ from selected report')
    elif task['kind']=='export':
        if name+'/bank/manifest.json' not in paths: raise ValueError('Missing bound bank')
        m=load_json(root/name/'bank/manifest.json'); _,parents=checkpoint(gate,spec,task['node_id'])
        validate(m,'BANK',parents)
        if m['teacher']!=task['node_id'] or m['factors']!=(task['node_id']=='PRIV_D000'):
            raise ValueError('Bank kind/teacher differs')
        for ref in [m['population']]+[s['payload'] for s in m['shards']]:
            if dict(ref,path=name+'/bank/'+ref['path']) not in r['outputs']:
                raise ValueError('Unbound bank payload')
    else:
        if name+'/results.json' not in paths: raise ValueError('Missing result artifact')
        result=load_json(root/name/'results.json'); validate(result,'RESULTS',dict(spec=spec['content_hash']))
        if [v['node_id'] for v in result['rows']]!=['M0HLT','OFFLINE','DIRECT_D000','COARSE_U100']+[n['node_id'] for n in gate['nodes']]:
            raise ValueError('Missing result rows')
        if any(v['validation'] is None for v in result['rows']): raise ValueError('Incomplete results')
    seen[name]=r
    return r


def execute_task(spec,gate,name,*,device='cuda',technical=False,before_publish=None):
    task=next(t for t in gate['tasks'] if t['task_id']==name)
    if task['kind']=='aggregate': raise ValueError('Aggregate is not a technical fit')
    deps={}
    for dep in task['dependencies']:
        row=completed(spec,gate,dep,technical=technical)
        if row is None: raise ValueError('Missing teacher/warm start')
        deps[dep]=row['content_hash']
    root=Path(spec['root']); directory=root/name; directory.mkdir(exist_ok=False)
    log=PhaseLog(directory,lambda:memory_snapshot(torch))
    node=anchor_node(gate['nodes']) if task['node_id']=='ANCHOR' else next(n for n in gate['nodes'] if n['node_id']==task['node_id'])
    started=time.monotonic()
    with log.phase('load_authenticated_views'):
        tr=data(gate,'train',node['coordinate'])
        ctx=None if node['context_coordinate'] is None else data(gate,'train',node['context_coordinate'])
        banks.paired(tr,ctx)
        if task['kind']=='train':
            val=data(gate,'validation',node['coordinate'])
            vctx=None if ctx is None else data(gate,'validation',node['context_coordinate'])
            banks.paired(val,vctx)
            path,warm=checkpoint(gate,spec,node['warm_start'])
            state=torch.load(path,map_location='cpu',weights_only=True)
            _,parents=checkpoint(gate,spec,node['teacher'])
            bank=banks.Bank(root/('export_'+node['teacher'])/'bank',tr,parents=parents,
                            teacher=node['teacher'],require_factors=bool(node['gate_weight']))
    loading=time.monotonic()-started
    with log.phase('fit' if task['kind']=='train' else 'export'):
        started=time.monotonic()
        if task['kind']=='train':
            warm={'warm_checkpoint':warm['checkpoint'],'warm_report':warm['report']}
            report,model=fit(directory,gate,node,tr,val,ctx,vctx,bank,state,warm_parents=warm,device=device,technical=technical)
            epoch=report['validation_history'][-1]
            epoch_seconds=epoch['train_seconds']+epoch['validation_seconds']
            measurement=dict(node_id=node['node_id'],kind='train',load_seconds=loading,
                epoch_seconds=epoch_seconds,other_seconds=max(0.,time.monotonic()-started-report['runtime_seconds']),
                frozen_backbone=bool(node['freeze_epochs']))
        else:
            path,parents=checkpoint(gate,spec,node['node_id'])
            model=new_model(node,torch.load(path,map_location='cpu',weights_only=True))
            banks.export(directory/'bank',model,tr,ctx,parents=parents,node=node,device=device)
            banks.Bank(directory/'bank',tr,parents=parents,teacher=node['node_id'],require_factors=node['node_id']=='PRIV_D000')
            measurement=dict(node_id=node['node_id'],kind='export',load_seconds=loading,
                             export_seconds=time.monotonic()-started)
        del model
    if before_publish is not None: before_publish()
    row=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=name,
        job_id=os.environ.get('SLURM_JOB_ID','local'),dependencies=deps,technical=technical,measurement=measurement,
        outputs=[file_ref(p,root=root) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'receipt.json',row)
    return completed(spec,gate,name,technical=technical)


def result_rows(spec,gate):
    rows=[]; seen={}; population=None
    meta=regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
    for name in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100'):
        refs=gate['anchor']['evidence']['models'][name]
        for ref in refs.values(): validate_file_ref(ref)
        report=load_json(refs['report']['path']); diag=load_json(refs['diagnostics']['path'])
        rows.append(dict(node_id=name,origin='parent_fixed30_reference',validation=report['validation'],
            original_validation=diag['original_validation'],input='HLT ONLY' if name in ('M0HLT','DIRECT_D000') else 'OFFLINE ORACLE'))
    for node in gate['nodes']:
        receipt=completed(spec,gate,'train_'+node['node_id'],seen)
        row=dict(node_id=node['node_id'],origin='new',validation=None,original_validation=None,input=input_label(node))
        if receipt:
            root=Path(spec['root'])/('train_'+node['node_id'])
            p=load_npz_arrays(root/'validation_predictions.npz')
            if population is None: population=(p['identities'],p['labels'])
            elif not np.array_equal(population[0],p['identities']) or not np.array_equal(population[1],p['labels']):
                raise ValueError('Compared validation populations differ')
            ix=alljets_data.comparison_indices(meta[2],meta[3],p['identities'],p['labels'])
            row.update(validation=evaluate_probabilities(p['labels'],p['probabilities']),
                original_validation=evaluate_probabilities(p['labels'][ix],p['probabilities'][ix]),
                costs=load_json(root/'costs.json'))
        rows.append(row)
    for row in rows:
        row['recovery']={field:None if row[field] is None else recovery(row[field],rows[0][field],rows[1][field])
                         for field in ('validation','original_validation')}
    return rows


def contrasts(rows):
    """Registered comparisons, not an adaptive validation winner search."""
    by_name={r['node_id']:r for r in rows}
    pairs=[('SELF_D000','WARM_D000'),('PRIV_D000','SELF_D000'),
           ('SELF_CONT','SELF_D000')]
    pairs += [(name,'SELF_CONT') for name in ('REPLACE_LOGIT','REPLACE_GATE','REPLACE_STAGED')]
    pairs += [('REPLACE_GATE','REPLACE_LOGIT'),('REPLACE_STAGED','REPLACE_GATE')]
    return {field:{f'{a} minus {b}':None if by_name[a][field] is None or by_name[b][field] is None else
                   {key:by_name[a][field][key]-by_name[b][field][key] for key in ('accuracy','macro_ovr_auc')}
                   for a,b in pairs} for field in ('validation','original_validation')}


def run(spec,name):
    gate,measured=campaign.check_spec(spec)
    task=next((t for t in gate['tasks'] if t['task_id']==name),None)
    if task is None: raise ValueError('Unregistered task; no test capability')
    done=completed(spec,gate,name)
    if done: return done
    if task['kind']!='aggregate':
        execution(gate,measured)
        def final_check():
            execution(gate,measured)
            campaign.check_spec(spec)
        result=execute_task(spec,gate,name,before_publish=final_check)
    else:
        deps={}
        for dep in task['dependencies']:
            r=completed(spec,gate,dep)
            if r is None: raise ValueError('All scientific arms must finish regardless of quality')
            deps[dep]=r['content_hash']
        directory=Path(spec['root'])/name; directory.mkdir(exist_ok=False)
        rows=result_rows(spec,gate)
        write_immutable_json(directory/'results.json',artifact('RESULTS',parents=dict(spec=spec['content_hash']),
            rows=rows,contrasts=contrasts(rows),single_seed=True,
            interpretation='Exploratory development only; privileged rows are oracles'))
        campaign.check_spec(spec)
        result=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=name,job_id=os.environ.get('SLURM_JOB_ID'),
            technical=False,dependencies=deps,outputs=[file_ref(directory/'results.json',root=Path(spec['root']))])
        write_immutable_json(directory/'receipt.json',result)
    campaign.check_spec(spec)
    return completed(spec,gate,name)
